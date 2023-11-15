# generate a 2D vector of wind velocities using a steady state,
# turbulence, and gusts.
#
# setup a find field with a 2D grid of points and return a wind vector
# when a position is queried from the grid.
#
# the wind field is defined by three contibutions:
# 1 steady state wind (same as base model)
# 2 continuous turbulence (TODO)
# 3 discrete gusts (based on 1-cos() theory)
import numpy as np
import pygame.math as math
import random


class Wind:
    def __init__(
        self, width: int, height: int, K_steady_state=5, K_turbulence=0, K_gusts=0
    ):
        self.width = width
        self.height = height
        self.K_steady_state = K_steady_state
        self.K_turbulence = K_turbulence
        self.K_gusts = K_gusts

        self.steady_state_on = True
        self.turbulence_on = True
        self.gusts_on = True

        if self.K_steady_state == 0:
            self.steady_state_on = False
        if self.K_turbulence == 0:
            self.turbulence_on = False
        if self.K_gusts == 0:
            self.gusts_on = False

        self.last_position = math.Vector2(0, 0)
        self.last_wind = math.Vector2(0, 0)
        self.first_call = True
        self.t = 0

        # 2D array to store v,w velocities for every pixel
        self.wind_field = np.zeros((width, height, 2))

        self.steady_state_max = 20

        # Inspried by https://arc.aiaa.org/doi/full/10.2514/1.C036772
        # there is a list of parameters for the discrete gusts
        # V = (wg0/2)*[1-cos(2*pi*t/lg)]
        # Each gust has:
        #   --theta (primary direction 0-2*pi, where zero is left to right)
        #   --wg0 (max gust velocity - calculated from aggressiveness)
        #   --lg (gust wavelength
        #           -- if 1 the gust is exactly 1s long,
        #           -- we want between 0.1 and 2 seconds duration(ish),
        #           -- therefore lg in range = [0.1 2]
        #   --pos (2D location of its centre)
        #   --t0 (time the gust began in s)
        #
        self.gust_params = []
        self.wg0_max = 50  # 50m/s max possible gust speed
        self.gust_rate_max = 10  # max of x10 new gusts per second (ish)
        self.last_gust_t0 = 0

        # init based on the current aggressiveness settings
        self.calc_init_grid()

    def set_aggressiveness(self, steady_state, turbulence, gusts):
        # set gains to control the aggressiveness of the three wind
        # contributions. levels 0 - 10 are used for each, where 0
        # is inactive/zero contribution and 10 is an extremely severe
        # storm/hurricane
        self.K_steady_state = steady_state
        self.K_turbulence = turbulence
        self.K_gusts = gusts

        # reset the wind field using the new values
        self.calc_init_grid()

    def calc_init_grid(self):
        # fill in all values in the wind array
        if self.steady_state_on:
            # set the steady state
            (v, w) = (
                (self.K_steady_state / 10)
                * self.steady_state_max
                * math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            )
            self.wind_field[:, :, 0] = v
            self.wind_field[:, :, 1] = w

        if self.turbulence_on:
            # set the initial turbulence
            pass

        if self.gusts_on:
            # randomly decide if a gust is already happening, and append to the list
            # TODO: needs to allow for a random number of gusts to be happening at the start
            if self.prob_gust():
                self.new_gust()

    def prob_gust(self):
        # has a new gust randomly occured. This is based
        # on the time elapsed since the last gust addition

        # 1/K_gusts+.1 = ~0.1-1s
        if (self.t - self.last_gust_t0) > random.uniform(0, 1 / (self.K_gusts + 0.1)):
            return 1
        else:
            return 0

    def new_gust(self):
        # make a new gust
        x = random.uniform(0, self.width - 1)
        y = random.uniform(0, self.height - 1)
        theta = random.uniform(0, 2 * np.pi)
        wg0 = random.uniform(0, self.wg0_max * (self.K_gusts / 10))
        lg = self.loguniform(0.1, 2)
        if self.t == 0:
            t0 = self.loguniform(-lg, 0)  # offset for how far along in time the gust is
        else:
            t0 = self.t
        self.gust_params.append([math.Vector2(x, y), theta, wg0, lg, t0])
        self.last_gust_t0 = t0

    def step(self, dt):
        # advance the time
        self.t = self.t + dt

        if self.steady_state_on:
            pass

        if self.turbulence_on:
            pass

        if self.gusts_on:
            if self.prob_gust():
                self.new_gust()

            for i in range(len(self.gust_params)):
                # find current value of each discrete gust
                # V = (wg0/2)*[1-cos(2*pi*t/lg)]
                rel_t = self.t - self.gust_params[i][4]  # how far into this gust
                # if within the gust period
                if rel_t < self.gust_params[i][3]:
                    V = (self.gust_params[i][2] / 2.0) * (
                        1 - np.cos((2 * np.pi * self.t) / self.gust_params[i][3])
                    )
                    # TODO: the gust effect is assumed uniform across entire domain
                    self.wind_field[:, :, 0] += np.cos(self.gust_params[i][1]) * V
                    self.wind_field[:, :, 1] += np.sin(self.gust_params[i][1]) * V
                else:
                    # gust is over
                    pass

    def get_wind(self, dt, position_px):
        # perform any time-stepping updates to the wind field
        self.step(dt)

        # check the drone is in the safe area, if not return zero and let the env handle the reset rather than throwing error
        if (
            position_px[0] >= 0
            and position_px[0] <= self.width
            and position_px[1] >= 0
            and position_px[1] <= self.height
        ):
            # use the pixel position to retrieve the current
            # wind at that locaiton
            return math.Vector2(
                self.wind_field[int(position_px[0]), int(position_px[1]), 0],
                self.wind_field[int(position_px[0]), int(position_px[1]), 1],
            )
        else:
            return math.Vector2(0, 0)

    def loguniform(self, low, high):
        # random loguniform number from range expressed in linear scale
        return np.exp(random.uniform(np.log(low), np.log(high)))

    # eof
