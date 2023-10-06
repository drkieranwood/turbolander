import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random
import os
import pygame
from drone import Drone
from pygame.math import Vector2
import helpers


class TurboLander2DEnv(gym.Env):
    """
    render_sim: (bool) if true, a graphic is generated
    render_path: (bool) if true, the drone's path is drawn
    render_shade: (bool) if true, the drone's shade is drawn
    shade_distance: (int) distance between consecutive drone's shades
    n_steps: (int) number of time steps
    n_fall_steps: (int) the number of initial steps for which the drone can't do anything
    change_target: (bool) if true, mouse click change target positions
    initial_throw: (bool) if true, the drone is initially thrown with random force
    """

    def __init__(
        self,
        render_sim=False,
        render_path=True,
        n_steps=500,
    ):
        self.render_sim = render_sim
        self.render_path = render_path

        # set up the drone object with default values
        self.drone = Drone(Vector2(400, 400), Vector2(0, 0), 0, 0, Vector2(0, 0), 1, 1)

        if self.render_sim is True:
            self.init_pygame()
            self.flight_path = []

        # Parameters
        self.max_time_steps = n_steps

        # Initial values
        self.first_step = True
        self.done = False
        self.info = {}
        self.current_time_step = 0
        self.left_force = -1
        self.right_force = -1

        # Generating target position
        self.x_target = random.uniform(50, 750)
        self.y_target = random.uniform(50, 750)

        # Defining spaces for action and observation
        min_action = np.array([-1, -1], dtype=np.float32)
        max_action = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(
            low=min_action, high=max_action, dtype=np.float32
        )

        min_observation = np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=min_observation, high=max_observation, dtype=np.float32
        )

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("TurboLander Environment")
        self.clock = pygame.time.Clock()
        self.drone.load_sprite(pygame.image.load("images/drone_2.png").convert_alpha())

        # script_dir = os.path.dirname(__file__)
        # icon_path = os.path.join("img", "icon.png")
        # icon_path = os.path.join(script_dir, icon_path)
        # pygame.display.set_icon(pygame.image.load(icon_path))

        # img_path = os.path.join("img", "shade.png")
        # img_path = os.path.join(script_dir, img_path)
        # self.shade_image = pygame.image.load(img_path)

    def step(self, action):
        # if self.first_step is True:

        self.drone.step(action, 1.0 / 60)
        self.current_time_step += 1

        # Saving drone's position for drawing
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True:
                self.add_postion_to_flight_path(self.drone.position)
            self.first_step = False

        else:
            if self.render_sim is True and self.render_path is True:
                self.add_postion_to_flight_path(self.drone.position)

        # Calulating reward function
        # obs = self.get_observation()
        # reward = (1.0 / (np.abs(obs[4]) + 0.1)) + (1.0 / (np.abs(obs[5]) + 0.1))

        # Stops episode, when drone is out of range or overlaps
        # if np.abs(obs[3]) == 1 or np.abs(obs[6]) == 1 or np.abs(obs[7]) == 1:
        # self.done = True
        # reward = -10

        # Stops episode, when time is up
        if self.current_time_step == self.max_time_steps:
            self.done = True

        # return obs, reward, self.done, self.info

    def get_observation(self):
        # velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point(
        #     (0, 0)
        # )
        # velocity_x = np.clip(velocity_x / 1330, -1, 1)
        # velocity_y = np.clip(velocity_y / 1330, -1, 1)

        # omega = self.drone.frame_shape.body.angular_velocity
        # omega = np.clip(omega / 11.7, -1, 1)

        # alpha = self.drone.frame_shape.body.angle
        # alpha = np.clip(alpha / (np.pi / 2), -1, 1)

        # x, y = self.drone.frame_shape.body.position

        # if x < self.x_target:
        #     distance_x = np.clip((x / self.x_target) - 1, -1, 0)

        # else:
        #     distance_x = np.clip(
        #         (-x / (self.x_target - 800) + self.x_target / (self.x_target - 800)),
        #         0,
        #         1,
        #     )

        # if y < self.y_target:
        #     distance_y = np.clip((y / self.y_target) - 1, -1, 0)

        # else:
        #     distance_y = np.clip(
        #         (-y / (self.y_target - 800) + self.y_target / (self.y_target - 800)),
        #         0,
        #         1,
        #     )

        # pos_x = np.clip(x / 400.0 - 1, -1, 1)
        # pos_y = np.clip(y / 400.0 - 1, -1, 1)

        # return np.array(
        #     [velocity_x, velocity_y, omega, alpha, distance_x, distance_y, pos_x, pos_y]
        # )
        pass

    def render(self, mode="human", close=False):
        if self.render_sim is False:
            return
        self.screen.fill((243, 243, 243))
        # pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(50, 50, 70, 70), 8)

        # self.screen.blit(
        #     self.drone.sprite, (self.drone.position[0], self.drone.position[1])
        # )

        helpers.blit_rotate(
            self.screen,
            self.drone.sprite,
            self.drone.position,
            (self.drone.width / 2, self.drone.height / 2),
            helpers.radians_to_degrees(-self.drone.attitude),
        )

        # # Drawing vectors of motor forces
        # vector_scale = 0.05
        # l_x_1, l_y_1 = self.drone.frame_shape.body.local_to_world(
        #     (-self.drone_radius, 0)
        # )
        # l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world(
        #     (-self.drone_radius, self.froce_scale * vector_scale)
        # )
        # pygame.draw.line(
        #     self.screen, (179, 179, 179), (l_x_1, 800 - l_y_1), (l_x_2, 800 - l_y_2), 4
        # )

        # l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world(
        #     (-self.drone_radius, self.left_force * vector_scale)
        # )
        # pygame.draw.line(
        #     self.screen, (255, 0, 0), (l_x_1, 800 - l_y_1), (l_x_2, 800 - l_y_2), 4
        # )

        # r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world(
        #     (self.drone_radius, 0)
        # )
        # r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world(
        #     (self.drone_radius, self.froce_scale * vector_scale)
        # )
        # pygame.draw.line(
        #     self.screen, (179, 179, 179), (r_x_1, 800 - r_y_1), (r_x_2, 800 - r_y_2), 4
        # )

        # r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world(
        #     (self.drone_radius, self.right_force * vector_scale)
        # )
        # pygame.draw.line(
        #     self.screen, (255, 0, 0), (r_x_1, 800 - r_y_1), (r_x_2, 800 - r_y_2), 4
        # )

        # pygame.draw.circle(
        #     self.screen, (255, 0, 0), (self.x_target, 800 - self.y_target), 5
        # )

        # Drawing drone's path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        self.__init__(
            self.render_sim,
            self.render_path,
            self.max_time_steps,
        )
        return self.get_observation()

    def close(self):
        pygame.quit()

    def add_postion_to_flight_path(self, position):
        self.flight_path.append((position[0], position[1]))
