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
from wall import Wall


class TurboLander2DEnv(gym.Env):
    """
    render_sim: (bool) if true, a graphic is generated
    render_path: (bool) if true, the drone's path is drawn
    n_steps: (int) number of time steps
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
        self.drone = Drone(Vector2(4, 4), Vector2(0, 0), 0, 0, Vector2(0, 0), 1, 0.5)

        self.max_speed = np.sqrt(
            2 * 9.81 * (self.drone.thrust_multiplier * 2 / self.drone.mass) * 8
        )

        self.walls = [Wall([0, 750, 800, 750], 0.6)]

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
        self.y_target = random.uniform(50, 750)
        self.z_target = 730
        # self.z_target = random.uniform(50, 750)

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
        if self.drone.check_collision(self.walls):
            self.done = True
            # reward = -0.5
            reward = -10
        self.current_time_step += 1

        # Saving drone's position for drawing
        if self.first_step is True:
            if self.render_sim is True and self.render_path is True:
                self.add_postion_to_flight_path(self.drone.position_px)
            self.first_step = False

        else:
            if self.render_sim is True and self.render_path is True:
                self.add_postion_to_flight_path(self.drone.position_px)

        # Calulating reward function
        obs = self.get_observation()
        # reward = (1.0 / (np.abs(obs[4]) + 2)) + (1.0 / (np.abs(obs[5]) + 2))
        reward = (1.0 / (np.abs(obs[4]) + 0.1)) + (1.0 / (np.abs(obs[5]) + 0.1))

        # Stops episode, when drone is out of range or overlaps
        if np.abs(obs[3]) == 1 or np.abs(obs[6]) == 1 or np.abs(obs[7]) == 1:
            self.done = True
            # reward = -1
            reward = -10

        # Stops episode, when time is up
        if self.current_time_step == self.max_time_steps:
            self.done = True

        return obs, reward, self.done, False, self.info

    def get_observation(self):
        velocity_y = np.clip(self.drone.velocity[0] / self.max_speed, -1, 1)
        velocity_z = np.clip(self.drone.velocity[1] / self.max_speed, -1, 1)

        angular_velocity = np.clip(self.drone.angular_velocity / 20, -1, 1)

        attitude = np.clip((self.drone.attitude / np.pi) - 1, -1, 1)

        position_y = np.clip(
            (self.drone.position_m[0] / 4) - 1, -1, 1
        )  # 4 is half the width and height of env in meters

        position_z = np.clip((self.drone.position_m[1] / 4) - 1, -1, 1)

        target_y_norm = (self.y_target / 400) - 1
        target_z_norm = (self.z_target / 400) - 1

        target_dist_y = np.clip(position_y - target_y_norm, -1, 1)
        target_dist_z = np.clip(position_z - target_z_norm, -1, 1)

        return np.array(
            [
                velocity_y,
                velocity_z,
                angular_velocity,
                attitude,
                target_dist_y,
                target_dist_z,
                position_y,
                position_z,
            ],
            dtype=np.float32,
        )

    def render(self, mode="human", close=False):
        if self.render_sim is False:
            return
        self.screen.fill((243, 243, 243))

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(
                self.screen,
                (34, 139, 34),
                (wall.coordinates[0], wall.coordinates[1]),
                (wall.coordinates[2], wall.coordinates[3]),
                8,
            )

        # Draw drone
        helpers.blit_rotate(
            self.screen,
            self.drone.sprite,
            self.drone.position_px,
            (self.drone.width_px / 2, self.drone.height_px / 2),
            helpers.radians_to_degrees(-self.drone.attitude),
        )

        # Draw target
        pygame.draw.circle(self.screen, (255, 0, 0), (self.y_target, self.z_target), 5)

        # Draw drone's path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.__init__(
            self.render_sim,
            self.render_path,
            self.max_time_steps,
        )
        return self.get_observation(), self.info

    def close(self):
        pygame.quit()

    def add_postion_to_flight_path(self, position):
        self.flight_path.append((position[0], position[1]))
