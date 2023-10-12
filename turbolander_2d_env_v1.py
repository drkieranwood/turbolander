import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random
import os
import pygame
import pygame.freetype
from pygame.math import Vector2
from . import helpers
from .drone import Drone
from .wall import Wall
from typing import Optional
import pathlib


class TurboLander2DEnvV1(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    """
    render_sim: (bool) if true, a graphic is generated
    render_path: (bool) if true, the drone's path is drawn
    n_steps: (int) number of time steps
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        render_path=True,
        n_steps=500,
    ):
        self.render_mode = render_mode
        self.render_path = render_path
        self.last_observation = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.last_action = np.array([0, 0], dtype=np.float32)

        # set up the drone object with default values
        self.drone = Drone(Vector2(4, 4), Vector2(0, 0), 0, 0, Vector2(0, 0), 1, 0.5)

        self.max_speed = np.sqrt(
            2 * 9.81 * (self.drone.thrust_multiplier * 2 / self.drone.mass) * 8
        )

        self.walls = [Wall([0, 750, 800, 750], 0.6)]

        # if self.render_sim is True:
        #     self.init_pygame()
        #     self.flight_path = []
        if self.render_mode == "human":
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
        self.y_target_px = random.uniform(50, 750)
        self.y_target_m = self.y_target_px / 100
        self.z_target_px = 750
        self.z_target_m = self.z_target_px / 100
        self.target_radius_m = 1

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

        self.drone.load_sprite(
            pygame.image.load(
                str(pathlib.Path(__file__).parent.resolve()) + "/images/drone_2.png"
            ).convert_alpha()
        )

    def step(self, action):
        # if self.first_step is True:
        reward = 0
        self.drone.step(action, 1.0 / 60)
        collided, landed = self.drone.check_collision(self.walls)
        if collided:
            self.done = True
            if (
                landed
                and self.drone.velocity.magnitude() < 0.2
                and abs(self.drone.attitude) < 15 * np.pi / 180
                # and abs(self.drone.angular_velocity) < 0.05
                and abs(self.drone.position_m[0] - self.y_target_m)
                < self.target_radius_m
            ):
                reward += (self.max_time_steps * 2) - (self.current_time_step * 2)
            else:
                reward += -100

        # Saving drone's position for drawing
        if self.first_step is True:
            if self.render_mode == "human" and self.render_path is True:
                self.add_postion_to_flight_path(self.drone.position_px)
            self.first_step = False

        else:
            if self.render_mode == "human" and self.render_path is True:
                self.add_postion_to_flight_path(self.drone.position_px)

        # Calulating reward function
        obs = self.get_observation()
        # provides a max reward of 1 if on top of point
        reward += (
            (1.0 / (np.abs(obs[4]) + 0.01)) + (1.0 / (np.abs(obs[5]) + 0.01))
        ) / 200

        # Stops episode, when drone is out of range or overlaps
        if np.abs(obs[3]) == 1 or np.abs(obs[6]) == 1 or np.abs(obs[7]) == 1:
            self.done = True
            reward += -100
            # reward = -10

        # Stops episode, when time is up
        self.current_time_step += 1
        if self.current_time_step == self.max_time_steps:
            self.done = True

        self.last_observation = obs
        self.last_action = action

        if self.render_mode == "human":
            self.render()
        return obs, reward, self.done, False, self.info

    def get_observation(self):
        velocity_y = np.clip(self.drone.velocity[0] / self.max_speed, -1, 1)
        velocity_z = np.clip(self.drone.velocity[1] / self.max_speed, -1, 1)

        angular_velocity = np.clip(self.drone.angular_velocity / 20, -1, 1)

        attitude = np.clip(self.drone.attitude / np.pi, -1, 1)

        position_y = np.clip(
            (self.drone.position_m[0] / 4) - 1, -1, 1
        )  # 4 is half the width and height of env in meters

        position_z = np.clip((self.drone.position_m[1] / 4) - 1, -1, 1)

        target_y_norm = (self.y_target_px / 400) - 1
        target_z_norm = (self.z_target_px / 400) - 1

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

    def render(self):
        if self.render_mode != "human":
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

        self.draw_ui()

        # Draw drone
        helpers.blit_rotate(
            self.screen,
            self.drone.sprite,
            self.drone.position_px,
            (self.drone.width_px / 2, self.drone.height_px / 2),
            helpers.radians_to_degrees(-self.drone.attitude),
        )

        # Draw target
        # pygame.draw.circle(
        #     self.screen, (255, 0, 0), (self.y_target_px, self.z_target_px), 5
        # )

        pygame.draw.line(
            self.screen,
            (255, 0, 0),
            (
                (self.y_target_m - self.target_radius_m) * 100,
                self.z_target_px,
            ),
            ((self.y_target_m + self.target_radius_m) * 100, self.z_target_px),
            8,
        )

        # Draw drone's path
        if len(self.flight_path) > 2:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

        pygame.display.flip()
        self.clock.tick(60)

    def draw_ui(self):
        # Draw left throttle command
        pygame.draw.line(  # Background
            self.screen,
            (211, 211, 211),
            (20, 120),
            (20, 20),
            8,
        )
        pygame.draw.line(  # Throttle 1 bar
            self.screen,
            (255, 105, 97),
            (20, 120),
            (20, 120 - np.rint(((self.last_action[0] + 1) / 2 * 100))),
            8,
        )

        # Draw right throttle command
        pygame.draw.line(  # Background
            self.screen,
            (211, 211, 211),
            (40, 120),
            (40, 20),
            8,
        )

        pygame.draw.line(  # Throttle 2 bar
            self.screen,
            (255, 105, 97),
            (40, 120),
            (40, 120 - np.rint(((self.last_action[1] + 1) / 2 * 100))),
            8,
        )

        # Draw observations
        offset = 13
        ft_font = pygame.freetype.Font(None, 11)
        ft_font.render_to(self.screen, (50, 21), "v", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + offset), "w", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + 2 * offset), "p", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + 3 * offset), "phi", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + 4 * offset), "d_y", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + 5 * offset), "d_z", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + 6 * offset), "y", (0, 0, 0))
        ft_font.render_to(self.screen, (50, 21 + 7 * offset), "z", (0, 0, 0))

        current_offset = 0
        for observation in self.last_observation:
            pygame.draw.line(
                self.screen,
                (211, 211, 211),
                (75, 23 + current_offset),
                (115, 23 + current_offset),
                8,
            )
            rounded_observation = np.rint(observation * 20)
            if rounded_observation != 0:
                pygame.draw.line(
                    self.screen,
                    (255, 184, 97),
                    (95, 23 + current_offset),
                    (95 + rounded_observation, 23 + current_offset),
                    8,
                )
            pygame.draw.line(
                self.screen,
                (255, 32, 21),
                (95, 20 + current_offset),
                (95, 27 + current_offset),
                1,
            )
            current_offset = current_offset + offset

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.__init__(
            self.render_mode,
            self.render_path,
            self.max_time_steps,
        )
        return self.get_observation(), self.info

    def close(self):
        pygame.quit()

    def add_postion_to_flight_path(self, position):
        self.flight_path.append((position[0], position[1]))
