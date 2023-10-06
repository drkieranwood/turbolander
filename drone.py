# from pygame.math import Vector2
import pygame.math as math
import helpers
import numpy as np


class Drone:
    def __init__(
        self,
        position: math.Vector2,
        velocity: math.Vector2,
        attitude: float,
        angular_velocity: float,
        wind_vector: math.Vector2,
        mass: float,
        rotational_inertia: float,
    ):
        self.position = position
        self.velocity = velocity
        self.attitude = attitude
        self.angular_velocity = angular_velocity
        self.wind_vector = wind_vector
        self.mass = mass
        self.rotational_inertia = rotational_inertia

        self.thrust_multiplier = 20
        self.arm_length = 0.5
        self.box = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]

    def load_sprite(self, sprite):
        self.sprite = sprite
        self.width, self.height = self.sprite.get_size()
        self.update_box()

    def step(self, action, dt):
        u_1 = action[0] / 2 + 0.5
        u_2 = action[1] / 2 + 0.5

        thrust_1 = u_1 * self.thrust_multiplier
        thrust_2 = u_2 * self.thrust_multiplier

        thrust_vector: math.Vector2 = math.Vector2(
            np.sin(self.attitude),
            -np.cos(self.attitude),
        ).elementwise() * (thrust_1 + thrust_2)

        gravitational_acceleration = math.Vector2(0, 9.81)

        acceleration = (
            thrust_vector.elementwise() / self.mass + gravitational_acceleration
        )
        self.velocity = self.velocity + acceleration * dt
        self.position = self.position + self.velocity * dt

        torque = (thrust_1 - thrust_2) * self.arm_length
        angular_acceleration = torque / self.rotational_inertia
        self.angular_velocity = self.angular_velocity + angular_acceleration * dt
        self.attitude = self.attitude + self.angular_velocity * dt
        if self.attitude > 2 * np.pi:
            np.fmod(self.attitude, 2 * np.pi)

        self.update_box()

    def check_collision(self, walls):
        for wall in walls:
            if helpers.box_line_collided(self.box, wall.coordinates):
                return True
        return False

    def update_box(self):
        self.box = [
            helpers.rotate_point(
                self.position,
                self.position + math.Vector2(-self.width / 2, -self.height / 2),
                self.attitude,
            ),
            helpers.rotate_point(
                self.position,
                self.position + math.Vector2(self.width / 2, -self.height / 2),
                self.attitude,
            ),
            helpers.rotate_point(
                self.position,
                self.position + math.Vector2(-self.width / 2, self.height / 2),
                self.attitude,
            ),
            helpers.rotate_point(
                self.position,
                self.position + math.Vector2(self.width / 2, self.height / 2),
                self.attitude,
            ),
        ]
