# from pygame.math import Vector2
import pygame.math as math
from . import helpers
import numpy as np


class Drone:
    def __init__(
        self,
        position_m: math.Vector2,
        velocity: math.Vector2,
        attitude: float,
        angular_velocity: float,
        mass: float,
        rotational_inertia: float,
    ):
        self.position_m = position_m
        self.position_px = position_m * 100
        self.velocity = velocity
        self.attitude = attitude
        self.angular_velocity = angular_velocity
        self.mass = mass
        self.rotational_inertia = rotational_inertia
        self.drag_coefficient = 0.5
        self.reference_area = 0.1
        self.air_density = 1.225
        self.width_px = 50
        self.height_px = 10

        self.thrust_multiplier = 20
        self.arm_length = 0.25
        self.box = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]

    def load_sprite(self, sprite):
        self.sprite = sprite
        self.width_px, self.height_px = self.sprite.get_size()
        self.update_box()

    def step(self, action, dt, wind_vector):
        u_1 = action[0] / 2 + 0.5
        u_2 = action[1] / 2 + 0.5

        thrust_1 = u_1 * self.thrust_multiplier
        thrust_2 = u_2 * self.thrust_multiplier

        # thrust_vector: math.Vector2 = math.Vector2(
        #     np.sin(self.attitude),
        #     -np.cos(self.attitude),
        # ).elementwise() * (thrust_1 + thrust_2)]

        thrust_vector: math.Vector2 = math.Vector2(
            np.sin(self.attitude),
            -np.cos(self.attitude),
        ).elementwise() * (thrust_1 + thrust_2)

        # Changed the upright flight attitude to be pi radians instead of 0 so that the observation is 0 when the drone is upright
        # means that there isnt a discontinous change in observation when the drone rolls left causing failure as it went to -1 and caused end of episode

        gravitational_acceleration = math.Vector2(0, 9.81)

        air_relative_velocity = self.velocity - wind_vector

        drag = (
            0.5
            * self.drag_coefficient
            * self.reference_area
            * self.air_density
            * (air_relative_velocity.magnitude() ** 2)
        ) * -air_relative_velocity.normalize()

        # acceleration = (
        #     thrust_vector.elementwise() / self.mass + gravitational_acceleration
        # )

        acceleration = (
            thrust_vector.elementwise() / self.mass
            + gravitational_acceleration
            + drag.elementwise() / self.mass
        )

        self.velocity = self.velocity + acceleration * dt
        self.position_m = self.position_m + self.velocity * dt
        self.position_px = self.position_m * 100

        torque = (thrust_1 - thrust_2) * self.arm_length
        angular_acceleration = torque / self.rotational_inertia
        self.angular_velocity = self.angular_velocity + angular_acceleration * dt
        self.attitude = self.attitude + self.angular_velocity * dt
        if self.attitude > np.pi:
            self.attitude = -np.pi + np.fmod(self.attitude, np.pi)
        elif self.attitude < -np.pi:
            self.attitude = np.pi + np.fmod(self.attitude, np.pi)

        self.update_box()
        # if self.check_collision(walls):
        #     v_normal = self.velocity.dot(walls[0].normal) * walls[0].normal

        #     self.velocity = self.velocity - v_normal * 2

    def check_collision(self, walls):
        for wall in walls:
            if helpers.box_line_collided(self.box, wall.coordinates):
                if wall.is_ground:
                    return True, True
                return True, False
        return False, False

    def update_box(self):
        self.box = [
            helpers.rotate_point(
                self.position_px,
                self.position_px
                + math.Vector2(-self.width_px / 2, -self.height_px / 2),
                self.attitude,
            ),
            helpers.rotate_point(
                self.position_px,
                self.position_px + math.Vector2(self.width_px / 2, -self.height_px / 2),
                self.attitude,
            ),
            helpers.rotate_point(
                self.position_px,
                self.position_px + math.Vector2(-self.width_px / 2, self.height_px / 2),
                self.attitude,
            ),
            helpers.rotate_point(
                self.position_px,
                self.position_px + math.Vector2(self.width_px / 2, self.height_px / 2),
                self.attitude,
            ),
        ]
