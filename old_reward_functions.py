# if collided: # Used for models 34, 35 + 36
#     self.done = True
#     if (
#         landed
#         and self.drone.velocity.magnitude() < 0.5
#         and abs(self.drone.attitude) < 15 * (np.pi / 180)
#     ):
#         if (
#             abs(self.drone.position_m[0] - self.y_target_m)
#             < self.target_radius_m
#         ):
#             reward += (self.max_time_steps * 2) - (self.current_time_step * 2)
#         else:
#             reward += 10
#     else:
#         reward += -50

# if collided:  # Used for models 33 + 37
#     self.done = True
#     if landed:
#         reward += 1000 * np.exp(
#             -5
#             * (np.abs(obs[4]) + (1 - np.abs(obs[1])) + (1 - np.abs(obs[3])))
#             / 3
#         )
#     else:
#         reward += -100

# if collided:  # Used for model 38
#     self.done = True
#     if landed:
#         reward += (
#             10000
#             * np.exp(
#                 -5
#                 * (np.abs(obs[4]) + (1 - np.abs(obs[1])) + (1 - np.abs(obs[3])))
#                 / 3
#             )
#         ) * (1 / (self.current_time_step + 0.001))
#     else:
#         reward += -100

# if collided:  # Used for model 39 + 40 (is wrong)
#     self.done = True
#     if landed:
#         reward += 10000 * np.exp(
#             -5
#             * (np.abs(obs[4]) + (1 - np.abs(obs[1])) + (1 - np.abs(obs[3])))
#             / 3
#         )
#     else:
#         reward += -100

# if collided:
#     self.done = True
#     if landed:
#         if (
#             abs(self.drone.position_m[0] - self.y_target_m)
#             < self.target_radius_m
#         ):
#             reward += 1000 * np.exp(
#                 -2 * ((1 - np.abs(obs[1])) + (1 - np.abs(obs[3])))
#             )
#         else:
#             reward += -50
#     else:
#         reward += -100

# if collided:  # Used for model 41 + 42 + 44
#     self.done = True
#     if landed:
#         reward += 1000 * np.exp(
#             -5
#             * (
#                 np.abs(obs[4])
#                 + np.abs(obs[0])
#                 + np.abs(obs[1])
#                 + np.abs(obs[3])
#             )
#             / 4
#         )
#     else:
#         reward += -100

# if collided:  # Used for model 45
#     self.done = True
#     if landed:
#         reward += 1000 * np.exp(
#             -10
#             * (
#                 np.abs(obs[4])
#                 + np.abs(obs[0])
#                 + np.abs(obs[1])
#                 + np.abs(obs[3])
#             )
#             / 4
#         )
#     else:
#         reward += -100

# if collided:  # Used for model 46, 47, 48 and 49 (only 49 has 10x for horizontal speed, others has 2x)
#     self.done = True
#     if landed:
#         reward += 1000 * np.exp(
#             -5
#             * (
#                 np.abs(obs[4])
#                 + np.abs(obs[0]) * 2
#                 + np.abs(obs[1]) * 2
#                 + np.abs(obs[3])
#             )
#             / 6
#         )
#     else:
#         reward += -100

# if collided:  # Used for model 50
#     self.done = True
#     if landed:
#         reward += 1000 * np.exp(
#             -5
#             * (
#                 np.abs(obs[4])
#                 + np.abs(obs[0]) * 10
#                 + np.abs(obs[1]) * 2
#                 + np.abs(obs[3])
#             )
#             / 14
#         ) - (self.current_time_step * 2)
#     else:
#         reward += -100


# Calulating reward function
# reward += (
#     (1.0 / (np.abs(obs[4]) + 0.01)) + (1.0 / (np.abs(obs[5]) + 0.01))
# ) / 200

# reward += (
#    1 - (np.abs(obs[4]) + np.abs(obs[5])) / 2
# )  # Attraction to landing point

# reward -= np.abs(obs[3])  # Making it not want to tilt too much


# if collided:  # new test reward
#     self.done = True
#     if (
#         landed
#         and abs(self.drone.velocity[0]) < 1
#         and abs(self.drone.velocity[1]) < 0.5
#         and abs(self.drone.attitude) < 15 * (np.pi / 180)
#     ):
#         if (
#             abs(self.drone.position_m[0] - self.y_target_m)
#             < self.target_radius_m
#         ):
#             reward += 2000
#         else:
#             reward += 10

#     elif landed:
#         reward += 1000 * np.exp(
#             -5
#             * (
#                 np.abs(obs[4])
#                 + np.abs(obs[0]) * 2
#                 + np.abs(obs[1])
#                 + np.abs(obs[3])
#             )
#             / 5
#         )
#     else:
#         reward += -50

# if collided:  # new test reward for sac model 10, 11 and 12
#     self.done = True
#     if (  # Criteria for safe landing
#         landed
#         and abs(self.drone.velocity[0]) < 0.2
#         and abs(self.drone.velocity[1]) < 0.5
#         and abs(self.drone.attitude) < 15 * (np.pi / 180)
#     ):
#         if (  # Criteria for landing on target
#             abs(self.drone.position_m[0] - self.y_target_m)
#             <= self.target_radius_m
#         ):
#             reward += 50 - (
#                 self.current_time_step * 0.1
#             )  # Safe landing on target
#         else:
#             reward += 0  # Safe landing off target
#     else:
#         reward += -50  # Crash

# Landing reward
# if collided:  # Used for SAC model 9
#     self.done = True
#     # reward += -100
#     if landed:
#         reward += 50 * np.exp(
#             -10
#             * (
#                 np.abs(obs[4])
#                 + np.abs(obs[0]) * 2
#                 + np.abs(obs[1])
#                 + np.abs(obs[3])
#             )
#             / 5
#         )
#     else:
#         reward += -50

# reward += 0.1 * np.exp(
#     -5 * (np.abs(obs[4]) + np.abs(obs[5]))
# )  # Attraction to landing point (used for model 22 + 53), used as first step for training without penalty. Used for sac model 11
