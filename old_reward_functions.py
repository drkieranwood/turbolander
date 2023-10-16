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


# Calulating reward function
# reward += (
#     (1.0 / (np.abs(obs[4]) + 0.01)) + (1.0 / (np.abs(obs[5]) + 0.01))
# ) / 200

# reward += (
#    1 - (np.abs(obs[4]) + np.abs(obs[5])) / 2
# )  # Attraction to landing point

# reward -= np.abs(obs[3])  # Making it not want to tilt too much
