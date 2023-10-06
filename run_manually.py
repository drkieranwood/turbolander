import pygame
from turbolander_2d_env import TurboLander2DEnv

environment = TurboLander2DEnv(
    render_sim=True,
    render_path=True,
    n_steps=500,
)

# creating a bool value which checks
# if game is running
running = True


# Game loop
# keep game running till running is true
while running:
    # Check for event if user has pushed
    # any event in queue
    action = [-1.0, -1.0]
    for event in pygame.event.get():
        # if event is of type quit then set
        # running bool to false
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                environment.reset()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action[0] = 1.0
    if keys[pygame.K_RIGHT]:
        action[1] = 1.0

    environment.step(action)
    environment.render()
