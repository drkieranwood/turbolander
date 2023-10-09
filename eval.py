from stable_baselines3 import PPO
import gymnasium as gym
import time
import sys

from turbolander_2d_env import TurboLander2DEnv

gym.envs.register(
    id="turbolander-2d-custom-v0",
    entry_point="turbolander_2d_env:TurboLander2DEnv",
    kwargs={
        "render_sim": True,
        "render_path": True,
        "n_steps": 500,
    },
)

continuous_mode = (
    True  # if True, after completing one episode the next one will start automatically
)

render_sim = True  # if True, a graphic is generated

env = gym.make(
    "turbolander-2d-custom-v0",
    render_sim,
    render_path=True,
    n_steps=500,
)

"""
The example agent used here was originally trained with Python 3.7
For this reason, it is not compatible with Python version >= 3.8
Agent has been adapted to run in the newer version of Python,
but because of this, you cannot easily resume their training.
If you are interested in resuming learning, please use Python 3.7.
"""

model = PPO.load("agent_1.zip")

model.set_env(env)

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs, info = env.reset()

try:
    while True:
        if render_sim:
            env.render()

        action, _states = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        if done is True:
            if continuous_mode is True:
                state = env.reset()
            else:
                break

finally:
    env.close()
