from stable_baselines3 import PPO
import gymnasium as gym
import sys
from turbolander_2d_env_v1 import TurboLander2DEnvV1

gym.envs.register(
    id="turbolander-2d-custom-v1",
    entry_point="turbolander_2d_env_v1:TurboLander2DEnv",
    kwargs={
        "render_sim": False,
        "render_path": True,
        "n_steps": 500,
    },
)
env = gym.make(
    "turbolander-2d-custom-v0",
    render_sim=False,
    render_path=True,
    n_steps=500,
)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=2000000)
model.save("agent_7")
