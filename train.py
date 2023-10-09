from stable_baselines3 import PPO
import gymnasium as gym

from turbolander_2d_env import TurboLander2DEnv

gym.envs.register(
    id="turbolander-2d-custom-v0",
    entry_point="turbolander_2d_env:TurboLander2DEnv",
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

model.learn(total_timesteps=1800000)
model.save("agent_1")
