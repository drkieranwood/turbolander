from turbolander_2d_env import TurboLander2DEnv
import gymnasium as gym

gym.envs.register(
    id="turbolander-2d-custom-v0",
    entry_point="TurboLander2DEnv",
    kwargs={
        "render_sim": False,
        "render_path": True,
        "n_steps": 500,
    },
)
