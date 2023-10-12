from .turbolander_2d_env import TurboLander2DEnv
from .turbolander_2d_env_v1 import TurboLander2DEnvV1
import gymnasium as gym

gym.envs.register(
    id="turbolander-2d-custom-v0",
    entry_point="turbolander:TurboLander2DEnv",
    kwargs={
        "render_sim": False,
        "render_path": True,
        "n_steps": 500,
    },
)

gym.envs.register(
    id="turbolander-2d-custom-v1",
    entry_point="turbolander:TurboLander2DEnvV1",
    kwargs={
        "render_mode": None,
        "render_path": True,
        "n_steps": 500,
    },
)
