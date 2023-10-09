import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from turbolander_2d_env import TurboLander2DEnv


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
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
            env_id,
            render_sim=False,
            render_path=True,
            n_steps=500,
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    num_cpu = 16  # Number of processes to use

    vec_env = SubprocVecEnv(
        [make_env("turbolander-2d-custom-v0", i) for i in range(num_cpu)]
    )

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=20000000)
    model.save("agent_5")
