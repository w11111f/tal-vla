import gym
from src.baselines.metadt.envs.tp_envs.environments.wrappers import VariBadWrapper


def make_env(env_id, episodes_per_task, seed=None, **kwargs):
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.seed(seed)
    env = VariBadWrapper(env=env,
                         episodes_per_task=episodes_per_task,
                         )
    return env


