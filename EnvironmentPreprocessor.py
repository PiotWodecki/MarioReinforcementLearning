from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def remove_colors_from_environment(env):
    env = GrayScaleObservation(env, keep_dim=True)
    return env


def vectorize_environment(env):
    env = env = DummyVecEnv([lambda: env])
    return env


def create_environment_stack(env, frames=4, order='last'):
    env = VecFrameStack(env, frames, channels_order=order)
    return env


def preprocess_environment(env):
    env = remove_colors_from_environment(env)
    env = vectorize_environment(env)
    env = create_environment_stack(env)
    return env
