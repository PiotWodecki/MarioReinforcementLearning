from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros


from ResizeObservation import ResizeObservation

# the original environment object
from SkipFrame import SkipFrame

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# TODO wrap the given env with GrayScaleObservation
env = GrayScaleObservation(env, keep_dim=False)
# TODO wrap the given env with ResizeObservation
env = ResizeObservation(env, shape=84)
# TODO wrap the given env with FrameStack
env = FrameStack(env, num_stack=4)

env = SkipFrame(env)