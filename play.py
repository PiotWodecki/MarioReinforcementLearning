import os

import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, TransformObservation, FrameStack
from nes_py.wrappers import JoypadSpace

from CustomReward import CustomReward
from Mario import Mario
from pathlib import Path

from ResizeObservation import ResizeObservation
from SkipFrame import SkipFrame

os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
env = CustomReward(env)

checkpoint = Path('checkpoints/2022-05-18T00-14-04/mario_net_218.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint)
mario.load(checkpoint)
# Start the game
state = env.reset()
# Loop through the game
while True:
    action = mario.act(state)

    # 5. Agent performs action
    next_state, reward, done, info = env.step(action)
    env.render()