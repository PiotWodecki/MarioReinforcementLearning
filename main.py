import os

import torch

from CustomReward import CustomReward
from Mario import Mario
from MetricLogger import MetricLogger
from ResizeObservation import ResizeObservation
from SkipFrame import SkipFrame

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from datetime import datetime as dt
from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace


env = gym_super_mario_bros.make('SuperMarioBros-v0')

env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

print("Cuda" + str(torch.cuda.is_available()))

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
env = CustomReward(env)

env.reset()

save_dir = Path('checkpoints') / dt.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
# mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint)
model_chkpt = Path('mario_net_218.chkpt')
mario.load(model_chkpt)

logger = MetricLogger(save_dir)

episodes = 40000

for e in range(episodes):

    # env.seed(1)
    state = env.reset()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        env.render()

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )