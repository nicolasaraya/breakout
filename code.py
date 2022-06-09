from ale_py import ALEInterface
from ale_py.roms import Breakout
import time
import random
import gym
ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',        # Use all actions
    render_mode='human'                  # None | human | rgb_array
)

height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.unwrapped.get_action_meanings()
episodes = 10
for episode in range(1, episodes+1):
   state = env.reset()
   done = False
   score = 0 
while not done:
   time.sleep(0.1)
   action = random.choice([0,1])
   n_state, reward, done, info = env.step(action)       
   score+=reward
print('Episode:{} Score:{}'.format(episode, score))
env.close()