from baselines.common.atari_wrappers import wrap_deepmind
from ale_py import ALEInterface
from ale_py.roms import Breakout
import time
import gym
import numpy as np
from collections import deque
'''''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(210, 160, 4))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)
'''
memory = deque(maxlen = 100000)

ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',        # Use all actions
    render_mode='human'                  # None | human | rgb_array
)

env = wrap_deepmind(env, frame_stack=True, scale=True)

height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.unwrapped.get_action_meanings()
max_steps = 1000;
episode = 0;
while True:
    state = np.array(env.reset())
    score = 0
    episode += 1
    for step in range(1, max_steps):
        #time.sleep(0.1)
        action = np.random.choice(actions)
        new_state, reward, done, info = env.step(action)
        score += reward
        new_state = np.array(new_state)
        memory.append((new_state, action, reward, state, done))
        print(new_state.shape)
        if done:
            break
        
    print('Episode:{} Score:{}'.format(episode, score))
    if(score > 40):
        break
env.close()