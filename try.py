from math import gamma
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import gym
from baselines.common.atari_wrappers import wrap_deepmind
from ale_py import ALEInterface
from ale_py.roms import Breakout
from collections import deque


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = tf.keras.layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

memory = deque(maxlen = 100000)

ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',        # Use all actions
    render_mode='human'                  # None | human | rgb_array
)
env = wrap_deepmind(env, frame_stack=True, scale=True)

model = create_q_model()
model.load_weights('weights\Episode_1341_Score_6.0.h5')

#height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.unwrapped.get_action_meanings()
max_steps = 10000
episode = 0
while True:
    state = np.array(env.reset())
    score = 0
    episode += 1
    for step in range(1, max_steps):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        new_state, reward, done, info = env.step(action)
        score += reward
        new_state = np.array(new_state)
        memory.append((state, new_state, action, reward, done))
        if(done):
            if(score > 40):
                break
            else:
                env.step(1)
    print('Episode:{} Score:{}'.format(episode, score))

env.close()