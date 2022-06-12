import numpy as np
import tensorflow as tf
import time
import gym
from baselines.common.atari_wrappers import wrap_deepmind
from ale_py import ALEInterface
from ale_py.roms import Breakout
from collections import deque


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = tf.keras.layers.Input(shape=(84, 84, 4))

    # Convolutions on the frames on the screen
    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    layer5 = tf.keras.layers.Dense(512, activation="relu")(layer4)
    action = tf.keras.layers.Dense(4, activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)

memory = deque(maxlen = 100000)

ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',        # Use all actions
    render_mode='human'                  # None | human | rgb_array
)
env = wrap_deepmind(env, frame_stack=True, scale=True)

model = create_q_model()

epsilon = 1.0
epsilon_min = 0.1
random_frames = 0
epsilon_random = 0.9 / 1000000

#height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.unwrapped.get_action_meanings()
max_steps = 10000;
episode = 0;
while True:
    state = np.array(env.reset())
    score = 0
    episode += 1
    for step in range(1, max_steps):
        #time.sleep(0.1)
        if(step < random_frames or epsilon > np.random.rand(1)[0]):
            action = np.random.choice(actions)
            
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        new_state, reward, done, info = env.step(action)
        score += reward
        new_state = np.array(new_state)
        memory.append((new_state, action, reward, state, done))
        epsilon -= epsilon_random
        epsilon = max(epsilon, epsilon_min)

        if done:
            break
        
    print('Episode:{} Score:{} Epsilon:{}'.format(episode, score, epsilon))
    if(score > 40):
        break

env.close()