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

memory = deque(maxlen = 400000)
score_history = []

ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',        # Use all actions
    render_mode='rgb_array'                  # None | human | rgb_array
)
env = wrap_deepmind(env, frame_stack=True, scale=True)

model = create_q_model()
model_target = create_q_model()
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=0.95)
loss_function = keras.losses.Huber()

epsilon = 1.0
epsilon_min = 0.01
random_frames = 100000
delta_epsilon = 0.99 / 1000000

batch_size = 32
gamma = 0.99

update_target_network = 10000

#height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.unwrapped.get_action_meanings()
max_steps = 10000
running_reward = 0
episode = 0
max_score = 0
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
        memory.append((state, new_state, action, reward, done))
        epsilon -= delta_epsilon
        epsilon = max(epsilon, epsilon_min)


        if(step%4 == 0 and len(memory) >= batch_size):
            indices = np.random.choice(range(len(memory)), size = batch_size)
            
            # Using list comprehension to sample from replay buffer
            state_sample = np.array([memory[i][0] for i in indices])
            state_next_sample = np.array([memory[i][1] for i in indices])
            action_sample = [memory[i][2] for i in indices]
            rewards_sample = [memory[i][3] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(memory[i][4]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode, step))

        if done:
            break
        
    score_history.append(score)
    if len(score_history) > 100:
        del score_history[:1]
    running_reward = np.mean(score_history)


    print('Episode:{} Score:{} Epsilon:{}'.format(episode, score, epsilon))
    if(score > max_score):
        model.save_weights('weights/Episode_{}_Score_{}.h5'.format(episode,score))
        max_score = score
    if(score > 40):
        break

env.close()