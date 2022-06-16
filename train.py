import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import gym
from baselines.common.atari_wrappers import wrap_deepmind
from ale_py import ALEInterface
from ale_py.roms import Breakout
from collections import deque
import glob

###### Cambiar solo estos datos ######
MODELO = 'modelo_j_3'
#######################################

codigo_modelo = glob.glob('./modelos/{}.py'.format(MODELO))
for linea_de_codigo in codigo_modelo: 
     o = open(linea_de_codigo)  
     r = o.read()       
     exec(r)

memory = deque(maxlen = memory_size)

ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',           # Use all actions
    render_mode='rgb_array'                 # None | human | rgb_array
)
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

model = create_q_model()
model_target = create_q_model()

actions = env.action_space.n

episode = 0
score_prom = 0
score_prom_history = []
episode_history = []
max_score_prom = 0

while True:
    state = np.array(env.reset())
    score = 0
    episode += 1
    if(episode > random_episodes):
        epsilon -= delta_epsilon
        epsilon = max(epsilon, epsilon_min)

    for step in range(1, max_steps):
        #time.sleep(0.1)
        if(episode < random_episodes or epsilon > np.random.rand(1)[0]):
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
        state = new_state

        if(step % update_after_actions == 0 and len(memory) >= batch_size):
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
            future_rewards = model_target.predict(state_next_sample, verbose = 0)
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
            
        if done:
            break
    
    # print()
    score_prom += score/10.0
    if(episode > 9 and episode%10 == 0):
        print('Episode:{} Score:{} Epsilon:{}'.format(episode, round(score_prom, 2), round(epsilon,3)))
        score_prom_history.append( score_prom )
        episode_history.append(episode)
        if(score_prom > max_score_prom):
            max_score_prom = score_prom
            model.save_weights('modelos/weights/{}.h5'.format(MODELO,episode,round(score_prom, 2)))
            print('____________SAVE____________')
        score_prom = 0

    #save graph
    if(episode >= 100 and episode%100 == 0):
        plt.clf()
        plt.plot(episode_history, score_prom_history)
        plt.savefig('modelos/graphs/{}.png'.format(MODELO,episode,round(score_prom, 2)))

    # update the the target network with new weights
    if(episode>= update_q_after_episodes):
        model_target.set_weights(model.get_weights())

    if(score > 40):
        break

env.close()