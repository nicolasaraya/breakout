from joblib import PrintTime
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
import glob

###### Cambiar solo estos dartos ######
MODELO = 'mejor_modelo.py'
PESOS_MODELO = 'Episode_10400_Score_6.7.h5'
#######################################

codigo_modelo = glob.glob('./modelos/{}'.format(MODELO))
for linea_de_codigo in codigo_modelo: 
     o = open(linea_de_codigo)  
     r = o.read()       
     exec(r)

ale = ALEInterface()
ale.loadROM(Breakout)
env = gym.make('ALE/Breakout-v5',        # Use all actions
    render_mode='human'                  # None | human | rgb_array
)
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

model = create_q_model()
model.load_weights('modelos/weights/{}'.format(PESOS_MODELO))

episode = 0
while True:
    state = np.array(env.reset())
    score = 0
    episode += 1
    for step in range(1, max_steps):
        # From environment state
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        new_state, reward, done, info = env.step(action)
        score += reward
        new_state = np.array(new_state)
        if(done):
            break
    print('Episode:{} Score:{}'.format(episode, score))
    if(score > 40):
        print('__________WIN__________(❁´◡`❁)')
        break

env.close()