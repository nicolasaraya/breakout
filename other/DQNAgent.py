import tensorflow as tf
import tensorflow.compat.v1 as tfc
import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential
class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0, reuse=False):
        """A simple DQN agent"""
        tf.compat.v1.disable_eager_execution()
        with tfc.variable_scope(name, reuse=reuse):
            
            #< Define your network body here. Please make sure you don't use any layers created elsewhere >
            self.network = Sequential()
            self.network.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
            self.network.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
            self.network.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
            self.network.add(Flatten())
            self.network.add(Dense(256, activation='relu'))
            self.network.add(Dense(n_actions, activation='linear'))
            
            # prepare a graph for agent step
            self.state_t = tfc.placeholder('float32', [None,] + list(state_shape))
            self.qvalues_t = self.get_symbolic_qvalues(self.state_t, n_actions)
            
        self.weights = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t, n_actions):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        #< apply your network layers here >
        qvalues = self.network(state_t) #< symbolic tensor for q-values >
        
        assert tfc.is_numeric_tensor(qvalues) and qvalues.shape.ndims == 2, \
            "please return 2d tf tensor of qvalues [you got %s]" % repr(qvalues)
        assert int(qvalues.shape[1]) == n_actions
        
        return qvalues
    
    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tfc.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})
    
    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)