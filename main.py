import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import keras
import datetime
import gym.wrappers
import argparse
from other.PreprocessAtari import PreprocessAtari
from other.FrameBuffer import FrameBuffer
from other.DQNAgent import DQNAgent
from other.ReplayBuffer import ReplayBuffer
from ale_py import ALEInterface
from ale_py.roms import Breakout
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, Flatten, InputLayer
from tqdm import trange
from IPython.display import clear_output
from pandas import DataFrame

def parse():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--train_dqn', type=int, nargs = '?',const=10000, default = None,  help = 'Train model with DQN')
    parser.add_argument('--test_dqn', action = 'store_true',  help = 'Test model')
    parser.add_argument('--render', action = 'store_true', default = None, help = 'Render model')
    args = parser.parse_args()
    return args

def make_env(render = False):
    ale = ALEInterface()
    ale.loadROM(Breakout)
    if(render):
         env = gym.make('ALE/Breakout-v5',
                obs_type='rgb',                   # ram | rgb | grayscale
                frameskip=4,                      # frame skip
                mode=None,                        # game mode, see Machado et al. 2018
                difficulty=None,                  # game difficulty, see Machado et al. 2018
                repeat_action_probability=0.25,   # Sticky action probability
                full_action_space=False,          # Use all actions
                render_mode='human'                  # None | human | rgb_array
            )
    else: 
        env = gym.make('ALE/Breakout-v5',
                obs_type='rgb',                   # ram | rgb | grayscale
                frameskip=4,                      # frame skip
                mode=None,                        # game mode, see Machado et al. 2018
                difficulty=None,                  # game difficulty, see Machado et al. 2018
                repeat_action_probability=0.25,   # Sticky action probability
                full_action_space=False,          # Use all actions
                render_mode=None                  # None | human | rgb_array
            )
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')
    return env

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000, render = None, test=False):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        if test: env.step(1)
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            if render: 
                env.render(mode= 'rgb_array')
            s, r, done, _ = env.step(action)
            reward += r
            if done: 
                break
            
        rewards.append(reward)
    return np.mean(rewards)

def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    :returns: return sum of rewards over time
    
    Note: please do not env.reset() unless env is done.
    It is guaranteed that env has done=False when passed to this function.
    """
    # State at the beginning of rollout
    s = env.framebuffer
    
    # Play the game for n_steps as per instructions above
    reward = 0.0
    for t in range(n_steps):
        # get agent to pick action given state s
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(action)
        
        # add to replay buffer
        exp_replay.add(s, action, r, next_s, done)
        reward += r
        if done:
            s = env.reset()
        else:
            s = next_s
    return reward
        
def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tfc.assign(w_target, w_agent, validate_shape=True))
    tfc.get_default_session().run(assigns)


def sample_batch(exp_replay, batch_size, obs_ph, actions_ph, rewards_ph, next_obs_ph, is_done_ph):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
    return { 
        obs_ph:obs_batch, actions_ph:act_batch, rewards_ph:reward_batch,  next_obs_ph:next_obs_batch, is_done_ph:is_done_batch
    }

def train(args):
    env = make_env()
    env.reset()
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n

    '''
    for _ in range(50):
        obs, _, _, _ = env.step(env.action_space.sample())

    plt.title("Game image")
    plt.imshow(env.render(mode = "rgb_array"))
    plt.show()
    plt.title("Agent observation (4 frames left to right)")
    plt.imshow(obs.transpose([0,2,1]).reshape([state_dim[0],-1]));
    plt.show()
    '''

    ##network 
    tfc.reset_default_graph()
    sess = tfc.InteractiveSession()

    agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=0.5)
    sess.run(tfc.global_variables_initializer())
    ##agente sin entrenar
    print (evaluate(env, agent, n_games=1))
    ##experiencia
    target_network = DQNAgent("target_network", state_dim, n_actions)
    load_weigths_into_target_network(agent, target_network) 
    sess.run([tf.assert_equal(w, w_target) for w, w_target in zip(agent.weights, target_network.weights)]);

    obs_ph = tfc.placeholder(tf.float32, shape=(None,) + state_dim)
    actions_ph = tfc.placeholder(tf.int32, shape=[None])
    rewards_ph = tfc.placeholder(tf.float32, shape=[None])
    next_obs_ph = tfc.placeholder(tf.float32, shape=(None,) + state_dim)
    is_done_ph = tfc.placeholder(tf.float32, shape=[None])

    is_not_done = 1 - is_done_ph
    gamma = 0.99

    current_qvalues = agent.get_symbolic_qvalues(obs_ph, n_actions)
    current_action_qvalues = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_qvalues, axis=1)

    # compute q-values for NEXT states with target network
    next_qvalues_target = target_network.get_symbolic_qvalues(next_obs_ph, n_actions) #<your code> 

    # compute state values by taking max over next_qvalues_target for all actions
    next_state_values_target = tf.reduce_max(next_qvalues_target, axis=1) #<YOUR CODE>

    # compute Q_reference(s,a) as per formula above.
    reference_qvalues = rewards_ph + gamma * next_state_values_target #<YOUR CODE>

    # Define loss function for sgd.
    td_loss = (current_action_qvalues - reference_qvalues) ** 2
    td_loss = tf.reduce_mean(td_loss)

    train_step = tfc.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=agent.weights)

    sess.run(tfc.global_variables_initializer())

    mean_rw_history = []
    td_loss_history = []

    exp_replay = ReplayBuffer(10**5)
    play_and_record(agent, env, exp_replay, n_steps=10000)
    agent = loadLastState(env)
    moving_average = lambda x, span, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(span=span, **kw).mean().values
    num_episodes = args.train_dqn
    agent.epsilon = 1
    eps_or = agent.epsilon
    print("Training with %i episodes"  %num_episodes)
    for i in trange(num_episodes): #trange muestra progreso en %
        # play
        play_and_record(agent, env, exp_replay, 30)
        
        # train
        _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, 
                                                            batch_size=64, 
                                                            obs_ph=obs_ph, 
                                                            actions_ph=actions_ph,
                                                            rewards_ph=rewards_ph, 
                                                            next_obs_ph=next_obs_ph, 
                                                            is_done_ph=is_done_ph))
        td_loss_history.append(loss_t)
        
        # adjust agent parameters
        
        if i % 500 == 0 and i != 0:
            load_weigths_into_target_network(agent, target_network)
            agent.epsilon = max(agent.epsilon -  0.05 , 0.01)
            mean_rw_history.append(evaluate(make_env(), agent, n_games=3))
        
        if i % 1000 == 0 and i != 0:
            s = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
            clear_output(True)
            print("buffer size = %i, epsilon = %.5f" % (len(exp_replay), agent.epsilon))
            
            fig, (plt1, plt2) = plt.subplots(2)
            fig.suptitle('Breakout training')
            plt1.set_title("Mean reward per game")
            plt1.plot(mean_rw_history, 'tab:blue')
            plt1.set(xlabel='Games', ylabel  = 'Reward')
            plt1.grid()
            plt2.set_title("TD loss history (moving average)")
            plt2.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100), 'tab:red')
            plt2.set(xlabel='Episodes', ylabel  = 'Value')
            plt2.grid()
            fig.tight_layout()
            fig.savefig('./imgs/graphic_2_{}.png'.format(s))
            fig.clear()
        #if np.mean(mean_rw_history[-10:]) > 10.:
        #    break
        if i % 10000 == 0 or i == num_episodes-1:
            if i != 0 : 
                s = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
                agent.network.save_weights('./model/weights_{}.h5'.format(s + "eps_" + str(agent.epsilon)))


def loadLastState(env):
    files = [f for f in os.listdir("./model") if os.path.isfile(os.path.join("./model", f))]
    x = files[len(files)-1] #ultimo peso
    path = "./model/" + x

    print("Model path: ")
    print(path)
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=0)
    agent.network.load_weights(path)
    return agent

def test(args):
    
    env = make_env()
    if args.render:
        env = make_env(render = True)
    env.reset()
    
    agent = loadLastState(env)

    ##network 
    tfc.reset_default_graph()
    sess = tfc.InteractiveSession()
    
    agent.epsilon = 0
    n_games = 1
    sess.run(tfc.global_variables_initializer())
    result = evaluate(env, agent, n_games=n_games, render=args.render, test = True)
    print ("Reward mean: %i , games = %i" % (result,n_games))
    

if __name__ == '__main__':
    env = make_env()
    print(env.action_space)
    #env.sample_actions

    args = parse()
    if args.train_dqn:
        train(args)
    if args.test_dqn:
        test(args)

