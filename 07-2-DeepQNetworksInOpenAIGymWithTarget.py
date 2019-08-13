import gym
import random
import time
import numpy as np
import tensorflow as tf
from collections import deque

print('Gym:', gym.__version__)
print('TensorFlow', tf.__version__)
#%%
env_name = "CartPole-v0"
env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

#%% 
class QNetwork():
    def __init__(self, state_shape, action_size, tau=0.01):
        # since we are using scope, we need to have the following line to reset the graph before defining a new network architecture
        tf.reset_default_graph()
        self.state_in = tf.placeholder(dtype=tf.float32, shape=[None, *state_shape])
        self.action_in = tf.placeholder(dtype=tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(dtype=tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        self.q_state_local  = self.bulid_model(action_size, 'local')
        self.q_state_target = self.bulid_model(action_size, 'target')
        
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state_local, action_one_hot), axis=1)
        
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
    
        # get each net vars
        self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='local')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        self.updater = tf.group([tf.assign(t, t+tau*(l-t)) for t,l in zip(self.target_vars, self.local_vars)])
        
        
    def bulid_model(self, action_size, scope):
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
            q_state= tf.layers.dense(hidden1, action_size, activation=None)
        return q_state
        
    def update_model(self, sess, state, action, q_target):
        feed = {self.state_in:state, self.action_in: action, self.q_target_in:q_target}
        sess.run([self.optimizer, self.updater], feed_dict=feed)
        
    def get_q_state(self, sess, state, use_target=False):
        q_state_op = self.q_state_target if use_target else self.q_state_local
        q_state = sess.run(q_state_op, feed_dict={self.state_in:state})
        return q_state
    
#%%
class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))
    
#%%%
class DQNAgent():
    def __init__(self, env, gamma=0.97, alpha=0.01, buffer_size=1000):
        
        self.epsilon = 1.0
        self.gamma = gamma
        self.alpha = alpha
        
        self.replay_buffer = ReplayBuffer(maxlen=buffer_size)
        
        self.state_shape = env.observation_space.shape
        self.action_size = env.action_space.n
        
        self.q_network = QNetwork(self.state_shape, self.action_size)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_gready = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        return action_random if random.random() < self.epsilon else action_gready
    
    def train(self, experience, batch_size=50):
        self.batch_size = batch_size
        self.replay_buffer.add(experience)
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)        
        q_next_states = self.q_network.get_q_state(self.sess, next_states, use_target=True)
        q_next_states[dones] = np.zeros([self.action_size])
        q_maxs = np.max(q_next_states, axis=1)
        q_targets = rewards + self.gamma * q_maxs
        self.q_network.update_model(self.sess, states, actions, q_targets)

        if done:
            self.epsilon = max(0.99 * self.epsilon, 0.01)
        
    def __del__(self):
        self.sess.close()
#%%
agent = DQNAgent(env)
total_reward = []

#%%
n_episodes = 400
for ep in range(n_episodes):
    episode_reward = []
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        experience = (state, action, next_state, reward, done)
        agent.train(experience)
        
        episode_reward.append(reward)
        
        state = next_state
        
#        env.render()
#        time.sleep(0.01)
        
#        print('Current state-action pair is: ({}, {})'.format(state, action))
    total_reward.append(np.sum(episode_reward))
    if (ep + 1) % 10 == 0: 
        print('Episode: {}, Episode Reward: {}'.format(ep+1, np.sum(episode_reward)))

print('The sum of all episodes reward: ', np.sum(total_reward))
env.close()
#with tf.variable_scope("q_table", reuse=True):
#            weights = agent.sess.run(tf.get_variable("kernel"))
#            print(weights)
