import gym
import random
import time
import numpy as np
from gym.envs.registration import register
import tensorflow as tf
from collections import deque
#%%
try:
    register(
        id='FrozenLakeNoSlip-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery':False},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    pass

#%%
#env_name = "FrozenLake-v0"
env_name = "FrozenLakeNoSlip-v0"
#env_name = "CartPole-v1"
#env_name = "MountainCar-v0"
#env_name = "MountainCarContinuous-v0"
#env_name = "Acrobot-v1"
#env_name = "Pendulum-v0"
#env_name = "FrozenLake-v0"
env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

#%%
class Agent():
    def __init__(self, env):
        self.env = env
        self.is_discrete_action = \
            type(env.action_space) == gym.spaces.discrete.Discrete
        
        self.is_discrete_space = \
            type(env.observation_space) == gym.spaces.discrete.Discrete
        
        if self.is_discrete_action:
            self.action_size = self.env.action_space.n
            print('Action size: ', self.action_size)
            
        else:
            self.action_low = self.env.action_space.low
            self.action_hight = self.env.action_space.high
            self.action_shape = self.env.action_space.shape
            print('Action range is {} to {}. And action shape is {}.'.format(self.action_low, self.action_hight, self.action_shape))
            
        if self.is_discrete_space:
            self.state_size = env.observation_space.n
            print("State size:", self.state_size)
            
        else:
            self.state_low = self.env.observation_space.low
            self.state_hight = self.env.observation_space.high
            self.state_shape = self.env.observation_space.shape
            print('State range is {} to {}. And state shape is {}.'\
                  .format(self.state_low, self.state_hight, self.state_shape))
        
    def get_action(self, state):
        if self.is_discrete_action:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, 
                                       self.action_hight, 
                                       self.action_shape)
        return action

#%% 
class QNAgent(Agent):
    def __init__(self, env, gamma=0.97, alpha=0.001, buffer_size=1000):
        super().__init__(env)
        
        self.epsilon = 1.0
        self.gamma = gamma
        self.alpha = alpha
        self.buffer_size = buffer_size
        
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        self.build_model()
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def build_model(self):
        tf.reset_default_graph()
        self.state_in = tf.placeholder(tf.int32, shape=[None])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.target_in = tf.placeholder(tf.float32, shape=[None])
        
        if self.is_discrete_space:
            print('The state space in discrete')
            self.state = tf.one_hot(self.state_in, depth=self.state_size)
            
        if self.is_discrete_action:
            print('The action space in discrete')
            self.action = tf.one_hot(self.action_in, depth=self.action_size)
            
        self.q_state = tf.layers.dense(self.state, units=self.action_size, name='q_table')
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)
            
        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)
    
    def get_action(self, state):
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in:[state]})
        action_gready = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.epsilon else action_gready
    
    def train(self, experience, batch_size=50):
        self.batch_size = batch_size
        self.replay_buffer.append(experience)
        samples = random.choices(self.replay_buffer, k=self.batch_size)
        
        state, action, next_state, reward, done = \
            (list(col) for col in zip(experience, *samples))
        
        q_next = self.sess.run(self.q_state, feed_dict={self.state_in:next_state})
        q_next[done] = np.zeros([self.action_size])
        q_max = np.max(q_next, axis=1)
        q_target = reward + self.gamma * q_max
        
        feed = {self.state_in: state, 
                self.action_in: action, 
                self.target_in: q_target}
        self.sess.run(self.optimizer, feed_dict=feed)
        
        if experience[4]:
            self.epsilon = 0.99 * self.epsilon
        
    def __del__(self):
        self.sess.close()
#%%
agent = QNAgent(env)
total_reward = []

#%%
for ep in range(100):
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
        
        env.render()
        time.sleep(0.01)
        
#        print('Current state-action pair is: ({}, {})'.format(state, action))
    total_reward.append(np.sum(episode_reward))
#    if (ep + 1) % 100 == 0: 
#        print('Episode: {}, Episode Reward: {}'.format(ep+1, np.sum(episode_reward)))
#    with tf.variable_scope("q_table", reuse=True):
#            weights = agent.sess.run(tf.get_variable("kernel"))
#            print(weights)
print('The sum of all episodes reward: ', np.sum(total_reward))
env.close()

