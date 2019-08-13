import gym
import random
import time
import numpy as np
import tensorflow as tf

#%%
env_name = "CartPole-v0"
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
            print('Discrete action size: ', self.action_size)
            
        else:
            self.action_low = self.env.action_space.low
            self.action_hight = self.env.action_space.high
            self.action_size = self.env.action_space.shape
            print('Continues action range is {} to {}. And action shape is {}.' \
                  .format(self.action_low, self.action_hight, self.action_size))
            
        if self.is_discrete_space:
            self.state_shape = env.observation_space.n
            print("Discrete state size:", self.state_shape)
            
        else:
            self.state_low = self.env.observation_space.low
            self.state_hight = self.env.observation_space.high
            self.state_shape = self.env.observation_space.shape
            print('Continues state range is {} to {}. And state shape is {}.'\
                  .format(self.state_low, self.state_hight, self.state_shape))
        
    def get_action(self, state):
        if self.is_discrete_action:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, 
                                       self.action_hight, 
                                       self.action_size)
        return action
    
#%%
agent = Agent(env)
total_reward = []

#%%
n_episodes = 205
for ep in range(n_episodes):
    episode_reward = []
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        episode_reward.append(reward)
        
        env.render()
        time.sleep(0.01)
        
        if (ep+1) % 50 == 0:
            print('Episode: {}, Reward: {}'.format(ep, reward))
            
        state = next_state
        
    total_reward.append(np.sum(episode_reward))
print('The sum of all episodes reward: ', np.sum(total_reward))
env.close()
