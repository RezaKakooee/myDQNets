import gym
import random
import time
import numpy as np

#%%
env_name = "CartPole-v0"
env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

#%%
class HillClimbingAgent():
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
            print('Action range is {} to {}. And action shape is {}.'\
                  .format(self.action_low, self.action_hight, self.action_shape))
            
        if self.is_discrete_space:
            self.state_size = env.observation_space.n
            print("State size:", self.state_size)
            
        else:
            self.state_low = self.env.observation_space.low
            self.state_hight = self.env.observation_space.high
            self.state_shape = self.env.observation_space.shape
            print('State range is {} to {}. And state shape is {}.'\
                  .format(self.state_low, self.state_hight, self.state_shape))
        
        self.build_model()
    def build_model(self):
        if self.is_discrete_action and not self.is_discrete_space:
            self.weights = 1e-4 * np.random.rand(*self.state_shape, self.action_size)
        self.best_weights = np.copy(self.weights)
        self.best_reward = -np.inf
        self.noise_scale = 1e-2
        
    def get_action(self, state):
        p = np.dot(state, self.weights)
        action = np.argmax(p)
        return action
    
    def update_model(self, reward):
        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_weights = np.copy(self.weights)
            self.noise_scale = max(self.noise_scale/2, 1e-3)
        else:
            self.noise_scale = min(self.noise_scale*2, 2)

        self.weights = self.best_weights + self.noise_scale * \
                    np.random.rand(*self.state_shape, self.action_size)

#%%
agent = HillClimbingAgent(env)
total_reward = []

#%%
n_episodes = 100
for ep in range(n_episodes):
    episode_reward = []
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        experience = (state, action, next_state, reward, done)
        
        state = next_state
        episode_reward.append(reward)
        
    agent.update_model(np.sum(episode_reward))
        
#        env.render()
#        time.sleep(0.01)
        
#        print('Current state-action pair is: ({}, {})'.format(state, action))
    total_reward.append(np.sum(episode_reward))
    print('Episode: {}, Episode Reward: {}'.format(ep, np.sum(episode_reward)))

print('The sum of all episodes reward: ', np.sum(total_reward))
env.close()
