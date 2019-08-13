import gym
import random
import time
import numpy as np
from gym.envs.registration import register

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
env_name = "FrozenLake-v0"
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
class QAgent(Agent):
    def __init__(self, env, gamma=0.97, alpha=0.01):
        super().__init__(env)
        
        self.epsilon = 1.0
        self.gamma = gamma
        self.alpha = alpha
        
        self.build_model()
        
    def build_model(self):
        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])
    
    def get_action(self, state):
        q_state = self.q_table[state]
        action_gready = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.epsilon else action_gready
    
    def train(self, experience):
        state, action, next_state, reward, done = experience
        
        q_next = self.q_table[next_state] * (1 - done)
        q_max = np.max(q_next)
        q_target = reward + self.gamma * q_max
        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * q_update
        
        if done:
            self.epsilon = 0.99 * self.epsilon
        
#%%
agent = QAgent(env)
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
        
#        env.render()
#        time.sleep(0.01)
        
#        print('Current state-action pair is: ({}, {})'.format(state, action))
    total_reward.append(np.sum(episode_reward))
    print('Episode: {}, Episode Reward: {}'.format(ep, np.sum(episode_reward)))

print('The sum of all episodes reward: ', np.sum(total_reward))
env.close()
print(agent.q_table)
