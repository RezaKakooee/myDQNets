import gym
import random
import time
import numpy as np
#%%
env_name = "CartPole-v1"
env_name = "MountainCar-v0"
env_name = "MountainCarContinuous-v0"
env_name = "Acrobot-v1"
env_name = "Pendulum-v0"
env_name = "FrozenLake-v0"

env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

#%%
class Agent():
    def __init__(self, env):
        self.env = env
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete
        
        if self.is_discrete:
            self.action_size = self.env.action_space.n
            print('Action size: ', self.action_size)
        else:
            self.action_low = self.env.action_space.low
            self.action_hight = self.env.action_space.high
            self.action_shape = self.env.action_space.shape
            print('Action range is {} to {}. '.format(self.action_low, self.action_hight))
            
    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, 
                                       self.action_hight, 
                                       self.action_shape)
        
        return action
    
#%%
agent = Agent(env)
state = env.reset()

for i in range(205):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
    print('i: {} reward: {}'.format(i, reward))

env.close()
