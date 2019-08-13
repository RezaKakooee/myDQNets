import gym
import random
import time
#%%
env_name = 'CartPole-v1'
env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

#%%
class Agent():
    def __init__(self, env):
        self.env = env
        self.action_size = self.env.action_space.n
        print('Action size:', self.action_size)
        
    def get_action(self, state):
#        action = random.choice(range(self.action_size))
        
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1 # 0=left, 1=right     
        
        return action
    
#%%
agent = Agent(env)
state = env.reset()

for i in range(205):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    print('i: {} reward: {}'.format(i, reward))

env.close()
