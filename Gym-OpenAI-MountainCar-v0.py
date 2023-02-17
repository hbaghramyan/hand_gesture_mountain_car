import gym
import random
import numpy as np

#import sys
#!{sys.executable} -m pip install gym[all]

# En caso de solo querer instalar las dependencias de classic control utilizar:
#!{sys.executable} -m pip install gym[classic_control]

env_name = "MountainCar-v0"
env = gym.make(env_name, render_mode='human')
env.action_space.seed(42)
print('action_space.seed(42) = ', env.action_space.seed(42))
observation, info = env.reset(seed=42)

for i in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if (terminated or truncated):
        observation, info = env.reset(seed=42)
        print(observation)
env.close()