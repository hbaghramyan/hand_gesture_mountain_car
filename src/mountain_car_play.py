# Description: Play MountainCar-v0 environment using the keyboard.
# Left arrow key: Accelerate left
# Down arrow key: Don't accelerate
# Right arrow key: Accelerate right
# Refs:
#  1. https://www.gymlibrary.dev/
#  2. https://www.geeksforgeeks.org/pygame-event-handling/
####################################################################

# native dependencies
import sys
import os

# third-party dependencies
import pygame
import gym
from dotenv import load_dotenv
from omegaconf import OmegaConf

# local dependencies
from utils_classes import Car

load_dotenv()

config_path = os.getenv("CONFIG_PATH", None)

# loading the config file
conf = OmegaConf.load(config_path)

# get the necessary configurations
id, render, entry, steps, threshold = (
    conf.car_env.id,
    conf.car_env.render,
    conf.car_env.entry,
    conf.car_env.steps,
    conf.car_env.threshold,
)

# create the MountainCarPlay-v0 custom environment by registering into the env
gym.envs.register(
    id=id,
    entry_point=entry,
    max_episode_steps=steps,
    reward_threshold=threshold,
)

env = gym.make(id, render_mode=render)
env.reset()

# instantiate the Car class
car = Car(env)

# render the environment
env.render()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            pressed_key = event.key
            action = car.get_action(pressed_key)
            print(action)
            # take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Render the environment
            env.render()

            # Print the observation, reward, and info
            if terminated or truncated:
                env.reset()
