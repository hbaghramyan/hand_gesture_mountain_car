# Description: Play MountainCar-v0 environment with hand gesture recognition.
# Refs:
#  1. https://www.gymlibrary.dev/
#  2. https://www.geeksforgeeks.org/pygame-event-handling/
####################################################################

# native dependencies
import time
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

# Instantiate the Car class
car = Car(env)

# Render the environment
env.render()

# Main loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            # Check for key presses
            pressed_key = event.key
            # print(pressed_key)
            # Get corresponding action from the Car class
            action = car.get_action(pressed_key)
            print(action)
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Render the environment
            env.render()

            # Print the observation, reward, and info
            if terminated or truncated:
                # print(f"Observation: {observation}, Reward: {reward}, Info: {info}")
                observation, info = env.reset()

# Pause to control loop speed
time.sleep(5)
