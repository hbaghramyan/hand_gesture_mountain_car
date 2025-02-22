#!/bin/env python3.9
# Title: mountain-car-play.py
# Description:
#    Play MountainCar-v0 environment with hand gesture recognition.
# Refs:
#  1. https://www.gymlibrary.dev/
#  2. https://www.geeksforgeeks.org/pygame-event-handling/
####################################################################

import time
import sys

import pygame
import gym

from utils_classes import Car

# Create the MountainCarPlay-v0 custom environment by registering into the env
# register in gym/envs/__init__.py. Currently, the goal is not to solve the MountainCar-v0
# challenge using RL, but control the environment with handgestures. Since we want a smooth
# play feel, the 'max_episode_steps' are increased to 10.000 steps. See: https://github.com/openai/gym/wiki/FAQ
gym.envs.register(
    id="MountainCarPlay-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=10000,  # Regula MountainCar-v0 env uses 200
    reward_threshold=-110.0,
)

env = gym.make("MountainCarPlay-v0", render_mode="human")
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
