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
from gym.utils.play import play

class Car():
    def __init__(self, env):
        """
        Initialize the Car class.
         
        Args: 
           env: MountainCar-v0 environment
        """
        # Action size = 3
        # Go left, Go right, Don't move
        self.action_size = env.action_space.n
        print(f'MountainCAr-v0 action size: {self.action_size}')
        
        # Define actions in a key map
        self.KEY_MAPPING = {
            pygame.K_LEFT: 0,  # Accelerating to the left: Action(0)
            pygame.K_RIGHT: 2, # Accelrating to the right: Action(2)
            pygame.K_DOWN: 1,  # No acceleration: Action(1)
        }

    def get_action(self, pressed_key):
        """
         Map pressed key to a corresponding action.

         Args:
            pressed_key: Key code of the pressed key

         Returns:
            int: Action corresponding to the pressed key
        """
        # Set default behavior to DOWN button. This way the mountain car won't move if
        # no signal is detected.
        return self.KEY_MAPPING.get(pressed_key, self.KEY_MAPPING[pygame.K_DOWN])
        
# Create the MountainCar-v0 environment
env = gym.make('MountainCar-v0', render_mode='human')
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
                
           # Get corresponding action from the Car class
           action = car.get_action(pressed_key)

           # Take a step in the environment
           observation, reward, terminated, truncated, info = env.step(action)
           
           # Render the environment
           env.render() 
           
           # Print the observation, reward, and info
           if terminated or truncated:
                print(f'Observation: {observation}, Reward: {reward}, Info: {info}')
                observation, info = env.reset()

# Pause to control loop speed
time.sleep(10)


