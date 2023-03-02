#!/bin/env python3.9
# Title: mountain-car-play.py
# Description: 
#    Play MountainCar-v0 environment with hand gesture recognition.
# Refs:
#  1. https://www.gymlibrary.dev/
#  2. https://www.geeksforgeeks.org/pygame-event-handling/
####################################################################

import pygame
import gym
from gym.utils.play import play
import time
import sys

class Car():
    def __init__(self, env):
        # Action size = 3
        # Go left, Go right, Don't move
        self.action_size = env.action_space.n
        print(f'MountainCAr-v0 action size: {self.action_size}')

        # Define actions
        self.K_LEFT  = 0 # Accelerating to the left: Action(0)
        self.K_RIGHT = 2 # Accelrating to the right: Action(2)
        self.K_DOWN  = 1 # No acceleration: Action(1)

    def get_action(self, pressed_key):
        # Handle the pressed key actions
        # return the current action mapped to the pressed key
        if pressed_key == self.K_LEFT:
            action = self.K_LEFT
        elif pressed_key == self.K_RIGHT:
            action = self.K_RIGHT
        else:
            action = self.K_DOWN
        
        return action
        

env = gym.make('MountainCar-v0', render_mode='human')
env.reset()
car = Car(env)
env.render()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
           if event.key == pygame.K_LEFT:
                print('LEFT KEY PRESSED')
                pressed_key = 0
                action = car.get_action(pressed_key)
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()
           if event.key == pygame.K_RIGHT:
                print('RIGHT KEY PRESSED')
                pressed_key = 2
                action = car.get_action(pressed_key)
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()
           if event.key == pygame.K_DOWN:
                print('DOWN KEY PRESSED')
                pressed_key = 1
                action = car.get_action(pressed_key)
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()
           if terminated or truncated:
                print(f'Observation: {observation}, Reward: {reward}, Info: {info}')
                observation, info = env.reset()
time.sleep(10)


