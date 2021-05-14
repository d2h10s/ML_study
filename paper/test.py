import tensorflow as tf
import numpy as np
import gym
import time

env = gym.make('Acrobot-v1')
env.reset()
while True:
    env.step(1)
    env.render()