import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('FrozenLake-v0')
obs = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break