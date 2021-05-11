import gym
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


env = gym.make('Acrobot-v1')
state = env.reset()
while True:
    action = np.random.randint(env.action_space.n)
    env.step(action)
    #time.sleep(0.1)