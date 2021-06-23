import sys, io, os, yaml, shutil
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from pytz import timezone, utc
from datetime import datetime as dt
from A2C_AGENT import a2c_agent
from A2C_MODEL import a2c_model

env = gym.make('Acrobot-v2')

SEED = 3
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

observation_n = env.observation_space.shape[0]
hidden_n = 128
action_n = env.action_space.n

print(observation_n)
model = a2c_model(observation_n, hidden_n, action_n)
agent = a2c_agent(model, sampling_time=0.025, suffix="test")
agent.train(env)

