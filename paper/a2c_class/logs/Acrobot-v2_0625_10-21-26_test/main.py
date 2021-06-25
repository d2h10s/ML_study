import os,shutil
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


def file_backup(log_dir):
    file_name_list = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if '.py' in x]
    file_path_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)),x) for x in file_name_list if '.py' in x]
    for fname, fpath in zip(file_name_list, file_path_list):
        shutil.copy(src=fpath, dst=os.path.join(log_dir,fname))

INIT_MESSAGE = '''
using acrobot-v2 environment which is d2h10s edition v3.0
definition of reward : [reward = -abs(cos(theta_1))]
termination condition: [None]
'''

env = gym.make('Acrobot-v2')

SEED = 3
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

observation_n = env.observation_space.shape[0]
hidden_n = 128
action_n = env.action_space.n

model = a2c_model(observation_n, hidden_n, action_n)
agent = a2c_agent(model, lr=1e-3, sampling_time=0.025, suffix="_test")
agent.init_message(INIT_MESSAGE)
file_backup(agent.log_dir)

agent.train(env)
agent.run_test()

