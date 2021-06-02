import tensorflow as tf
import numpy as np
import gym
import time, os
import matplotlib.pyplot as plt
original_path = os.path.join(os.curdir,'logs', 'Acrobot-v2_05-28_13h-56m-33s_fail')
original_model_path = os.path.join(original_path, 'tf_model')

li = os.listdir(original_model_path)
M=li[np.argmax([int(x[13:]) for x in li])]
print(M)