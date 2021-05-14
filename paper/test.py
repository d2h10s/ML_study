import tensorflow as tf
import numpy as np
import gym
import time

log_dir = './logs/test_'+str(time.ctime()[-13:-4])
log_dir = './logs/test_'+time.strftime('%m-%d_%H:%M:%S', time.localtime())

print(log_dir)