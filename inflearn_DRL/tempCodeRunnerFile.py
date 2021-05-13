import numpy as np
import tensorflow as tf
import gym

env = gym.make('FrozenLake-v0')
env.reset()

input_size = env.observation_space.n
output_size = env.action_space.n

X = tf.Variable()
W = tf.Variable(tf.random.uniform(shape=[input_size, output_size], minval=0, maxval=0.01))
print(W)

Q_prediction = tf.matmul(X, W)