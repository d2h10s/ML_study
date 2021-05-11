import numpy as np
from tensorflow.python.ops.logging_ops import get_summary_op
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import gym
import matplotlib.pyplot as plt
import os
tf.disable_eager_execution()

log_dir = os.path.join() 

env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32) # [[0,2,...]] 1x16
W = tf.Variable(tf.random.uniform(shape=[input_size, output_size], minval=0, maxval=0.01)) # 16 x 4

Qpred = tf.matmul(X, W) # [1x16]@[16x4]=[1x4]
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99
num_episode = 2000
rList = []

def one_hot(x):
    return np.identity(16)[x:x+1]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episode):
        state = env.reset()
        rAll = 0
        e = 1./((i/50)+10)
        done = False
        local_loss = []

        while not done:
            Qs = sess.run(Qpred, feed_dict={X:one_hot(state)})
            action = np.argmax(Qs+np.random.randn(1, output_size)[0] / (i + 1))
            '''if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)'''

            new_state, reward, done, _ = env.step(action)

            if done:
                Qs[0, action] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(new_state)})
                Qs[0, action] = reward + dis * np.max(Qs1)
            
            sess.run(train, feed_dict={X: one_hot(state), Y:Qs})

            rAll += reward
            state = new_state
        rList.append(rAll)
        tf.summary.scalar(rAll)
    merged = tf.summary.mergge_all()
    writer = tf.summary.FileWriter('/logs', sess.graph)
    summary = sess.run(merged, feed_dict={X:x, Y:y_data})
plt.style.use('classic')
print('percent of successful episods: {}%'.format(sum(rList)/num_episode))
plt.bar(range(len(rList)), rList)