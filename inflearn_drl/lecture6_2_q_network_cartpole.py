import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf.disable_eager_execution()

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

X = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='input_x') # [[0,2,...]] 1x16
W1 = tf.get_variable('W1', shape=[input_size, output_size], initializer=tf2.truncated_normal_initializer(stddev=0.1))

Qpred = tf.matmul(X, W1)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 5000
dis = 0.9
rList = []

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    #env.render()
    e = 1./((i/10)+1)
    rAll = 0
    step_count = 0
    done = False

    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = -100
        else:
            x1 = np.reshape(s1, [1, input_size])
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
        sess.run(train, feed_dict={X:x, Y: Qs})
        s = s1
    rList.append(step_count)
    print('Episode: {} steps: {}'.format(i, step_count))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break
observation = env.reset()
reward_sum = 0
while True:
    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X: x})
    a = np.argmax(Qs)
    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print('total score: {}'.format(reward_sum))
        break