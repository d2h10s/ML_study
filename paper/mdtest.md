Q러닝 기반의 네트워크이므로 모든 state와 action을 표현하여야 가장 정확하다.

(𝑠, 𝑎) 쌍의 Q-가치를 근사하는 approximate Q-learning을 사용하여 적절한 개수의 파라미터를 찾으면 가능하다.


딥마인드팀은 이를 기반하여 Deep Q-Network를 사용하는 Deep Q-Learning을 사용하였다.


## DQN이 제시하는 Q-learning의 한계 돌파법

1. Go Deep
1. Capture & replay – correlations between samples
1. Separate networks – create a target network

![](https://images.velog.io/images/d2h10s/post/8afddddc-663b-4707-8bd4-289336cc9ea6/image.png)

```python
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
input_shape = [4]
n_outputs = 2
replay_buffer = deque(maxlen=2000)

model = keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=input_shape),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(n_outputs)])

def epsilon_greedy_policy(state, elsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size = batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = \
        [np.array([experience[field_index] for experience in batch]) \
            for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

all_reward = []
for episode in range(600):
    print('episode {}'.format(episode))
    obs = env.reset()
    sum_reward = 0
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        sum_reward += reward
        if done:
            break
    all_reward.append(sum_reward)
    if episode > 50:
        training_step(batch_size)

plt.plot(range(len(all_reward)), all_reward)
plt.show()
```