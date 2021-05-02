import numpy as np
import gym
import random
from gym.envs.registration import register
import matplotlib.pyplot as plt

def rargmax(v):
    m = np.amax(v) # max 함수의 별칭이다.
    indices = np.nonzero(v == m)[0] # 0이 아닌 값들을 반환한다. 행과 열 튜플이 하나씩 반환된다.
    return random.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':False}
)
env = gym.make('FrozenLake-v3')
Q = np.zeros((env.observation_space.n, env.action_space.n))
num_episodes = 2000
gamma = 0.99
rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state
    rList.append(rAll)

print('Success rate: {}'.format(sum(rList)/num_episodes))
print('Final Q-Table Value')
print('LEFT DOWN RIGHT UP')
print(Q)
for i in range(4):
    print('   ', end='')
    for j in range(4): # up
        print('{:0.2f}'.format(Q[i*4+j, 3]), end='       ')
    print()
    for j in range(4):
        print('{:0.2f}'.format(Q[i*4+j, 0]), end=' ')
        print('{:0.2f}'.format(Q[i*4+j, 2]), end='  ')
    print('\n   ', end='')
    for j in range(4):
        print('{:0.2f}'.format(Q[i*4+j, 1]), end='       ')
    print('\n')


plt.bar(range(len(rList)), rList, color='blue')
plt.show()