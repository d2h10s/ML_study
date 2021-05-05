import numpy as np
import tensorflow as tf

transition_probabilities = [  # Shape = [s, a, s'], 행동 a1를 한 후 s2에서 s0'으로 갈 확률 transition_probabilities[2][1][0]
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None,            [0.0, 0.0, 1.0]],
    [None,            [0.8, 0.1, 0.1], None]]

rewards = [  # shape = [s, a, s']
    [[10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [40, 0, 0], [0, 0, 0]]]

possible_actions = [[0, 1, 2], [0, 2], [1]]

Q_values = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0

gamma = 0.9  # 할인 계수

for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp]
                * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp])) for sp in range(3)
            ])
print(Q_values)

np.argmax(Q_values)