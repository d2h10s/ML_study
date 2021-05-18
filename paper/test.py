import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

def _motor_profile(deg):
    t, a, b = MAX_TORQUE, AVAIL_DIRECTION[0], AVAIL_DIRECTION[1]
    trisection = (b-a)/3
    slope = t/trisection

    if deg < a + trisection:
        torque = slope*(deg-a)
    elif deg < b-(b-a)/3:
        torque = t
    else:
        torque = -slope*(deg-b)
    
    return torque

MAX_TORQUE = .7
AVAIL_DIRECTION = [-100, +10]
t, a, b = MAX_TORQUE, AVAIL_DIRECTION[0], AVAIL_DIRECTION[1]
print(a+(b-a)/2)
x = np.linspace(AVAIL_DIRECTION[0], AVAIL_DIRECTION[1], 1000)
y = [_motor_profile(xx) for xx in x]
plt.plot(x, y)
plt.grid(True)
plt.show()