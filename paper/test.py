import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

x = np.linspace(-180, 180)
s = np.sin(np.deg2rad(x))
c = np.cos(np.deg2rad(x))
t = np.arctan2(s,c)
plt.plot(x,np.rad2deg(t))
plt.show()