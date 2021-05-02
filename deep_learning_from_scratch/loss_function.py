import numpy as np
import matplotlib.pyplot as plt


# Mean Squared Error
def MSE(y, t):
    return np.sum(np.power(y-t,2))/len(y)


# Cross Entropy Error
def CEE(y, t):
    delta = 1e-7 # 무한대로 발산하지 않게 하기 위함.
    return -np.sum(t*np.log(y+delta)) # 자연로그


y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
print(MSE(y, t))

y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(MSE(y, t))