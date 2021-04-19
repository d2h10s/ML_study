import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def softmax2(x):
    c = np.max(x)
    expx = np.exp(x - c)
    return expx/np.sum(expx)


x = np.linspace(-10, 10, 200)
y = step_function(x)
plt.plot(x,y, 'r')
y = sigmoid(x)
plt.plot(x,y, 'g')
y = ReLU(x)
plt.plot(x,y, 'b')
plt.legend(['step', 'sigmoid', 'ReLU'])
plt.show()