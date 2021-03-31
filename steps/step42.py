if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import dezero as dz
from dezero import functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = dz.Variable(x), dz.Variable(y)

W = dz.Variable(np.zeros((1, 1)))
b = dz.Variable(np.zeros(1))

def predict(x: dz.Variable) -> dz.Variable:
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0: dz.Variable, x1: dz.Variable) -> dz.Variable:
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


LR = 0.1
NUM_ITER = 100

for _ in range(NUM_ITER):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    x.cleargrad()
    loss.backward()

    W.data -= LR * W.grad.data
    b.data -= LR * b.grad.data
    print(W, b, loss)