if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import dezero as dz
from dezero import functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sum(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = dz.Variable(0.01 * np.random.randn(I, H))
b1 = dz.Variable(np.zeros(H))
W2 = dz.Variable(0.01 * np.random.randn(H, O))
b2 = dz.Variable(np.zeros(O))

def predict(x: dz.Variable) -> dz.Variable:
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


LR = 0.2
NUM_ITER = 20

for i in range(NUM_ITER):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= LR * W1.grad.data
    b1.data -= LR * b1.grad.data
    W2.data -= LR * W2.grad.data
    b2.data -= LR * b2.grad.data
    if i % 1 == 0:
        print(loss)


x = np.linspace(0, 1, 100).reshape(100, 1)
y = predict(dz.Variable(x)).data

import matplotlib.pyplot as plt
plt.plot(x, y)