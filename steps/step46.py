if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import dezero as dz
from dezero import functions as F
from dezero import layers as L
from dezero import models
from dezero import optimizers

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = models.MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()
    optimizer.update()
    
    if i % 1000 == 0:
        print(loss)

import matplotlib.pyplot as plt

plt.scatter(x, y)
x_data = np.linspace(0, 1, 100).reshape(100, 1)
y = model(dz.as_variable(x_data))
plt.plot(x_data, y.data)
plt.show()