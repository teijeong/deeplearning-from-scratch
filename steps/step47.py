if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Union

import numpy as np

import dezero as dz
from dezero import functions as F
from dezero import layers as L
from dezero import models
from dezero import optimizers

np.random.seed(0)
model = models.MLP((10, 3))


def softmax1d(x: Union[np.ndarray, dz.Variable]) -> dz.Variable:
    x = dz.as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


x = np.array([[0.2, -0.4]])
y = model(x)
p = softmax1d(y)
print(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy(y, t)
print(loss)