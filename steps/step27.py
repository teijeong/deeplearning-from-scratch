if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import dezero as dz
from dezero import utils
import numpy as np

class Sin(dz.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x: dz.Variable) -> dz.Variable:
    return Sin()(x)


def my_sin(x: dz.Variable, threshold=0.0001) -> dz.Variable:
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == '__main__':
    x = dz.Variable(np.array(np.pi/4))
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)

    x = dz.Variable(np.array(np.pi/4))
    y = my_sin(x)
    y.backward()

    print(y.data)
    print(x.grad)

    print(utils.get_dot_graph(y, verbose=False))