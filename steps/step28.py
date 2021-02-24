if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import dezero as dz
import numpy as np


_LEARNING_RATE = 1e-3
_NUM_ITER = 1000


def rosenbrock(x0: dz.Variable, x1: dz.Variable) -> dz.Variable:
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y


if __name__ == '__main__':
    x0 = dz.Variable(np.array(0.0))
    x1 = dz.Variable(np.array(2.0))

    y = rosenbrock(x0, x1)
    y.backward()
    print(x0.grad, x1.grad)

    for i in range(_NUM_ITER):
        if i % (_NUM_ITER // 100) == 0:
            print(x0, x1)
        y = rosenbrock(x0, x1)
        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= _LEARNING_RATE * x0.grad
        x1.data -= _LEARNING_RATE * x1.grad