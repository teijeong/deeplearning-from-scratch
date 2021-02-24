if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as dz
import numpy as np

_NUM_ITER = 10


def f(x: dz.Variable) -> dz.Variable:
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x: np.ndarray) -> np.ndarray:
    return 12 * x ** 2 - 4


if __name__ == '__main__':
    x = dz.Variable(np.array(2.0))

    for i in range(_NUM_ITER):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward()

        x.data -= x.grad / gx2(x.data)
