if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero as dz
import numpy as np

_NUM_ITERS = 10


def f(x: dz.Variable) -> dz.Variable:
    y = x ** 4 - 2 * x ** 2
    return y


if __name__ == '__main__':
    x = dz.Variable(np.array(2.0))

    for i in range(_NUM_ITERS):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        x.cleargrad()
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data

    print(x)