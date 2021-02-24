if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero as dz
from dezero import utils


def goldstein(x: dz.Variable, y: dz.Variable) -> dz.Variable:
    z = 1 + (x + y + 1) ** 2 *  (19 - 14 * x + 3 * x ** 2 - 14 * y +
                                 6 * x * y + 3 * y ** 2)
    z = z * (30 + (2 * x - 3 * y) ** 2 *
                  (18 - 32 * x + 12 * x ** 2 + 48 * y -
                   36 * x * y + 27 * y ** 2))
    return z

x = dz.Variable(np.array(1.0))
y = dz.Variable(np.array(1.0))
z = goldstein(x, y)

x.name = 'x'
y.name = 'y'
z.name = 'z'
print(utils.get_dot_graph(z, verbose=False))