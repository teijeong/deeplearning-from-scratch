if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import dezero as dz
from dezero import functions as F

x0 = dz.Variable(np.array([1, 2, 3]))
x1 = dz.Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x0.grad, x1.grad)