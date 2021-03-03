if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

import dezero as dz
from dezero import functions as F
from dezero import utils


x = dz.Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 4

for _ in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = f'gx{iters + 1}'
graph_dot = utils.get_dot_graph(gx, verbose=False)

with open('tanh.dot', 'w') as f:
    f.write(graph_dot)