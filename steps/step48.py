if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np
import matplotlib.pyplot as plt

from dezero import datasets
from dezero import models
from dezero import optimizers
from dezero import functions as F

np.random.seed(0)
x, t = datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

model = models.MLP((hidden_size, 3))
optimizer = optimizers.MomentumSGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f'epoch {epoch + 1}, loss {avg_loss:.2f}')

for i in range(3):
    plt.scatter(x[t==i][:,0], x[t==i][:,1])
plt.show()

