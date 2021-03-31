from typing import List, Sequence

import dezero as dz
from dezero import functions as F
from dezero import layers as L
from dezero import utils

class Model(dz.Layer):
    def plot(self, *inputs: dz.Variable, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(
            y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(
        self,
        fc_output_sizes: Sequence[int],
        activation=F.sigmoid) -> None:
        super().__init__()
        self.activation = activation
        self.layers: List[dz.Layer] = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x: dz.Variable) -> dz.Variable:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)