from dezero import core
import numpy as np

class Sin(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x: core.Variable) -> core.Variable:
    return Sin()(x)


class Cos(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.cos(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x: core.Variable) -> core.Variable:
    return Cos()(x)


class Tanh(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)
        return gx


def tanh(x: core.Variable) -> core.Variable:
    return Tanh()(x)
