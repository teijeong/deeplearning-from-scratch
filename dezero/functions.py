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