from typing import Optional, Sequence, Union

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


class Reshape(core.Function):
    def __init__(self, shape: Sequence[int]):
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return reshape(gy, self.x_shape)


def reshape(x: core.Variable, shape: Sequence[int]) -> core.Variable:
    if x.shape == shape:
        return core.as_variable(x)
    return Reshape(shape)(x)



class Transpose(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.transpose(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return transpose(gy)


def transpose(x: core.Variable) -> core.Variable:
    return Transpose()(x)


class Sum(core.Function):
    def __init__(
            self,
            axis: Optional[Union[int, Sequence[int]]],
            keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(
        x: core.Variable,
        axis: Optional[Union[int, Sequence[int]]],
        keepdims: bool = False) -> core.Variable:
    return Sum(axis, keepdims)(x)