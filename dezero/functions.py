from typing import Optional, Sequence, Union

from dezero import core
from dezero import utils
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


class Exp(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x: core.Variable) -> core.Variable:
    return Exp()(x)


class Log(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.log(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy / x
        return gx


def log(x: core.Variable) -> core.Variable:
    return Log()(x)


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
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False) -> core.Variable:
    return Sum(axis, keepdims)(x)


class SumTo(core.Function):
    def __init__(
            self, shape: Sequence[int]):
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        gx = sum_to(gy, self.x_shape)
        return gx


def sum_to(x: core.Variable, shape: Sequence[int]) -> core.Variable:
    if x.shape == shape:
        return core.as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(core.Function):
    def __init__(
            self, shape: Sequence[int]):
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: core.Variable, shape: Sequence[int]) -> core.Variable:
    if x.shape == shape:
        return core.as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(core.Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x: core.Variable, W: core.Variable) -> core.Variable:
    return MatMul()(x, W)


class MeanSquaredError(core.Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0: core.Variable, x1: core.Variable) -> core.Variable:
    return MeanSquaredError()(x0, x1)


class Linear(core.Function):
    def forward(
        self,
        x: np.ndarray,
        W: np.ndarray,
        b: Optional[np.ndarray]) -> np.ndarray:
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(
    x: core.Variable,
    W: core.Variable,
    b: Optional[core.Variable]=None) -> core.Variable:
    return Linear()(x, W, b)


def linear_simple(
    x: core.Variable,
    W: core.Variable,
    b: Optional[core.Variable]=None) -> core.Variable:
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # Release t.data (ndarray) for memory efficiency
    return y


def sigmoid_simple(x: Union[np.ndarray, core.Variable]) -> core.Variable:
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(core.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # y = 1 / (1 + xp.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x: core.Variable) -> core.Variable:
    return Sigmoid()(x)


class GetItem(core.Function):
    def __init__(self, slices) -> None:
        self.slices = slices

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x[self.slices]
        return y

    def backward(self, gy: np.ndarray):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(core.Function):
    def __init__(self, slices: Sequence, in_shape: Sequence[int]) -> None:
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy: np.ndarray) -> np.ndarray:
        gx = np.zeros(self.in_shape, dtype=gy.dtype)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx: np.ndarray) -> np.ndarray:
        return get_item(ggx, self.slices)


def get_item(x: core.Variable, slices) -> core.Variable:
    f = GetItem(slices)
    return f(x)



def softmax_simple(x: Union[np.ndarray, core.Variable], axis=1) -> core.Variable:
    x = dz.as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(core.Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x: core.Variable, axis=1) -> core.Variable:
    return Softmax(axis)(x)


def softmax_cross_entropy_simple(
    x: Union[np.ndarray, core.Variable],
    t: Union[np.ndarray, core.Variable]):
    x, t = core.as_variable(x), core.as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(core.Function):
    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x: core.Variable, t: core.Variable) -> core.Variable:
    return SoftmaxCrossEntropy()(x, t)

class Clip(core.Function):
    def __init__(self, x_min: float, x_max: float) -> None:
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x: core.Variable, x_min: float, x_max: float) -> core.Variable:
    return Clip(x_min, x_max)(x)