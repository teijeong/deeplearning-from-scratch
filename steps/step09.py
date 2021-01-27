from typing import Optional, Union

import numpy as np


def as_array(x: Union[float, np.ndarray]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f'Unsupported type: {type(data)}')

        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func: 'Function'):
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Variable) -> Variable:
        """Calls fucntion and saves called data."""
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gy = np.exp(x) * gy
        return gy


def square(x: np.ndarray) -> np.ndarray:
    f = Square()
    return f(x)


def exp(x: np.ndarray) -> np.ndarray:
    return Exp()(x)


if __name__ == '__main__':
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)

    x = Variable(np.array(1.0))
    x = Variable(None)

    try:
        x = Variable(1.0)
    except TypeError as e:
        print('TypeError raised!', e)