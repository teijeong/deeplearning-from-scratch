from typing import Sequence, Tuple

import numpy as np

from step09 import as_array

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
    def __call__(self, inputs: Sequence[Variable]) -> Variable:
        """Calls fucntion and saves called data."""
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.input = inputs
        self.output = outputs
        return outputs

    def forward(self, xs: Sequence[np.ndarray]) -> Tuple[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: Sequence[np.ndarray]) -> Tuple[np.ndarray]:
        raise NotImplementedError()

class Add(Function):
    def forward(self, xs: Sequence[np.ndarray]) -> Tuple[np.ndarray]:
        x0, x1 = xs
        y = x0 + x1
        return (y,)

if __name__ == '__main__':
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)