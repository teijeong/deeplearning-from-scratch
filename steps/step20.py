import contextlib
from typing import Any, List, Sequence, Tuple, Union
import weakref

import numpy as np

from step09 import as_array


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f'Unsupported type: {type(data)}')

        self.data = data
        self.name = name
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None
        self.generation = 0

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f: 'Function'):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self) -> None:
        self.grad = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'


class Function:
    def __call__(self, *inputs: Sequence[Variable]
            ) -> Union[List[Variable], Variable]:
        """Calls fucntion and saves called data."""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max(x.generation for x in inputs)
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs: Sequence[np.ndarray]) -> Tuple[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: Sequence[np.ndarray]) -> Tuple[np.ndarray]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> Tuple[np.ndarray]:
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0: Variable, x1: Variable) -> Tuple[Variable]:
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** 2
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable) -> Variable:
    return Square()(x)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: Variable, x1: Variable) -> Variable:
    return Mul()(x0, x1)


Variable.__mul__ = mul
Variable.__add__ = add


@contextlib.contextmanager
def using_config(name: str, value: Any):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(len(x))

    x = Variable(np.array([1, 2, 3]))
    print(x)

    x = Variable(None)
    print(x)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(x)

    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = a * b + c
    y.backward()

    print(y)
    print(a.grad)
    print(b.grad)

