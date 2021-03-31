import contextlib
from typing import Any, List, Optional, Sequence, Tuple, Union
import weakref

import numpy as np

import dezero as dz

def as_array(x: Union[float, np.ndarray]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Config:
    enable_backprop = True


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


class Variable:
    # __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f'Unsupported type: {type(data)}')

        self.data = data
        self.name = name
        self.grad: Optional[Variable] = None
        self.creator: Optional[Function] = None
        self.generation = 0

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False) -> None:
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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

            with using_config('enable_backprop', create_graph):
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

    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Variable":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dz.functions.reshape(self, shape)

    def transpose(self) -> "Variable":
        return dz.functions.transpose(self)

    def sum(
            self,
            axis: Optional[Union[int, Sequence[int]]],
            keepdims: bool = False) -> "Variable":
        return dz.functions.sum(self, axis, keepdims)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> "Variable":
        return dz.functions.transpose(self)
    
    
    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'


def as_variable(obj: Union[np.ndarray, Variable]) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs: Sequence[Variable]
            ) -> Union[List[Variable], Variable]:
        """Calls fucntion and saves called data."""
        inputs = [as_variable(x) for x in inputs]

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

InputType = Union[Variable, np.ndarray, float, int]


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dz.functions.sum_to(gx0, self.x0_shape)
            gx1 = dz.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        return gy, -gy


def sub(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)


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
        x0, x1 = self.inputs
        return gy * x1, gy * x0


def mul(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


def neg(x: Variable) -> Variable:
    return Neg()(x)


class Pow(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: InputType) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c: Union[float, int]):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** self.c
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: Variable, c: Union[float, int]) -> Variable:
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__neg__ = neg
    Variable.__pow__ = pow


class Parameter(Variable):
    pass