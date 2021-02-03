from typing import List, Sequence, Tuple, Union

import numpy as np

from step09 import as_array

class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f'Unsupported type: {type(data)}')

        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None
        self.generation = 0

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self) -> None:
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
            gys = [output.grad for output in f.outputs]
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

    def cleargrad(self) -> None:
        self.grad = None


class Function:
    def __call__(self, *inputs: Sequence[Variable]
            ) -> Union[List[Variable], Variable]:
        """Calls fucntion and saves called data."""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max(x.generation for x in inputs)
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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

if __name__ == '__main__':
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)