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
    def __call__(self, *inputs: Sequence[Variable]
            ) -> Union[List[Variable], Variable]:
        """Calls fucntion and saves called data."""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
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

def add(x0: Variable, x1: Variable) -> Tuple[Variable]:
    return Add()(x0, x1)

if __name__ == '__main__':
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)