from typing import Any, Iterator, List, MutableSet, Optional, Sequence, Union
import weakref

import numpy as np

from dezero import core
from dezero import functions as F

class Layer:
    def __init__(self) -> None:
        self._params: MutableSet[core.Parameter] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (core.Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: core.Variable
        ) -> Union[List[core.Variable], core.Variable]:
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs: Sequence[core.Variable]
        ) -> Union[List[core.Variable], core.Variable]:
        raise NotImplementedError()

    def params(self) -> Iterator[core.Parameter]:
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self) -> None:
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(
        self,
        out_size: int,
        nobias=False,
        dtype=np.float32,
        in_size: Optional[int] = None) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = core.Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = core.Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self) -> None:
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x: core.Variable) -> core.Variable:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = F.linear(x, self.W, self.b)
        return y