from typing import Callable, Dict, Optional, List

import numpy as np

import dezero as dz
from dezero import core

HookFn = Callable[[List[core.Parameter]], None]

class Optimizer:
    def __init__(self) -> None:
        self.target: dz.Layer = None
        self.hooks: List[HookFn] = []

    def setup(self, target: dz.Layer) -> "Optimizer":
        self.target = target
        return self

    def update(self) -> None:
        # collect parameters with gradients to `params`
        params = [p  for p in self.target.params() if p.grad is not None]

        # preprocess (optional)
        for f in self.hooks:
            f(params)

        # update parameters
        for param in params:
            self.update_one(param)

    def update_one(self, param: core.Parameter) -> None:
        raise NotImplementedError()

    def add_hook(self, f: HookFn) -> None:
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: core.Parameter) -> None:
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs: Dict[int, np.ndarray] = {}

    def update_one(self, param: core.Parameter) -> None:
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v