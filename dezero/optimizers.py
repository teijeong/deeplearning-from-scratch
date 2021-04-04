from typing import Callable, Optional, List

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