import numpy as np
from step01 import Variable

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

if __name__ == '__main__':
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
