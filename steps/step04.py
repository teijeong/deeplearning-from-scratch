import numpy as np

from step01 import Variable
from step02 import Function
from step02 import Square
from step03 import Exp

def numerical_diff(f: Function, x: Variable, eps=1e-4) -> np.ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def fn(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

if __name__ == '__main__':
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    x = Variable(np.array(0.5))
    dy = numerical_diff(fn, x)
    print(dy)