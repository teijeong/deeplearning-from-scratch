import numpy as np

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

if __name__ == '__main__':
    x = Variable(np.array(1.0))
    print(x.data)
