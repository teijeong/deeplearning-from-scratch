if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero as dz

x = dz.Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(x)
print(x.grad)
