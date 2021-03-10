import os
import subprocess
import tempfile

import dezero as dz

def _dot_var(v: dz.Variable, verbose=False) -> str:
    dot_var = '{} [label="{}", color=orange, style=filled]'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += '{} {}'.format(v.shape, v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f: dz.Function) -> str:
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]'
    defs = [dot_func.format(id(f), f.__class__.__name__)]

    dot_edge = '{} -> {}'
    for x in f.inputs:
        defs.append(dot_edge.format(id(x), id(f)))
    for y in f.outputs:
        defs.append(dot_edge.format(id(f), id(y())))
    return '\n'.join(defs)


def get_dot_graph(output: dz.Variable, verbose=True) -> str:
    defs = ['digraph g {']
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    defs.append(_dot_var(output, verbose))

    while funcs:
        func = funcs.pop()
        defs.append(_dot_func(func))
        for x in func.inputs:
            defs.append(_dot_var(x, verbose))
            if x.creator is not None:
                add_func(x.creator)
    defs.append('}')
    return '\n'.join(defs)


def plot_dot_graph(output: dz.Variable, verbose=True,
                   to_file='graph.png') -> None:
    dot_graph = get_dot_graph(output, verbose)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dot') as f:
        f.write(dot_graph)
        extension = os.path.splittext(to_file)[1][1:]
        cmd = f'dot {f.name} -T {extension} -o {to_file}'
        subprocess.run(cmd, shell=True)


# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================
def sum_to(x: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    """Sums elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape: 

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(
    gy: dz.Variable,
    x_shape: Sequence[int],
    axis: Optional[Union[int, Sequence[int]]],
    keepdims: bool) -> dz.Variable:
    """Reshapes gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy