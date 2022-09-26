""" Looping using native python loop. """
from typing import Callable
import jax.numpy as np


def index_axis(obj, i, axis):
    """
    Return obj indexed at i on axis

    Args:
        obj: list/array of objects
        i: index
        axis: which axes to index across

    Note: This is done manually to support both lists and numpy arrays
    """
    if axis is None:
        return obj

    if axis == 0:
        return obj[i]

    if axis == 1:
        return obj[:, i]

    if axis == 2:
        return obj[:, :, i]

    raise NotImplementedError(f'Indexing for axis {axis} at {i} is not supported')


def loop_fn(fn_to_batch: Callable, inputs, axes, dim: int, out_dim: int):
    """
    Batches fn_to_batch by looping through the required input axes.

    TODO: assert that dim is the same for all inputs
    """
    num_inputs = len(inputs)

    val_list = [[] for d in range(out_dim)]

    for i in range(dim):
        inputs_i = [index_axis(inputs[n], i, axes[n]) for n in range(num_inputs)]

        fn_out = fn_to_batch(*inputs_i)

        if out_dim > 1:
            val_list = [val_list[d] + [fn_out[d]] for d in range(out_dim)]
        else:
            val_list = [val_list[0] + [fn_out]]

    if out_dim > 1:
        val_list = [np.array(val_list[d]) for d in range(out_dim)]
    else:
        val_list = np.array(val_list[0])

    return val_list
