import objax
import jax
import jax.numpy as np
import typing
from typing import Callable

def index_axis(obj, i, a):
    """
    Return obj indexed at i on axis a
    This is done manually to support both lists and numpy arrays
    """
    if a ==  None:
        return obj
    elif a == 0:
        return obj[i]
    elif a == 1:
        return obj[:, i]
    else:
        raise NotImplementedError()


def loop_fn(fn, inputs, axes, dim: int, out_dim:int):
    #TODO: assert that dim is the same for all inputs
    num_inputs = len(inputs)

    val_list = [[] for d in range(out_dim)]

    for i in range(dim):
        inputs_i = [
            index_axis(inputs[n], i, axes[n])
            for n in range(num_inputs)
        ]

        fn_out = fn(*inputs_i)

        if out_dim > 1:
            val_list = [val_list[d] + [fn_out[d]] for d in range(out_dim)]
        else:
            val_list = [val_list[0] + [fn_out]]

    if out_dim > 1:
        val_list = [np.array(val_list[d]) for d in range(out_dim)]
    else:
        val_list = np.array(val_list[0])

    return val_list


