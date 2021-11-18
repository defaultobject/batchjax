import objax
import jax
import jax.numpy as np
import typing
from typing import Callable

from . import batch_fn

def index_axis(obj, axis):
    if axis == 0:
        return obj[0]
    elif axis == 1:
        return obj[:, 1]
    else:
        raise NotImplementedError()

def get_input_shape(obj, axis):
    if axis == 0:
        return len(obj)
    elif axis == 1:
        return len(obj[axis])
    else:
        raise NotImplementedError()

def loop_fn(fn, inputs, axes):
    #TODO: assert that num_iter is the same for all inputs

    num_inputs = len(inputs)
    num_iter = get_input_shape(inputs[0], axes[0])

    val_list = []
    for i in range(num_iter):
        inputs_i = [
            index_axis(inputs[n], axes[n])
            for n in range(num_inputs)
        ]
        val_list.append(
            fn(*inputs_i)
        )

    val = np.vstack(val_list)

    return val

def batch_or_loop(fn, inputs: list, axes: list, batch_flag: bool):
    if batch_flag:
        return batch_fn(fn, inputs, axes)
    else:
        return loop_fn(fn, inputs, axes)

