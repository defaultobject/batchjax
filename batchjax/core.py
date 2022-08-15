""" Helper functions for using batchjax."""
from enum import Enum
from .looper import loop_fn
from .batcher import batch_over_batched_list, batch_over_objax_list


class BatchType(Enum):
    """Enum to specify which type of batching should be used."""

    LOOP = 0
    BATCHED = 1
    OBJAX = 2


def batch_or_loop(
    fn_to_batch, inputs: list, axes: list, dim: int, out_dim: int, batch_type: BatchType
):
    """
    Main input function to batchjax for looping or batching fn_to_batch.

    Args:
        fn_to_batch: callable function to batch/loop over
        inputs: inputs to be passed to fn_to_batch
        axes: corresponding axis for each input to batch/loop over
        dim: the dimension of the axis that is being batched over
        out_dim: the number of arguments to returned by fn_to_batch
        batch_type: enum to specify whether native python loops or batching should be used.
    """
    if len(axes) != len(inputs):
        raise RuntimeError(
            f'There are {len(inputs)} inputs passed but only {len(axes)} axes passed!'
        )

    if batch_type == BatchType.LOOP:
        return loop_fn(fn_to_batch, inputs, axes, dim, out_dim)

    if batch_type == BatchType.BATCHED:
        return batch_over_batched_list(fn_to_batch, inputs, axes, out_dim)

    if batch_type == BatchType.OBJAX:
        return batch_over_objax_list(fn_to_batch, inputs, axes, out_dim)

    raise RuntimeError(f"Batch Type {batch_type} is not available!")
