from enum import Enum
from .looper import loop_fn
from .batcher import batch_over_batched_list, batch_over_objax_list

class BatchType(Enum):
    LOOP = 0
    BATCHED = 1
    OBJAX=2

def batch_or_loop(fn, inputs: list, axes: list, dim: int, out_dim: int, batch_type: BatchType):
    if batch_type == BatchType.LOOP:
        return loop_fn(fn, inputs, axes, dim, out_dim)

    elif batch_type == BatchType.BATCHED:
        return batch_over_batched_list(fn, inputs, axes, out_dim)

    elif batch_type == BatchType.OBJAX:
        return batch_over_objax_list(fn, inputs, axes, out_dim)

    else:
        raise RuntimeError(f'Batch Type {batch_type} is not available!')
