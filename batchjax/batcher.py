import objax
import jax
import jax.numpy as np
import typing
from typing import Callable

def get_batched_vars(obj_list):
    all_vars = {}

    #collect vars
    for obj in obj_list:
        var_collection = obj.vars()
        for key in var_collection.keys():
            if key not in all_vars:
                all_vars[key] = []

            all_vars[key].append(var_collection[key].value)

    #convert to jax array
    for obj in obj_list:
        var_collection = obj.vars()
        for key in var_collection.keys():
            all_vars[key] = np.array(all_vars[key])

    return all_vars

def dict_to_int(d, num):
    return {k: num for k,i in d.items()}

def get_objax_iter_index(vc):
    seen = set()
    idx = []
    for i, v in enumerate(vc.values()):
        if id(v) not in seen:
            seen.add(id(v))
            idx.append(i)
    return idx

def list_index(a, idx):
    new_a = list(map(a.__getitem__, idx))
    return new_a

def _batched(fn, input_module_flag, *args):

    ml_map = lambda items, is_ml_fn, not_ml_fn, bool_arr: [is_ml_fn(items[i], i) if bool_arr[i] else not_ml_fn(items[i], i) for i in range(len(items))]



    # The first half of args refer to modules
    num_args = len(args)
    num_m = int(num_args/2)
    
    # reference modules
    modules = [args[i] for i in range(num_m)]

    # batched variables
    batched_vars = [args[i] for i in range(num_m, num_args)]

    # modules is the array of referce variables which have not been batched
    #  if a module is not a ModuleList we need to replace with the corresonding tensor
    #  inside batched_vars
    modules = ml_map(
        modules,
        is_ml_fn = lambda x, i: x,
        not_ml_fn = lambda x, i: batched_vars[i],
        bool_arr = input_module_flag
    )
    
    original_tensors = ml_map(
        modules,
        is_ml_fn = lambda x, i: x.vars().tensors(),
        not_ml_fn = lambda x, i: x,
        bool_arr = input_module_flag,
    )

    # JAX does not ensure that dict will have same order after vmap
    # So we need re-order the batched varcollections to match that of the corresponding modules
    # See https://github.com/google/jax/issues/4085

    fix_order = lambda  d, m: {a: d[a] for a in m.vars().keys()}

    new_tensors = ml_map(
        batched_vars,
        is_ml_fn = lambda bv, i: [i for k, i in fix_order(bv, modules[i]).items()],
        not_ml_fn = lambda x, i: x,
        bool_arr = input_module_flag,
    )
    
    # assign new tensors to modules

    ml_map(
        modules,
        is_ml_fn = lambda x, i: x.vars().assign(
            list_index(
                new_tensors[i],
                get_objax_iter_index(x.vars())
            )
        ),
        not_ml_fn = lambda x, i: None,
        bool_arr = input_module_flag,
    )        
    
    val = fn(*modules)
    
    # assign old tensors back

    ml_map(
        modules,
        is_ml_fn = lambda x, i: x.vars().assign(original_tensors[i]),
        not_ml_fn = lambda x, i: None,
        bool_arr = input_module_flag,
    )  

    return val

def batch_fn(fn, inputs: list, axes: list, out_dim: int):
    N = len(inputs)

    # helper function to apply different functions to module list or now
    ml_map = lambda items, is_ml_fn, not_ml_fn, bool_arr: [is_ml_fn(items[i], i) if bool_arr[i] else not_ml_fn(items[i], i) for i in range(len(items))]

    # Figure out which inputs are modulelists
    input_module_flag = [type(i) == objax.ModuleList for i in inputs]

    # Construct reference inputs to vmap
    #  if an input is a ModuleList we select the first one as it should make no difference
    ref_vmap_inputs = ml_map(
        inputs,
        is_ml_fn = lambda x, i: x[0],
        not_ml_fn = lambda x, i: None,
        bool_arr = input_module_flag
    )

    ref_vmap_inputs_axes = [None for i in range(N)]


    # To vmap across module lists we compute a stacked tree of variables and then
    #  vmap across this.
    
    # Compute the stacked variable tree for each module list in inputs
    batched_inputs = ml_map(
        inputs,
        is_ml_fn = lambda x, i: get_batched_vars(x),
        not_ml_fn = lambda x, i: x,
        bool_arr = input_module_flag
    )

    in_axes_dict_list = ml_map(
        batched_inputs,
        is_ml_fn = lambda x, i: dict_to_int(x, axes[i]),
        not_ml_fn = lambda x, i: axes[i],
        bool_arr = input_module_flag
    )

    res =  jax.vmap(
        _batched,
        in_axes=[None, None, *ref_vmap_inputs_axes, *in_axes_dict_list],
        out_axes=0
    )(fn, input_module_flag, *ref_vmap_inputs, *batched_inputs)
    
    return res
