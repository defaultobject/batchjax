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

        #only need to do for the first object as all object should have the same var_collection

    return all_vars

def dict_to_int(d, num):
    return {k: num for k,i in d.items()}

def batch_fn(fn, inputs: list, axes: list):
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

    def _batched(*args):
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

        new_tensors = ml_map(
            batched_vars,
            is_ml_fn = lambda bv, i: [i for k, i in bv.items()],
            not_ml_fn = lambda x, i: x,
            bool_arr = input_module_flag,
        )
        
        # assign new tensors to modules

        ml_map(
            modules,
            is_ml_fn = lambda x, i: x.vars().assign(new_tensors[i]),
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
    
    res =  jax.vmap(
        _batched,
        in_axes=[*ref_vmap_inputs_axes, *in_axes_dict_list]
    )(*ref_vmap_inputs, *batched_inputs)
    
    return np.array(res).T
