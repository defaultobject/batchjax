import objax
import jax
import jax.numpy as np
import typing
from typing import Callable, List

def vc_to_dict(vc):
    """ Convert an objax varcolection to python dict """
    all_vars = {}

    for key in vc.keys():
        all_vars[key] =  np.array(vc[key].value)

    return all_vars

def remove_prefix_from_dict_keys(d: dict, prefix: str):
    """ Assumes that all keys in d start with the same prefix """
    return {
        k[len(prefix):]: v 
        for k, v in d.items()
    }

def get_state_var_names(obj_list):
    sv_names = []
    for obj in obj_list:
        var_collection = obj.vars()
        for key in var_collection.keys():
            if type(var_collection[key]) != objax.TrainVar:
                sv_names.append(key)

    return sv_names

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

def bool_map(
        items: list, true_fn: Callable, false_fn: Callable, bool_arr: List[bool]
    ) -> list:
    """ 
        Iterates over items and applies either true_fn or false_fn depending 
            on wether bool_arr ir true or false respectiely.

        Both true_fn and false_fn take 2 arguments:
            item , index
    """
    return [
        true_fn(items[i], i) if bool_arr[i] else false_fn(items[i], i) 
        for i in range(len(items))
    ]

class Batched(objax.Module):
    def __init__(self, mod_list: list):
        # use list to hide from objax
        self.templ_m = [mod_list[0]]

        mod_list = objax.ModuleList(mod_list)

        # Collect batched versions of all variables across mod_list
        var_list = get_batched_vars(mod_list)

        # Set each variable as a trainable var so that objax can find them

        sv_names = get_state_var_names(mod_list)
        for k, v in var_list.items():
            if k in sv_names:
                setattr(self, k, objax.StateVar(v))
            else:
                setattr(self, k, objax.TrainVar(v))

def _batched_vmap_wrapper(fn, bool_arr, *args):

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
    modules = bool_map(
        modules,
        true_fn = lambda x, i: x,
        false_fn = lambda x, i: batched_vars[i],
        bool_arr = bool_arr
    )
    
    original_tensors = bool_map(
        modules,
        true_fn = lambda x, i: x.vars().tensors(),
        false_fn = lambda x, i: x,
        bool_arr = bool_arr,
    )

    # JAX does not ensure that dict will have same order after vmap
    # So we need re-order the batched varcollections to match that of the corresponding modules
    # See https://github.com/google/jax/issues/4085

    fix_order = lambda  d, m: {a: d[a] for a in m.vars().keys()}

    new_tensors = bool_map(
        batched_vars,
        true_fn = lambda bv, i: [i for k, i in fix_order(bv, modules[i]).items()],
        false_fn = lambda x, i: x,
        bool_arr = bool_arr,
    )
    
    # assign new tensors to modules

    bool_map(
        modules,
        true_fn = lambda x, i: x.vars().assign(
            list_index(
                new_tensors[i],
                get_objax_iter_index(x.vars())
            )
        ),
        false_fn = lambda x, i: None,
        bool_arr = bool_arr,
    )        
    
    val = fn(*modules)
    
    # assign old tensors back

    bool_map(
        modules,
        true_fn = lambda x, i: x.vars().assign(original_tensors[i]),
        false_fn = lambda x, i: None,
        bool_arr = bool_arr,
    )  

    return val

def _batched(fn, inputs, axes, out_dim, bool_arr, module_ref_fn, var_fn):

    N = len(inputs)

    
    # For each Batched obj we need to pass through the objax module that is being matched

    ref_vmap_inputs = bool_map(
        inputs,
        true_fn = lambda x, i: module_ref_fn(x),
        false_fn = lambda x, i: None,
        bool_arr = bool_arr
    )

    # Do not batch the reference objax.Modules
    ref_vmap_inputs_axes = [None for i in range(N)]


    batched_inputs = bool_map(
        inputs,
        true_fn = lambda x, i: var_fn(x),
        false_fn = lambda x, i: x,
        bool_arr = bool_arr
    )

    in_axes_dict_list = bool_map(
        batched_inputs,
        true_fn = lambda x, i: dict_to_int(x, axes[i]),
        false_fn = lambda x, i: axes[i],
        bool_arr = bool_arr
    )

    res =  jax.vmap(
        _batched_vmap_wrapper,
        in_axes=[None, None, *ref_vmap_inputs_axes, *in_axes_dict_list],
        out_axes=0
    )(fn, bool_arr, *ref_vmap_inputs, *batched_inputs)
    
    return res


def batch_over_batched_list(fn, inputs, axes: list, out_dim: int):
    # For each input we can have either Batched or a jax type
    # Identify which inputs are of type Batched

    input_batched_flag = [type(i) == Batched for i in inputs]

    return _batched(
        fn,
        inputs, 
        axes, 
        out_dim, 
        input_batched_flag, 
        lambda x: x.templ_m[0],
        lambda x: remove_prefix_from_dict_keys(vc_to_dict(x.vars()), '(Batched).')
    )


def batch_over_objax_list(fn, inputs: list, axes: list, out_dim: int):

    input_batched_flag = [type(i) == objax.ModuleList for i in inputs]

    return _batched(
        fn,
        inputs, 
        axes, 
        out_dim, 
        input_batched_flag, 
        lambda x: x[0],
        lambda x: get_batched_vars(x)
    )
