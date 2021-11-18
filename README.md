# batchjax



## Example

```python
import legogp as lego
import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax

# generate data
P = 10

x = np.linspace(0, 1, 100)
y = np.sin(x*10)

X = x[:, None]
Y = y[:, None]

print(X.shape, Y.shape)

YP = np.hstack([Y*p for p in range(P)])
```

Create a list of models that we want to batch over

```python
m_list = objax.ModuleList([
    lego.models.GP(X, Y+p)
    for p in range(P)
])

def get_objective(y_p, m, m2):
    return jnp.sum(y_p), m.get_objective(), m2.get_objective()
```

manual loop

```python
%%timeit -r 1 -n 1
# Manual 
val_list = []
for p in range(P):
    val_list.append(
        get_objective(YP[:, p], m_list[p], m_list[p])
    )
val = np.vstack(val_list)
print(val)
```

automatic batching

```python
%%timeit -r 1 -n 1
# vmap approach
val = batchjax.batch_fn(get_objective, [YP, m_list, m_list], axes=[1, 0, 0])
print(val)
```

