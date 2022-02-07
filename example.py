import sys
import jax 
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import make_jaxpr
import numpy as onp
import objax
from objax.zoo.dnnet import DNNet
from objax.functional import tanh
from objax.functional.loss import mean_squared_error

import batchjax

from timeit import default_timer as timer

import matplotlib.pyplot as plt

class NN(objax.Module):
    def __init__(self, X, Y):
        self.model = DNNet(layer_sizes=[1, 128, 1], activation=tanh)
        self.X = objax.StateVar(X)
        self.Y = objax.StateVar(Y)

    def objective(self):
        return mean_squared_error(
            self.model(self.X.value),
            self.Y.value,
            keep_axis=None
        )

    def predict(self, XS):
        return self.model(XS)

class NNList(objax.Module):
    def __init__(self, m_list: list, batch_type):
        self.P = len(m_list)

        if batch_type == batchjax.BatchType.BATCHED:
            self.m_list = batchjax.Batched(m_list)
        else:
            self.m_list = objax.ModuleList(m_list)

        self.batch_type = batch_type

    def objective(self):
        obj_arr = batchjax.batch_or_loop(
            lambda x: x.objective(),
            inputs = [self.m_list],
            axes=[0],
            dim = self.P,
            out_dim = 1,
            batch_type = self.batch_type
        )

        return np.sum(obj_arr)

# Construct Datasets

def get_data(N, seed):
    onp.random.seed(seed)
    x = onp.linspace(0, 1, N)
    X = x[:, None]

    # Construct output with random input shift and additive Gaussian noise
    y = onp.sin((x+onp.random.randn(1))*10) + 0.01*onp.random.randn(N)
    Y = 0.8*y[:, None]

    xs = onp.linspace(-1, 2, N)
    XS = xs[:, None]

    return X, X, Y

# Construct models

P = 50

data = [
    get_data(200, p) for p in range(P)
]

if False:
    for p in range(P):
        X_p, Y_p = data[p][1], data[p][2]
        plt.plot(X_p, Y_p)

    plt.show()


# Construct all models

model_list = [
    NN(data[p][1], data[p][2]) for p in range(P)
]   

start = timer()


if sys.argv[-1] == '--loop':
    m = NNList(model_list, batchjax.BatchType.LOOP)

elif sys.argv[-1] == '--objax':
    m = NNList(model_list, batchjax.BatchType.OBJAX)

elif sys.argv[-1] == '--batched':
    m = NNList(model_list, batchjax.BatchType.BATCHED)
else:
    raise RuntimeError()

# Train

onp.random.seed(0)
opt = objax.optimizer.Adam(m.vars())
lr = 1e-3
epochs = 500
gv = objax.GradValues(m.objective, m.vars())

breakpoint()


@objax.Function.with_vars(m.vars() + gv.vars() + opt.vars())
def train_op():
    g, v = gv()  # returns gradients, loss
    opt(lr, g)
    return v


train_op = objax.Jit(train_op)  # Compile train_op to make it run faster.

loss_arr = []
for i in range(epochs):
    v = train_op()
    loss_arr.append(v)

end = timer()

print('Time taken: ', end - start)
print('Final loss: ', loss_arr[-1])
