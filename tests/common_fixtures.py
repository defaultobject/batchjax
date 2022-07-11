""" Helper fixtures """

import pytest
import numpy as onp
import jax.numpy as np
import objax
from objax.zoo.dnnet import DNNet
from objax.functional import tanh
from objax.functional.loss import mean_squared_error

class _NN(objax.Module):
    """ Simple fully connected Neural Network model wrapper """
    def __init__(self, X, Y, layer_size):
        self.model = DNNet(layer_sizes=layer_size, activation=tanh)
        self.X = objax.StateVar(np.array(X))
        self.Y = objax.StateVar(np.array(Y))

    def objective(self):
        return mean_squared_error(
            self.model(self.X.value),
            self.Y.value,
            keep_axis=None
        )

    def predict(self, XS):
        return self.model(XS)


@pytest.fixture
def regression_1d_data(N):
    """ Generates a noisy sin curve with N observations. """

    onp.random.seed(0)

    x = onp.linspace(0, 1, N)
    X = x[:, None]

    # Construct output with random input shift and additive Gaussian noise
    y = onp.sin((x+onp.random.randn(1))*10) + 0.01*onp.random.randn(N)
    Y = 0.8*y[:, None]

    return X, Y

@pytest.fixture
def neural_network_list(regression_1d_data, num_models):
    """ Generate a list of neural networks of size [1, 128, 1] """
    X, Y = regression_1d_data

    data = [
        [X, Y] for p in range(num_models)
    ]

    # Construct all independent neural networks
    model_list = [
        _NN(data[p][0], data[p][1], [1, 128, 1]) for p in range(num_models)
    ]   
    
    return model_list
