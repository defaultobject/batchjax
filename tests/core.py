""" Highlevel tests to ensure that results are consistent across all modes """

import pytest

import numpy as onp
import jax.numpy as np
import objax

import batchjax

from .common_fixtures import regression_1d_data, neural_network_list

class _NNList(objax.Module):
    """ Wrapper around a NN to add suport for multiple Neural networks. """
    def __init__(self, m_list: list, batch_type):
        self.P = len(m_list)

        if batch_type == batchjax.BatchType.BATCHED:
            self.m_list = batchjax.Batched(m_list)
        else:
            self.m_list = objax.ModuleList(m_list)

        self.batch_type = batch_type

    def objective(self):
        # Use batchjax to batch across each neural network
        obj_arr = batchjax.batch_or_loop(
            lambda x: x.objective(),
            inputs = [self.m_list],
            axes=[0],
            dim = self.P,
            out_dim = 1,
            batch_type = self.batch_type
        )

        return np.sum(obj_arr)

@pytest.mark.parametrize('N', [100])
@pytest.mark.parametrize('num_models', [10])
def test__batch_or_loop__loop_vs_explicit(neural_network_list):
    # Setup
    m_loop = _NNList(neural_network_list, batchjax.BatchType.LOOP)

    # Run

    truth_val = onp.sum([nn.objective() for nn in neural_network_list])
    loop_val = m_loop.objective()

    # Assert
    onp.testing.assert_allclose(truth_val, loop_val)

@pytest.mark.parametrize('N', [100])
@pytest.mark.parametrize('num_models', [10])
def test__batch_or_loop__loop_vs_objax(neural_network_list):
    # Setup

    m_loop = _NNList(neural_network_list, batchjax.BatchType.LOOP)
    m_objax = _NNList(neural_network_list, batchjax.BatchType.OBJAX)

    # Run

    loop_val = m_loop.objective()
    objax_val = m_objax.objective()

    # Assert
    onp.testing.assert_allclose(loop_val, objax_val)

@pytest.mark.parametrize('N', [100])
@pytest.mark.parametrize('num_models', [10])
def test__batch_or_loop__loop_vs_batched(neural_network_list):
    # Setup

    m_loop = _NNList(neural_network_list, batchjax.BatchType.LOOP)
    m_batched = _NNList(neural_network_list, batchjax.BatchType.BATCHED)

    # Run

    loop_val = m_loop.objective()
    batched_val = m_batched.objective()

    # Assert
    onp.testing.assert_allclose(loop_val, batched_val)

