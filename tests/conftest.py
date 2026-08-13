import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_dataset():
    """Synthetic 5-param, 100-point, 20-observable dataset."""
    rng = np.random.default_rng(42)
    n_pts, n_params, n_obs = 100, 5, 20
    X = rng.uniform(0.0, 1.0, size=(n_pts, n_params))
    A = rng.standard_normal((n_params, n_obs))
    Y = X @ A + 0.01 * rng.standard_normal((n_pts, n_obs))
    return X, Y, n_params, n_obs


@pytest.fixture
def param_ranges(toy_dataset):
    _, _, n_params, _ = toy_dataset
    return np.column_stack([np.zeros(n_params), np.ones(n_params)])
