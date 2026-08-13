import numpy as np
import pytest

from hic_bayes.pca import fit_pca


def test_pca_basic(toy_dataset):
    _, Y, _, n_obs = toy_dataset
    result = fit_pca(Y, n_components=3)
    assert result.n_components == 3
    assert result.variance_explained.shape == (3,)
    assert np.all(result.variance_explained >= 0)
    assert result.cumulative_variance[-1] <= 1.0 + 1e-10


def test_pca_full_roundtrip(toy_dataset):
    _, Y, _, n_obs = toy_dataset
    result = fit_pca(Y, n_components=n_obs)
    Z = result.transform(Y)
    Y_rec = result.inverse_transform(Z)
    np.testing.assert_allclose(Y_rec, Y, atol=1e-6)


def test_pca_transform_shape(toy_dataset):
    _, Y, _, _ = toy_dataset
    n_comp = 5
    result = fit_pca(Y, n_components=n_comp)
    Z = result.transform(Y)
    assert Z.shape == (len(Y), n_comp)


def test_pca_unit_variance(toy_dataset):
    """PC scores on the training set should have unit variance per component."""
    _, Y, _, _ = toy_dataset
    result = fit_pca(Y, n_components=5)
    Z = result.transform(Y)
    # Allow some numerical tolerance
    np.testing.assert_allclose(Z.std(axis=0), np.ones(5), atol=0.1)


def test_pca_cumulative_monotonic(toy_dataset):
    _, Y, _, _ = toy_dataset
    result = fit_pca(Y, n_components=10)
    diff = np.diff(result.cumulative_variance)
    assert np.all(diff >= -1e-12)
