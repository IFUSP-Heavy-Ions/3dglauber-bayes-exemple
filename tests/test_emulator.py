import numpy as np
import pytest

from hic_bayes.emulator import Emulator
from hic_bayes.pca import fit_pca


def test_emulator_fit_and_predict(toy_dataset, param_ranges):
    X, Y, _, _ = toy_dataset
    pca = fit_pca(Y, n_components=3)
    Z = pca.transform(Y)
    em = Emulator(n_restarts=0)
    em.fit(X, Z, param_ranges, n_jobs=1)

    mean, std = em.predict(X[:5], return_std=True)
    assert mean.shape == (5, 3)
    assert std.shape == (5, 3)
    assert np.all(std >= 0)


def test_emulator_no_std(toy_dataset, param_ranges):
    X, Y, _, _ = toy_dataset
    pca = fit_pca(Y, n_components=2)
    Z = pca.transform(Y)
    em = Emulator(n_restarts=0)
    em.fit(X, Z, param_ranges, n_jobs=1)

    mean, std = em.predict(X[:3], return_std=False)
    assert mean.shape == (3, 2)
    assert std is None


def test_emulator_interpolates_training(toy_dataset, param_ranges):
    """GP should predict training points near-exactly (very low noise kernel)."""
    X, Y, _, _ = toy_dataset
    pca = fit_pca(Y, n_components=3)
    Z = pca.transform(Y)
    em = Emulator(n_restarts=0)
    em.fit(X, Z, param_ranges, n_jobs=1)

    mean, _ = em.predict(X, return_std=True)
    # Training-set R² should be high for a well-fitted GP
    ss_res = np.sum((Z - mean) ** 2)
    ss_tot = np.sum((Z - Z.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.90, f"Training R² = {r2:.3f} — GP may not be fitting well"
