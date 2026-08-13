import numpy as np
import pytest

from hic_bayes.likelihood import build_covariance, mvn_loglike


def test_standard_normal_loglike():
    """log p(0 | 0, I) == -n/2 * log(2π)."""
    n = 10
    ll = mvn_loglike(np.zeros(n), np.eye(n))
    expected = -0.5 * n * np.log(2 * np.pi)
    assert abs(ll - expected) < 1e-10


def test_scalar_normal():
    ll = mvn_loglike(np.array([0.0]), np.array([[1.0]]))
    expected = -0.5 * np.log(2 * np.pi)
    assert abs(ll - expected) < 1e-10


def test_singular_covariance():
    """A singular covariance matrix should return -inf, not raise."""
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    ll = mvn_loglike(np.array([0.5, 0.5]), cov)
    assert ll == -np.inf


def test_loglike_decreases_with_distance():
    """
    Moving the observation away from the mean should decrease the log-likelihood.
    """
    cov = np.eye(3)
    ll_center = mvn_loglike(np.zeros(3), cov)
    ll_offset = mvn_loglike(np.array([1.0, 1.0, 1.0]), cov)
    assert ll_center > ll_offset


def test_build_covariance_diagonal():
    em_var = np.array([0.10, 0.20])
    exp_var = np.array([0.30, 0.40])
    cov = build_covariance(em_var, exp_var)
    np.testing.assert_allclose(np.diag(cov), [0.40, 0.60])
    # Should be diagonal — off-diagonal zeros
    assert cov[0, 1] == 0.0


def test_build_covariance_with_model_systematic():
    em_var = np.zeros(3)
    exp_var = np.zeros(3)
    pred = np.array([1.0, 2.0, 3.0])
    f = 0.1
    cov = build_covariance(em_var, exp_var, model_sys_fraction=f, model_prediction=pred)
    np.testing.assert_allclose(np.diag(cov), (f * pred) ** 2)
