"""Multivariate Gaussian log-likelihood with numerically stable Cholesky decomposition.

Uses LAPACK dpotrf/dpotrs directly for efficiency on large covariance matrices.

Observable-space helpers
------------------------
``build_experimental_covariance``   — diagonal, equicorrelated, fully
                                      correlated, exponentially correlated,
                                      or absent experimental covariance
``build_total_covariance_obs``      — sum emulator (obs-space) + exp covariance
``project_exp_cov_to_pc_space``     — propagate obs-space exp covariance → PC space
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lapack


def mvn_loglike(
    residuals: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Log-likelihood of a multivariate Normal distribution.

    Computes::

        log p(r | 0, C) = -0.5 [ r^T C^{-1} r + log|C| + n log(2π) ]

    Uses Cholesky factorisation (LAPACK dpotrf) for numerical stability.
    Returns ``-inf`` if the covariance matrix is not positive-definite.

    Parameters
    ----------
    residuals : array, shape (n_obs,)
        ``y_exp - y_model``.
    cov : array, shape (n_obs, n_obs)
        Total covariance matrix (emulator + experimental).

    Returns
    -------
    float
    """
    cov = np.asarray(cov, dtype=float)
    L, info = lapack.dpotrf(cov, lower=False)
    if info != 0:
        return -np.inf

    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    x, info = lapack.dpotrs(L, residuals.reshape(-1, 1), lower=False)
    if info != 0:
        return -np.inf

    chi2 = float(residuals @ x.ravel())
    n = len(residuals)
    return -0.5 * (chi2 + log_det + n * np.log(2.0 * np.pi))


def build_covariance(
    emulator_variance: np.ndarray,
    experimental_variance: np.ndarray,
    model_sys_fraction: float = 0.0,
    model_prediction: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the total diagonal covariance matrix.

    ``C = diag(emulator_var + exp_var [+ (f * y_model)^2])``

    Parameters
    ----------
    emulator_variance : array, shape (n_obs,)
        GP predictive variance at the proposed parameter point.
    experimental_variance : array, shape (n_obs,)
        Experimental (statistical + systematic) variance.
    model_sys_fraction : float
        If > 0, adds a fractional model systematic: ``(f * |y_model|)^2``.
    model_prediction : array, shape (n_obs,), optional
        Required when ``model_sys_fraction > 0``.

    Returns
    -------
    array, shape (n_obs, n_obs)
        Diagonal covariance matrix.
    """
    diag = emulator_variance + experimental_variance
    if model_sys_fraction > 0.0 and model_prediction is not None:
        diag = diag + (model_sys_fraction * model_prediction) ** 2
    return np.diag(diag)


# ---------------------------------------------------------------------------
# Observable-space covariance helpers
# ---------------------------------------------------------------------------

def build_experimental_covariance(
    exp_variance: np.ndarray,
    mode: str = "diag",
    corr_rho: float = 0.0,
    corr_length: float | None = None,
    jitter: float = 1e-12,
) -> np.ndarray:
    """Build the experimental covariance matrix in **observable space**.

    Parameters
    ----------
    exp_variance : array, shape (n_obs,)
        Per-observable experimental variance  ``σ_i²``.
    mode : {"none", "diag", "equicorr", "fullycorr", "exp"}
        ``"none"``    — no experimental covariance term at all.
        ``"diag"``     — diagonal (independent observables).
        ``"equicorr"`` — equicorrelated: ``C_ij = σ_i σ_j [(1-ρ)δ_ij + ρ]``.
        ``"fullycorr"`` — fully correlated systematic: ``C = σ σ^T``.
        ``"exp"``      — exponential correlation decay with index distance.
    corr_rho : float
        Off-diagonal correlation coefficient (used only for ``mode="equicorr"``).
        Must satisfy ``-1/(n_obs-1) < ρ < 1``.
    corr_length : float, optional
        Correlation length for ``mode="exp"``. The aliases ``exp_short`` and
        ``exp_long`` use built-in defaults when this is omitted.
    jitter : float
        Small diagonal regularisation added for numerical stability.

    Returns
    -------
    array, shape (n_obs, n_obs)
    """
    mode = mode.strip().lower()
    variances = np.asarray(exp_variance, dtype=float).flatten()
    if mode in {"none", "zero", "nocov", "noexp", "absent"}:
        return np.zeros((variances.shape[0], variances.shape[0]), dtype=float)
    if np.any(variances <= 0):
        raise ValueError("Experimental variances must be strictly positive.")

    if mode == "diag":
        cov = np.diag(variances)
    elif mode in {"equicorr", "constant_corr", "corr"}:
        n_obs = variances.shape[0]
        lower_bound = -1.0 / max(1, n_obs - 1)
        if not (lower_bound < corr_rho < 1.0):
            raise ValueError(
                f"corr_rho={corr_rho} out of valid equicorrelation range "
                f"({lower_bound:.6f}, 1)."
            )
        sigma = np.sqrt(variances)
        corr = np.full((n_obs, n_obs), corr_rho, dtype=float)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(sigma, sigma) * corr
    elif mode in {"fullycorr", "fullcorr", "totalcorr", "allcorr"}:
        sigma = np.sqrt(variances)
        cov = np.outer(sigma, sigma)
    elif mode in {
        "exp",
        "exponential",
        "exp_short",
        "short_exp",
        "short_exponential",
        "exp_long",
        "long_exp",
        "long_exponential",
    }:
        if corr_length is None:
            if mode in {"exp_short", "short_exp", "short_exponential"}:
                corr_length = 3.0
            elif mode in {"exp_long", "long_exp", "long_exponential"}:
                corr_length = 15.0
            else:
                raise ValueError("corr_length must be provided for mode='exp'.")
        if corr_length < 0:
            raise ValueError("corr_length must be strictly positive.")
        if corr_length == 0:
            # Limit corr_length -> 0 gives diagonal covariance (no off-diagonal correlation).
            cov = np.diag(variances)
            return cov + np.eye(cov.shape[0], dtype=float) * jitter
        sigma = np.sqrt(variances)
        idx = np.arange(variances.shape[0], dtype=float)
        dist = np.abs(idx[:, None] - idx[None, :])
        corr = np.exp(-dist / float(corr_length))
        cov = np.outer(sigma, sigma) * corr
    else:
        raise ValueError(
            f"Unknown covariance mode {mode!r}. Use 'none', 'diag', 'equicorr', 'fullycorr', or 'exp'."
        )

    return cov + np.eye(cov.shape[0], dtype=float) * jitter


def build_total_covariance_obs(
    emulator_cov_obs: np.ndarray,
    exp_cov_obs: np.ndarray,
) -> np.ndarray:
    """Assemble the total covariance matrix in observable space.

    ``C_total = C_emulator + C_exp``

    Both matrices must have the same shape ``(n_obs, n_obs)``.
    ``C_emulator`` is typically produced by propagating per-PC emulator
    variance back to observable space via the PCA inverse-transform matrix
    ``A``:  ``C_emulator = A^T diag(var_pc) A``.

    Parameters
    ----------
    emulator_cov_obs : array, shape (n_obs, n_obs)
    exp_cov_obs : array, shape (n_obs, n_obs)

    Returns
    -------
    array, shape (n_obs, n_obs)
    """
    return emulator_cov_obs + exp_cov_obs


def project_exp_cov_to_pc_space(
    exp_cov_obs: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """Project an observable-space experimental covariance into PC space.

    For the SVD-based PCA used in the Universality pipeline the forward
    transform is  ``z = (y - μ) @ A⁻¹`` and the inverse is ``y = z @ A + μ``
    where  ``A = inverse_tf_matrix``  (shape n_pc × n_obs).  The covariance
    in PC space is therefore:

        ``C_pc = A  C_obs  A^T``   (note: A maps obs → PC via A^{-T})

    When ``A`` is the inverse-transform matrix (PC→obs), the forward-transform
    matrix is the pseudo-inverse of A, so projecting obs-space covariance into
    PC space follows:

        ``C_pc = (A A^T)^{-1} A  C_obs  A^T (A A^T)^{-1}``

    For orthonormal PCA (unit-variance whitened PCs) the rows of A are
    orthogonal and ``A A^T = I``, which simplifies to:

        ``C_pc = A  C_obs  A^T``

    Parameters
    ----------
    exp_cov_obs : array, shape (n_obs, n_obs)
        Experimental covariance in observable space.
    transform_matrix : array, shape (n_pc, n_obs)
        The *inverse* transform matrix (PC → observable), i.e.
        ``inverse_tf_matrix`` from Bayesian_pipeline or the rows of
        ``PCAResult.transform_matrix``.

    Returns
    -------
    array, shape (n_pc, n_pc)
        Experimental covariance projected into PC space.
    """
    A = np.asarray(transform_matrix)  # (n_pc, n_obs)
    # Normalise rows so that A A^T ≈ I  (whitened basis)
    norms_sq = np.einsum("ij,ij->i", A, A)          # (n_pc,)
    # C_pc = A C_obs A^T / (||row_i|| ||row_j||)  is only exact for diagonal
    # C_obs.  For a general C_obs the exact formula is:
    #   C_pc = (A/norms) C_obs (A/norms)^T
    inv_norms = 1.0 / np.sqrt(norms_sq)
    A_norm = A * inv_norms[:, np.newaxis]
    return A_norm @ exp_cov_obs @ A_norm.T
