"""Single-fidelity GP emulator wrapping scikit-learn GaussianProcessRegressor.

One independent GP is trained per PC score dimension, allowing full
parallelism and individual kernel optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as krnl


@dataclass
class Emulator:
    """Collection of independent GP emulators, one per output dimension.

    Parameters
    ----------
    n_restarts : int
        Number of random restarts for kernel hyperparameter optimisation.

    Examples
    --------
    >>> em = Emulator(n_restarts=4)
    >>> em.fit(X_train, Z_train, param_ranges)
    >>> mean, std = em.predict(X_new, return_std=True)
    """

    n_restarts: int = 4
    gps: list = field(default_factory=list, repr=False)

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        param_ranges: np.ndarray,
        n_jobs: int = -1,
    ) -> Emulator:
        """Fit one GP per output column of Z.

        Parameters
        ----------
        X : array, shape (n_samples, n_params)
            Design points.
        Z : array, shape (n_samples, n_components)
            PC scores (training targets).
        param_ranges : array, shape (n_params, 2)
            ``[[min, max], ...]`` used to initialise RBF length-scale bounds.
        n_jobs : int
            Parallelism for training (-1 = all CPU cores).

        Returns
        -------
        self
        """
        ptp = param_ranges[:, 1] - param_ranges[:, 0]

        def _fit_one(z_col: np.ndarray) -> GaussianProcessRegressor:
            kernel = 1.0 * krnl.RBF(
                length_scale=ptp,
                length_scale_bounds=np.outer(ptp, (0.4, 100.0)),
            ) + krnl.WhiteKernel(
                noise_level=0.1,
                noise_level_bounds=(1e-2, 1e2),
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                alpha=1e-10,
            )
            gp.fit(X, z_col)
            return gp

        self.gps = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(Z[:, i]) for i in range(Z.shape[1])
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict PC scores at new parameter points.

        Parameters
        ----------
        X : array, shape (n_points, n_params)
        return_std : bool
            If True, also return predictive standard deviation.

        Returns
        -------
        mean : array, shape (n_points, n_components)
        std : array, shape (n_points, n_components), or None
        """
        means, stds = zip(*[gp.predict(X, return_std=True) for gp in self.gps])
        mean = np.column_stack(means)
        std = np.column_stack(stds)
        return mean, (std if return_std else None)

    def loo_cross_validate(self) -> dict:
        """Analytical leave-one-out (LOO) cross-validation.

        Uses the efficient analytical GPR LOO formula
        (Williams & Rasmussen 2006, §5.4.2): for a fitted GP with kernel
        matrix K and alpha vector a = K⁻¹y, the LOO predictive mean and
        variance at training point i are::

            μ_{-i} = y_i − a_i / [K⁻¹]_{ii}
            σ²_{-i} = 1 / [K⁻¹]_{ii}

        This avoids retraining n separate GPs and instead inverts the
        already-computed Cholesky factor once per output dimension.

        Returns
        -------
        dict
            y_pred      : ndarray (n_train, n_components) — LOO mean predictions
            y_std       : ndarray (n_train, n_components) — LOO predictive std devs
            y_true      : ndarray (n_train, n_components) — training targets
            r2          : ndarray (n_components,) — R² per PC
            rmse        : ndarray (n_components,) — RMSE per PC
            coverage_68 : float — fraction of LOO errors within ±1σ
            coverage_95 : float — fraction of LOO errors within ±2σ

        Raises
        ------
        RuntimeError
            If the emulator has not been fitted yet.
        """
        if not self.gps:
            raise RuntimeError(
                "Emulator must be fitted before calling loo_cross_validate."
            )

        loo_means, loo_stds, y_true_cols = [], [], []
        for gp in self.gps:
            n = len(gp.alpha_)
            K_inv = cho_solve((gp.L_, True), np.eye(n))
            diag = np.diag(K_inv)
            loo_means.append(gp.y_train_ - gp.alpha_ / diag)
            loo_stds.append(np.sqrt(np.maximum(1.0 / diag, 0.0)))
            y_true_cols.append(gp.y_train_)

        y_pred = np.column_stack(loo_means)
        y_std = np.column_stack(loo_stds)
        y_true = np.column_stack(y_true_cols)

        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2, axis=0)
        ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
        r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
        rmse = np.sqrt(np.mean(residuals ** 2, axis=0))

        z = np.abs(residuals) / np.where(y_std > 0, y_std, 1.0)
        return {
            "y_pred": y_pred,
            "y_std": y_std,
            "y_true": y_true,
            "r2": r2,
            "rmse": rmse,
            "coverage_68": float(np.mean(z <= 1.0)),
            "coverage_95": float(np.mean(z <= 2.0)),
        }
