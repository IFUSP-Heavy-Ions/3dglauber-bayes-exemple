"""Multi-fidelity GP emulator using emukit's linear autoregressive (AR1) model.

Trains a two-level emulator where cheap/abundant low-fidelity simulations
(e.g. SMASH without rescatterings) inform the expensive high-fidelity ones
(e.g. full collision runs), enabling accurate emulation from far fewer
high-fidelity points.

Requires the optional dependencies::

    pip install hic-bayes[multifidelity]
    # or directly:
    pip install emukit GPy
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from emukit.multi_fidelity.convert_lists_to_array import (
        convert_x_list_to_array,
        convert_xy_lists_to_arrays,
    )
    from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
    from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
    from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
    import GPy

    _EMUKIT_AVAILABLE = True
except ImportError:
    _EMUKIT_AVAILABLE = False


def _require_emukit() -> None:
    if not _EMUKIT_AVAILABLE:
        raise ImportError(
            "emukit and GPy are required for multi-fidelity emulation.\n"
            "Install with:  pip install hic-bayes[multifidelity]"
        )


@dataclass
class MultiFidelityEmulator:
    """Linear multi-fidelity GP emulator (AR1 model, two fidelity levels).

    The model is::

        f_high(x) = rho * f_low(x) + delta(x)

    where ``delta`` is an independent GP capturing the discrepancy.

    Parameters
    ----------
    n_optimization_restarts : int
        Number of random restarts for hyperparameter optimisation per PC.

    Examples
    --------
    >>> em = MultiFidelityEmulator()
    >>> em.fit(X_low, Z_low, X_high, Z_high)
    >>> mean, std = em.predict(X_new, return_std=True)
    """

    n_optimization_restarts: int = 5
    models: list = field(default_factory=list, repr=False)

    def fit(
        self,
        X_low: np.ndarray,
        Z_low: np.ndarray,
        X_high: np.ndarray,
        Z_high: np.ndarray,
    ) -> MultiFidelityEmulator:
        """Fit one multi-fidelity GP per output column.

        Parameters
        ----------
        X_low : array, shape (n_low, n_params)
            Low-fidelity design points.
        Z_low : array, shape (n_low, n_components)
            Low-fidelity PC scores.
        X_high : array, shape (n_high, n_params)
            High-fidelity design points.
        Z_high : array, shape (n_high, n_components)
            High-fidelity PC scores.

        Returns
        -------
        self
        """
        _require_emukit()
        n_components = Z_low.shape[1]
        n_params = X_low.shape[1]
        self.models = []

        for i in range(n_components):
            X_mf, Y_mf = convert_xy_lists_to_arrays(
                [X_low, X_high],
                [Z_low[:, i : i + 1], Z_high[:, i : i + 1]],
            )
            kernels = [
                GPy.kern.RBF(n_params),
                GPy.kern.RBF(n_params),
            ]
            lin_mf_kernel = LinearMultiFidelityKernel(kernels)
            gpy_model = GPyLinearMultiFidelityModel(
                X_mf, Y_mf, lin_mf_kernel, n_fidelities=2
            )
            model = GPyMultiOutputWrapper(
                gpy_model, 2, n_optimization_restarts=self.n_optimization_restarts
            )
            model.optimize()
            self.models.append(model)

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict at high-fidelity level for new parameter points.

        Parameters
        ----------
        X : array, shape (n_points, n_params)
        return_std : bool

        Returns
        -------
        mean : array, shape (n_points, n_components)
        std : array, shape (n_points, n_components), or None
        """
        _require_emukit()
        # Append fidelity index 1 (high) to each point
        fidelity_col = np.ones((len(X), 1))
        X_pred = np.hstack([X, fidelity_col])

        means, stds = [], []
        for model in self.models:
            m, v = model.predict(X_pred)
            means.append(m.ravel())
            stds.append(np.sqrt(np.maximum(v.ravel(), 0.0)))

        mean = np.column_stack(means)
        std = np.column_stack(stds)
        return mean, (std if return_std else None)
