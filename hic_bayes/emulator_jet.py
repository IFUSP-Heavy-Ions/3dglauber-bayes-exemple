"""JETSCAPE-compatible PCA and Gaussian-process emulator helpers.

This module mirrors the training choices in ``jetscape-emulator/emulator.py``
for studies that compare a JETSCAPE-trained ``.dill`` emulator with one trained
inside hic-bayes from the same flat CSV tables.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass
class JetPCAState:
    """PCA state using the same convention as JETSCAPE's emulator."""

    scaler: StandardScaler
    pca: PCA
    n_components: int
    transform_matrix: np.ndarray
    cov_trunc: np.ndarray


class _Covariance:
    """Proxy object matching the JETSCAPE emulator covariance interface."""

    def __init__(self, array: np.ndarray, slices: dict[str, slice]):
        self.array = array
        self._slices = slices

    def __getitem__(self, key):
        obs1, obs2 = key
        return self.array[..., self._slices[obs1], self._slices[obs2]]


def observable_slices(columns: list[str]) -> tuple[list[str], dict[str, slice]]:
    """Group flat ``observable__binNN`` columns into JETSCAPE-style slices."""
    observables: list[str] = []
    slices: dict[str, slice] = {}
    start = 0
    idx = 0
    while idx < len(columns):
        obs = columns[idx].rsplit("__bin", 1)[0] if "__bin" in columns[idx] else columns[idx]
        stop = idx + 1
        while stop < len(columns):
            next_obs = columns[stop].rsplit("__bin", 1)[0] if "__bin" in columns[stop] else columns[stop]
            if next_obs != obs:
                break
            stop += 1
        observables.append(obs)
        slices[obs] = slice(start, start + (stop - idx))
        start += stop - idx
        idx = stop
    return observables, slices


@dataclass
class JetCompatibleEmulator:
    """Emulator trained in hic-bayes with JETSCAPE PCA/GP conventions."""

    gps: list[GaussianProcessRegressor]
    pca_state: JetPCAState
    observables: list[str]
    slices: dict[str, slice]
    transform_design: bool = True

    @classmethod
    def fit(
        cls,
        design: np.ndarray,
        simulation: np.ndarray,
        param_ranges: np.ndarray,
        observable_columns: list[str],
        n_components: int,
        n_restarts: int = 4,
        n_jobs: int = -1,
        system_name: str | None = None,
        idf: int | str = 0,
        transform_design: bool = True,
    ) -> "JetCompatibleEmulator":
        observables, slices = observable_slices(observable_columns)
        LOGGER.info("Emulators for system %s", system_name or "<unknown>")
        LOGGER.info("with viscous correction type %s", idf)
        LOGGER.info("NPC: %d", n_components)
        LOGGER.info("Nrestart: %d", n_restarts)
        LOGGER.info("self.nobs = %d", simulation.shape[1])
        LOGGER.info("Active observables (%d): %s", len(observables), observables)
        pca_state, scores = fit_jet_pca(simulation, n_components)
        if transform_design:
            design = transform_jet_design(design)
            param_ranges = transform_jet_design_min_max(param_ranges)
            LOGGER.info("Transformed design to JETSCAPE viscosity grid: %s", design.shape)
        gps = train_jet_emulators(
            design,
            scores,
            param_ranges,
            n_jobs=n_jobs,
            n_restarts=n_restarts,
        )
        return cls(
            gps=gps,
            pca_state=pca_state,
            observables=observables,
            slices=slices,
            transform_design=transform_design,
        )

    def _inverse_transform(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        y = scores @ self.pca_state.transform_matrix[: scores.shape[-1]]
        y += self.pca_state.scaler.mean_
        return {obs: y[..., self.slices[obs]] for obs in self.observables}

    def predict(self, X, return_cov: bool = False, extra_std=0):
        X = np.asarray(X)
        if self.transform_design:
            X = transform_jet_design(X)
        gp_outputs = [gp.predict(X, return_cov=return_cov) for gp in self.gps]
        if return_cov:
            gp_mean, gp_cov = zip(*gp_outputs)
        else:
            gp_mean = gp_outputs

        mean = self._inverse_transform(
            np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
        )

        if not return_cov:
            return mean

        gp_var = np.concatenate(
            [cov.diagonal()[:, np.newaxis] for cov in gp_cov],
            axis=1,
        )
        extra_std = np.asarray(extra_std).reshape(-1, 1)
        gp_var += extra_std**2

        A = self.pca_state.transform_matrix[: len(self.gps)]
        var_trans = np.einsum("ki,kj->kij", A, A, optimize=False).reshape(
            len(self.gps),
            self.pca_state.cov_trunc.size,
        )
        cov = np.dot(gp_var, var_trans).reshape(
            X.shape[0],
            self.pca_state.cov_trunc.shape[0],
            self.pca_state.cov_trunc.shape[1],
        )
        cov += self.pca_state.cov_trunc
        return mean, _Covariance(cov, self.slices)


def zeta_over_s(T, zmax, T0, width, asym):
    """JETSCAPE zeta/s parametrization used for design transformation."""
    delta_t = T - T0
    sign = np.where(delta_t > 0, 1.0, -1.0)
    x = delta_t / (width * (1.0 + asym * sign))
    return zmax / (1.0 + x**2)


def eta_over_s(T, t_kink, slope_low, slope_high, eta_kink):
    """JETSCAPE eta/s parametrization used for design transformation."""
    y = np.where(
        T < t_kink,
        eta_kink + slope_low * (T - t_kink),
        eta_kink + slope_high * (T - t_kink),
    )
    return np.maximum(y, 0.0)


def transform_jet_design(X: np.ndarray) -> np.ndarray:
    """Apply JETSCAPE's 20-parameter -> 32-feature viscosity design transform."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != 20:
        raise ValueError(f"JETSCAPE design transform expects 20 columns, got {X.shape[1]}.")

    kept_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    extra_indices = [19]
    temperature_grid = np.linspace(0.13, 0.3, 10)

    eta_vals = np.array(
        [
            eta_over_s(T, X[:, 11], X[:, 12], X[:, 13], X[:, 14])
            for T in temperature_grid
        ]
    ).T
    zeta_vals = np.array(
        [
            zeta_over_s(T, X[:, 15], X[:, 16], X[:, 17], X[:, 18])
            for T in temperature_grid
        ]
    ).T
    return np.concatenate([X[:, kept_indices], eta_vals, zeta_vals, X[:, extra_indices]], axis=1)


def transform_jet_design_min_max(param_ranges: np.ndarray) -> np.ndarray:
    """Transform physical parameter min/max ranges to JETSCAPE GP feature ranges."""
    param_ranges = np.asarray(param_ranges, dtype=float)
    if param_ranges.shape != (20, 2):
        raise ValueError(f"JETSCAPE range transform expects shape (20, 2), got {param_ranges.shape}.")

    kept_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    extra_indices = [19]
    transformed_min = np.concatenate(
        [
            param_ranges[kept_indices, 0],
            np.full(10, 0.01),
            np.full(10, 0.01),
            param_ranges[extra_indices, 0],
        ]
    )
    transformed_max = np.concatenate(
        [
            param_ranges[kept_indices, 1],
            np.full(10, 0.65),
            np.full(10, 0.2),
            param_ranges[extra_indices, 1],
        ]
    )
    return np.column_stack([transformed_min, transformed_max])


def fit_jet_pca(Y: np.ndarray, n_components: int) -> tuple[JetPCAState, np.ndarray]:
    """Fit StandardScaler + sklearn PCA exactly as in JETSCAPE.

    JETSCAPE standardizes every observable, fits ``PCA(whiten=True,
    svd_solver="full")`` to all components, and then emulates only the first
    ``n_components`` whitened PC scores.  Neglected PCs contribute a truncation
    covariance in observable space.
    """
    LOGGER.info("Fitting JETSCAPE-style PCA")
    LOGGER.info("Y_Obs shape[Ndesign, Nobs] = %s", Y.shape)
    scaler = StandardScaler(copy=True)
    pca = PCA(copy=False, whiten=True, svd_solver="full")
    scores = pca.fit_transform(scaler.fit_transform(Y))[:, :n_components]

    transform_matrix = (
        pca.components_
        * np.sqrt(pca.explained_variance_[:, np.newaxis])
        * scaler.scale_
    )
    neglected = transform_matrix[n_components:]
    cov_trunc = neglected.T @ neglected
    cov_trunc.flat[:: cov_trunc.shape[0] + 1] += 1e-4 * scaler.var_
    explained = pca.explained_variance_ratio_[:n_components].sum()
    LOGGER.info(
        "PCA retained %d/%d components; cumulative explained variance = %.6f",
        n_components,
        pca.components_.shape[0],
        explained,
    )
    LOGGER.info("Truncation covariance shape = %s", cov_trunc.shape)

    return (
        JetPCAState(
            scaler=scaler,
            pca=pca,
            n_components=n_components,
            transform_matrix=transform_matrix,
            cov_trunc=cov_trunc,
        ),
        scores,
    )


def _fit_one_gp(index: int, z_col: np.ndarray, design: np.ndarray, kernel, n_restarts: int):
    logger = logging.getLogger(__name__)
    logger.info("Fitting PC #%d", index)
    start = time.time()
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.001,
        n_restarts_optimizer=n_restarts,
        copy_X_train=True,
        random_state=32,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(design, z_col)
    for warning in caught:
        logger.warning("PC #%d: %s: %s", index, warning.category.__name__, warning.message)
    logger.info("Finished PC #%d in %.1f s", index, time.time() - start)
    return gp


def train_jet_emulators(
    design: np.ndarray,
    pc_scores: np.ndarray,
    param_ranges: np.ndarray,
    n_jobs: int = -1,
    n_restarts: int = 4,
) -> list[GaussianProcessRegressor]:
    """Train one JETSCAPE-compatible sklearn GP per retained PC."""
    ptp = param_ranges[:, 1] - param_ranges[:, 0]
    kernel = 1.0 * kernels.RBF(
        length_scale=ptp,
        length_scale_bounds=np.outer(ptp, (2e-1, 1e2)),
    ) + kernels.WhiteKernel(
        noise_level=0.1,
        noise_level_bounds=(1e-4, 1e1),
    )
    LOGGER.info("Training %d independent GP emulators", pc_scores.shape[1])
    LOGGER.info("Design shape[Ndesign, Nparams] = %s", design.shape)
    LOGGER.info("Parameter range widths = %s", np.array2string(ptp, precision=6))
    LOGGER.info("Kernel template = %s", kernel)
    LOGGER.info("Parallel jobs = %s", n_jobs)
    with parallel_backend("threading", n_jobs=n_jobs):
        return Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_gp)(i, pc_scores[:, i], design, kernel, n_restarts)
            for i in range(pc_scores.shape[1])
        )


def predict_jet_obs(
    theta: np.ndarray,
    emulators: list[GaussianProcessRegressor],
    pca_state: JetPCAState,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict observable-space mean/covariance with JETSCAPE conventions."""
    theta_2d = np.array(theta).reshape(1, -1)
    gp_outputs = [gp.predict(theta_2d, return_cov=True) for gp in emulators]
    pc_means, pc_covs = zip(*gp_outputs)
    pc_mean = np.concatenate([mean[:, np.newaxis] for mean in pc_means], axis=1)
    mean = pc_mean @ pca_state.transform_matrix[: pc_mean.shape[-1]]
    mean = mean.reshape(-1) + pca_state.scaler.mean_

    pc_var = np.concatenate(
        [cov.diagonal()[:, np.newaxis] for cov in pc_covs],
        axis=1,
    ).reshape(-1)
    A = pca_state.transform_matrix[: len(emulators)]
    cov = A.T @ np.diag(pc_var) @ A
    cov += pca_state.cov_trunc
    return mean, cov
