"""SVD-based PCA for dimensionality reduction of model outputs.

Follows the normalisation convention used across all source repositories:
PC scores are rescaled so that each retained PC has unit variance on the
training set, making emulator targets well-conditioned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    """Container for a fitted PCA transformation.

    Attributes
    ----------
    scaler : StandardScaler
        Fitted scaler (zero mean, unit variance per observable).
    vh : np.ndarray, shape (n_obs, n_obs)
        Right singular vectors (PC basis).
    singular_values : np.ndarray, shape (min(n_pts, n_obs),)
    n_components : int
        Number of retained PCs.
    n_training_points : int
    """

    scaler: StandardScaler
    vh: np.ndarray
    singular_values: np.ndarray
    n_components: int
    n_training_points: int

    @property
    def variance_explained(self) -> np.ndarray:
        """Fraction of total variance explained by each retained PC."""
        importance = self.singular_values ** 2 / (self.n_training_points - 1)
        return importance[: self.n_components] / importance.sum()

    @property
    def cumulative_variance(self) -> np.ndarray:
        """Cumulative fraction of variance explained."""
        return np.cumsum(self.variance_explained)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Project observations into PC space.

        Parameters
        ----------
        Y : array, shape (n_samples, n_obs)

        Returns
        -------
        array, shape (n_samples, n_components)
            Unit-variance PC scores on the training set.
        """
        Ys = self.scaler.transform(Y)
        scores = Ys @ self.vh[: self.n_components].T
        norm = self.singular_values[: self.n_components] / np.sqrt(
            self.n_training_points - 1
        )
        return scores / norm

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct observations from PC scores.

        Parameters
        ----------
        Z : array, shape (n_samples, n_components)

        Returns
        -------
        array, shape (n_samples, n_obs)
        """
        norm = self.singular_values[: self.n_components] / np.sqrt(
            self.n_training_points - 1
        )
        Ys = (Z * norm) @ self.vh[: self.n_components]
        return self.scaler.inverse_transform(Ys)

    @property
    def transform_matrix(self) -> np.ndarray:
        """Linear map from unit-variance PC scores to observables.

        Shape: (n_components, n_obs).  For scores Z of shape
        (..., n_components), the reconstruction is::

            Y ≈ Z @ transform_matrix + scaler.mean_

        Useful for propagating per-component standard deviations into
        observable space::

            std_obs = sqrt(std_pc**2 @ transform_matrix**2)
        """
        norm = self.singular_values[: self.n_components] / np.sqrt(
            self.n_training_points - 1
        )
        return (
            norm[:, np.newaxis]
            * self.vh[: self.n_components]
            * self.scaler.scale_[np.newaxis, :]
        )

    def inverse_transform_cov(self, Z_cov: np.ndarray) -> np.ndarray:
        """Propagate a PC-space covariance matrix back to observable space.

        Parameters
        ----------
        Z_cov : array, shape (n_components, n_components)

        Returns
        -------
        array, shape (n_obs, n_obs)
        """
        norm = self.singular_values[: self.n_components] / np.sqrt(
            self.n_training_points - 1
        )
        B = np.diag(norm) @ self.vh[: self.n_components]  # (n_comp, n_obs)
        # scale back through StandardScaler
        s = self.scaler.scale_  # (n_obs,)
        B_obs = B * s  # broadcast: multiply each row of B by s
        return B_obs.T @ Z_cov @ B_obs


def fit_pca(Y: np.ndarray, n_components: int) -> PCAResult:
    """Fit PCA to the model output matrix.

    Parameters
    ----------
    Y : array, shape (n_samples, n_observables)
        Matrix of simulation outputs.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    PCAResult
    """
    scaler = StandardScaler()
    Ys = scaler.fit_transform(Y)
    n_pts = Ys.shape[0]
    _, s, vh = np.linalg.svd(Ys, full_matrices=True)
    return PCAResult(
        scaler=scaler,
        vh=vh,
        singular_values=s,
        n_components=n_components,
        n_training_points=n_pts,
    )
