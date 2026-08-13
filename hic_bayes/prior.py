"""Prior distributions for Bayesian parameter estimation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UniformPrior:
    """Independent uniform priors on each parameter.

    Parameters
    ----------
    mins : array-like, shape (n_params,)
        Lower bounds.
    maxs : array-like, shape (n_params,)
        Upper bounds.

    Examples
    --------
    >>> prior = UniformPrior(mins=[0.0, 0.1], maxs=[0.5, 1.5])
    >>> prior.log_prob(np.array([0.3, 0.8]))
    0.0
    >>> prior.log_prob(np.array([0.6, 0.8]))
    -inf
    """

    mins: np.ndarray
    maxs: np.ndarray

    def __post_init__(self) -> None:
        self.mins = np.asarray(self.mins, dtype=float)
        self.maxs = np.asarray(self.maxs, dtype=float)
        if np.any(self.mins >= self.maxs):
            raise ValueError("All mins must be strictly less than maxs.")

    def log_prob(self, theta: np.ndarray) -> float:
        """Return 0 if theta is in bounds, -inf otherwise."""
        theta = np.asarray(theta)
        if np.any(theta < self.mins) or np.any(theta > self.maxs):
            return -np.inf
        return 0.0

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw n uniform samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples.
        seed : int, optional

        Returns
        -------
        array, shape (n, n_params)
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(self.mins, self.maxs, size=(n, len(self.mins)))

    @property
    def ranges(self) -> np.ndarray:
        """Parameter ranges as array of shape (n_params, 2): [[min, max], ...]."""
        return np.column_stack([self.mins, self.maxs])

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.mins)

    @classmethod
    def from_config(cls, params: list[dict]) -> UniformPrior:
        """Build from a list of parameter dicts as defined in the YAML config.

        Each dict must have keys ``'min'`` and ``'max'``.

        Parameters
        ----------
        params : list of dict
            E.g. ``[{'name': 'eta_s', 'min': 0.0, 'max': 0.5}, ...]``

        Returns
        -------
        UniformPrior
        """
        mins = [p["min"] for p in params]
        maxs = [p["max"] for p in params]
        return cls(mins=mins, maxs=maxs)
