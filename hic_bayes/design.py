"""Latin Hypercube and Sobol sequence design generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


def lhs_design(
    param_bounds: dict[str, tuple[float, float]],
    n_points: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a Latin Hypercube Sample design.

    Parameters
    ----------
    param_bounds : dict
        ``{name: (min, max)}`` for each parameter.
    n_points : int
        Number of design points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Design matrix with one column per parameter.
    """
    names = list(param_bounds.keys())
    bounds = np.array(list(param_bounds.values()))
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    sample = sampler.random(n=n_points)
    scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    return pd.DataFrame(scaled, columns=names)


def sobol_design(
    param_bounds: dict[str, tuple[float, float]],
    n_points: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a Sobol sequence design.

    Parameters
    ----------
    param_bounds : dict
        ``{name: (min, max)}`` for each parameter.
    n_points : int
        Requested number of points (rounded up to next power of 2 internally).
    seed : int
        Random seed for scrambling.

    Returns
    -------
    pd.DataFrame
        Design matrix with one column per parameter.
    """
    names = list(param_bounds.keys())
    bounds = np.array(list(param_bounds.values()))
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    sample = sampler.random(n=n_points)
    scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    return pd.DataFrame(scaled, columns=names)


def load_design(
    path: str | Path,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load design points from a CSV (or whitespace-delimited) text file.

    Parameters
    ----------
    path : str or Path
    drop_columns : list of str, optional
        Column names to drop after loading (e.g. redundant timing columns).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(Path(path))
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    return df


def remove_nan_rows(*dfs: pd.DataFrame, reference_index: int = 0) -> list[pd.DataFrame]:
    """Drop rows that contain NaN in the reference DataFrame from all DataFrames.

    All DataFrames must have the same number of rows.

    Parameters
    ----------
    *dfs : pd.DataFrame
        DataFrames to filter, all with the same row count.
    reference_index : int
        Which DataFrame to check for NaNs (default: first).

    Returns
    -------
    list of pd.DataFrame
        Filtered DataFrames with consistent indices.
    """
    mask = ~dfs[reference_index].isna().any(axis=1)
    return [df[mask].reset_index(drop=True) for df in dfs]
