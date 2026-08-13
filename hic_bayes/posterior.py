"""Posterior analysis: MAP estimation, credible intervals, and corner plots."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def map_estimate(
    flat_chain: np.ndarray,
    log_prob_flat: np.ndarray,
) -> np.ndarray:
    """Return the Maximum A Posteriori (MAP) parameter vector.

    Parameters
    ----------
    flat_chain : array, shape (n_samples, n_params)
    log_prob_flat : array, shape (n_samples,)

    Returns
    -------
    array, shape (n_params,)
    """
    return flat_chain[np.argmax(log_prob_flat)]


def credible_interval(
    flat_chain: np.ndarray,
    level: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equal-tailed credible intervals for each parameter.

    Parameters
    ----------
    flat_chain : array, shape (n_samples, n_params)
    level : float
        Credibility level (e.g. ``0.9`` for 90%).

    Returns
    -------
    lower, upper : arrays, shape (n_params,)
    """
    alpha = (1.0 - level) / 2.0
    return (
        np.quantile(flat_chain, alpha, axis=0),
        np.quantile(flat_chain, 1.0 - alpha, axis=0),
    )


def posterior_summary(
    flat_chain: np.ndarray,
    param_names: list[str],
    level: float = 0.9,
) -> dict[str, dict]:
    """Summarise the posterior for each parameter.

    Parameters
    ----------
    flat_chain : array, shape (n_samples, n_params)
    param_names : list of str
    level : float

    Returns
    -------
    dict mapping parameter name to a dict with keys
    ``'mean'``, ``'median'``, ``'std'``, ``'ci_low'``, ``'ci_high'``.
    """
    lower, upper = credible_interval(flat_chain, level)
    return {
        name: {
            "mean": float(flat_chain[:, i].mean()),
            "median": float(np.median(flat_chain[:, i])),
            "std": float(flat_chain[:, i].std()),
            "ci_low": float(lower[i]),
            "ci_high": float(upper[i]),
        }
        for i, name in enumerate(param_names)
    }


def corner_plot(
    flat_chain: np.ndarray,
    param_labels: list[str],
    truths: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    font_scale: float = 1.15,
    axis_label_fontsize: int = 8,
    tick_label_fontsize: int = 6,
    tick_rotation: int = 45,
    tick_pad: int = 2,
):
    """Produce a Universality-style corner plot using seaborn PairGrid.

    Parameters
    ----------
    flat_chain : array, shape (n_samples, n_params)
    param_labels : list of str
        LaTeX or plain-text labels for each parameter.
    truths : array or None
        True parameter values to overlay (useful for closure tests).
    title : str or None
    font_scale : float
        Seaborn context scale.  Keep this modest for dense corner plots.
    axis_label_fontsize, tick_label_fontsize : int
        Font sizes for parameter labels and numerical tick labels.
    tick_rotation : int
        Rotation angle for numerical tick labels.
    tick_pad : int
        Padding between axes and numerical tick labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("Install seaborn with: pip install seaborn")

    if flat_chain.ndim != 2:
        raise ValueError("flat_chain must be a 2-D array of posterior samples.")
    if flat_chain.shape[1] != len(param_labels):
        raise ValueError("param_labels must have the same length as the chain dimension.")

    df = pd.DataFrame(flat_chain, columns=param_labels)

    sns.set_context("notebook", font_scale=font_scale)
    sns.set_style("ticks")

    n_dim = len(param_labels)
    height = max(1.3, 12.0 / max(n_dim, 1))
    g = sns.PairGrid(
        df,
        vars=param_labels,
        corner=True,
        diag_sharey=False,
        height=height,
    )
    g.map_lower(
        lambda x, y, **kwargs: sns.kdeplot(
            x=x,
            y=y,
            fill=True,
            levels=5,
            thresh=0.06,
            alpha=0.25,
            **kwargs,
        )
    )
    g.map_diag(
        sns.kdeplot,
        fill=True,
        alpha=0.55,
        linewidth=1.6,
    )

    if truths is not None:
        if len(truths) != len(param_labels):
            raise ValueError("truths must match the number of plotted parameters.")
        for idx, truth in enumerate(truths):
            ax = g.axes[idx][idx]
            ax.axvline(truth, linestyle="--", color="black", linewidth=1.0)

    fig = g.figure
    if title:
        fig.suptitle(title, y=1.01)

    for row in g.axes:
        for ax in row:
            if ax is None:
                continue
            ax.xaxis.label.set_size(axis_label_fontsize)
            ax.yaxis.label.set_size(axis_label_fontsize)
            ax.tick_params(
                axis="x",
                labelsize=tick_label_fontsize,
                labelrotation=tick_rotation,
                pad=tick_pad,
            )
            ax.tick_params(
                axis="y",
                labelsize=tick_label_fontsize,
                labelrotation=tick_rotation,
                pad=tick_pad,
            )

    fig.tight_layout()
    return fig
