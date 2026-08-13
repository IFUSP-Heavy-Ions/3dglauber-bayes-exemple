"""Shared plotting utilities for hic-bayes analyses."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_variance_explained(
    variance_explained: np.ndarray,
    max_pcs: int = 10,
    ax: Optional[tuple] = None,
) -> plt.Figure:
    """Bar charts of individual and cumulative variance explained by PCs.

    Parameters
    ----------
    variance_explained : array, shape (n_components,)
    max_pcs : int
        Maximum number of PCs to plot.
    ax : tuple of (ax1, ax2), optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = min(max_pcs, len(variance_explained))
    idx = np.arange(1, n + 1)
    cumulative = np.cumsum(variance_explained[:n])

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    ax1.bar(idx, variance_explained[:n])
    ax1.set_xlabel("PC index")
    ax1.set_ylabel("Variance explained")
    ax1.set_title("Variance per PC")

    ax2.bar(idx, cumulative)
    ax2.axhline(0.99, color="red", linestyle="--", lw=1, label="99%")
    ax2.axhline(0.95, color="orange", linestyle="--", lw=1, label="95%")
    ax2.set_xlabel("First $n$ PCs")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("Cumulative variance")
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_emulator_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    observable_labels: Optional[list[str]] = None,
    n_cols: int = 5,
) -> plt.Figure:
    """Predicted-vs-true scatter plots, one panel per observable.

    Parameters
    ----------
    y_true : array, shape (n_val, n_obs)
    y_pred : array, shape (n_val, n_obs)
    y_std : array, shape (n_val, n_obs), optional
        If given, plot 2σ error bars.
    observable_labels : list of str, optional
    n_cols : int
        Number of columns in the subplot grid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_obs = y_true.shape[1]
    n_rows = int(np.ceil(n_obs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n_obs):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=12, alpha=0.6, zorder=3)
        if y_std is not None:
            ax.errorbar(
                y_true[:, i],
                y_pred[:, i],
                yerr=2 * y_std[:, i],
                fmt="none",
                alpha=0.25,
                zorder=2,
            )
        lims = [
            min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "r--", lw=1.0, zorder=1)
        if observable_labels:
            ax.set_title(observable_labels[i], fontsize=7)
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)

    for j in range(n_obs, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_trace(
    chain: np.ndarray,
    param_labels: list[str],
    burnin: Optional[int] = None,
) -> plt.Figure:
    """Walker trace plots for MCMC convergence inspection.

    Parameters
    ----------
    chain : array, shape (n_walkers, n_steps, n_params)
    param_labels : list of str
    burnin : int, optional
        If given, shade the burn-in region in red.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_params = chain.shape[2]
    fig, axes = plt.subplots(
        n_params, 1, figsize=(10, 2 * n_params), sharex=True
    )
    if n_params == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, param_labels)):
        ax.plot(chain[:, :, i].T, alpha=0.3, lw=0.5, color="steelblue")
        ax.set_ylabel(label, fontsize=9)
        if burnin is not None:
            ax.axvspan(
                0,
                burnin,
                alpha=0.15,
                color="red",
                label="burn-in" if i == 0 else None,
            )

    axes[-1].set_xlabel("Step")
    if burnin is not None:
        axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_calibration_report(
    report: dict,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Display calibration metric values as a simple table-style figure.

    Parameters
    ----------
    report : dict
        Output of :func:`hic_bayes.calibration.calibration_report`.
    ax : plt.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2))
    else:
        fig = ax.figure

    rows = [
        ["Metric", "Value"],
        ["Mahalanobis dist.", f"{report['mahalanobis']:.3f}"],
        ["Energy Score", f"{report['energy_score']:.4f}"],
        ["KNN log-density", f"{report['log_density']:.3f}"],
    ]
    ax.axis("off")
    tbl = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(True)
    tbl.scale(1.2, 1.6)
    ax.set_title("Posterior calibration", fontsize=10, pad=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Leave-one-out emulator validation
# ---------------------------------------------------------------------------

def plot_loo_validation(
    loo_result: dict,
    pca=None,
    observable_labels: Optional[list] = None,
    n_cols: int = 5,
) -> plt.Figure:
    """Predicted-vs-true scatter plots from LOO cross-validation.

    Parameters
    ----------
    loo_result : dict
        Output of :meth:`hic_bayes.emulator.Emulator.loo_cross_validate`.
    pca : hic_bayes.pca.PCAResult, optional
        If given, maps PC-space results back to observable space and
        propagates predictive uncertainties through the linear PCA basis.
    observable_labels : list of str, optional
        Panel titles.  Only used when *pca* is provided.
    n_cols : int
        Number of subplot columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = loo_result["y_true"]      # (n_train, n_comp)
    y_pred = loo_result["y_pred"]
    y_std  = loo_result["y_std"]       # (n_train, n_comp) in PC space
    r2_pc  = loo_result["r2"]
    cov68  = loo_result["coverage_68"]
    cov95  = loo_result["coverage_95"]

    if pca is not None:
        # Map to observable space
        y_true = pca.inverse_transform(y_true)
        y_pred = pca.inverse_transform(y_pred)
        # Propagate std through linear PCA basis (independent PC components)
        M = pca.transform_matrix   # (n_comp, n_obs)
        y_std = np.sqrt(y_std ** 2 @ M ** 2)   # (n_train, n_obs)
        # Per-observable R²
        res = y_true - y_pred
        ss_res = np.sum(res ** 2, axis=0)
        ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
        r2_plot = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
        def _label(i):
            lbl = observable_labels[i] if observable_labels else f"Obs {i}"
            return f"{lbl}\n$R^2$={r2_plot[i]:.2f}"
    else:
        def _label(i):
            return f"PC {i + 1}  $R^2$={r2_pc[i]:.2f}"

    n_dim = y_true.shape[1]
    n_cols = min(n_cols, n_dim)
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n_dim):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=12, alpha=0.6, zorder=3)
        ax.errorbar(
            y_true[:, i], y_pred[:, i],
            yerr=2 * y_std[:, i],
            fmt="none", alpha=0.20, zorder=2,
        )
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, zorder=1)
        ax.set_title(_label(i), fontsize=7)
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("LOO pred.", fontsize=8)

    for j in range(n_dim, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"LOO emulator validation  |  68% cov: {cov68:.1%}  |  95% cov: {cov95:.1%}",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prior and posterior predictive checks
# ---------------------------------------------------------------------------

def plot_predictive_check(
    samples: np.ndarray,
    emulator,
    pca,
    y_exp: np.ndarray,
    y_err: np.ndarray,
    observable_labels: Optional[list] = None,
    quantiles: tuple = (0.05, 0.50, 0.95),
    n_cols: int = 5,
    title: str = "Predictive check",
    universality_finish: bool = True,
) -> plt.Figure:
    """Prior or posterior predictive check: model envelope vs experimental data.

    Parameter samples are pushed through the emulator and inverse-PCA to
    produce predicted observables.  Each panel shows the 90% prediction
    band, the median, and the data point with its uncertainty.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Parameter samples — use :func:`hic_bayes.prior.UniformPrior.sample`
        for a prior check or the ``flat_chain`` from MCMC for a posterior
        check.
    emulator : hic_bayes.emulator.Emulator
    pca : hic_bayes.pca.PCAResult
    y_exp : ndarray, shape (n_obs,)
        Experimental central values.
    y_err : ndarray, shape (n_obs,)
        Experimental 1σ uncertainties.
    observable_labels : list of str, optional
    quantiles : tuple of 3 floats
        Lower, median, upper quantile fractions (default: 5%, 50%, 95%).
    n_cols : int
        Number of subplot columns.
    title : str
        Figure suptitle.  Suggest "Prior predictive check" or
        "Posterior predictive check".
    universality_finish : bool
        If ``True``, uses the Universality_Github finishing style for the
        uncertainty band: gray face, black edge, and hatched fill.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Z_pred, _ = emulator.predict(samples)
    Y_pred = pca.inverse_transform(Z_pred)   # (n_samples, n_obs)

    q_lo, q_med, q_hi = [np.quantile(Y_pred, q, axis=0) for q in quantiles]

    n_obs = Y_pred.shape[1]
    n_cols = min(n_cols, n_obs)
    n_rows = int(np.ceil(n_obs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n_obs):
        ax = axes[i]
        lbl = observable_labels[i] if observable_labels else f"Obs {i}"

        # Prediction band (horizontal, x in [0, 1])
        if universality_finish:
            ax.axhspan(
                q_lo[i],
                q_hi[i],
                alpha=0.30,
                facecolor="gray",
                edgecolor="black",
                hatch="///",
                linewidth=0.8,
                label=f"{int((quantiles[2]-quantiles[0])*100)}% band" if i == 0 else None,
            )
            ax.axhline(
                q_med[i],
                color="black",
                lw=1.5,
                label="Median" if i == 0 else None,
            )
        else:
            ax.axhspan(
                q_lo[i], q_hi[i],
                alpha=0.30, color="steelblue",
                label=f"{int((quantiles[2]-quantiles[0])*100)}% band" if i == 0 else None,
            )
            ax.axhline(q_med[i], color="steelblue", lw=1.5,
                       label="Median" if i == 0 else None)

        # Experimental datum
        ax.errorbar(
            0.5, y_exp[i], yerr=y_err[i],
            fmt="o", color="tomato", ms=5, capsize=3, zorder=5,
            label="Data ±1σ" if i == 0 else None,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_title(lbl, fontsize=7)
        ax.set_ylabel("Value", fontsize=8)

    for j in range(n_obs, len(axes)):
        axes[j].set_visible(False)

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower right", fontsize=9, ncol=3,
               bbox_to_anchor=(1.0, 0.0))
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity (Sobol' / Jacobian) plots
# ---------------------------------------------------------------------------

def plot_sensitivity(
    sensitivity_result: dict,
    param_labels: list,
    observable_labels: Optional[list] = None,
    kind: str = "bar",
) -> plt.Figure:
    """Visualise Sobol' sensitivity indices.

    Parameters
    ----------
    sensitivity_result : dict
        Output of :func:`hic_bayes.sensitivity.sobol_sensitivity`.
    param_labels : list of str
    observable_labels : list of str, optional
        Required when *kind* = ``'heatmap'``.
    kind : {'bar', 'heatmap'}
        ``'bar'``     — aggregated first-order + total-effect side-by-side.
        ``'heatmap'`` — per-observable first-order indices as a 2-D heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    S1  = sensitivity_result["first_order"]
    ST  = sensitivity_result["total_effect"]
    S1p = sensitivity_result["first_order_per_obs"]
    n_params = len(param_labels)
    x = np.arange(n_params)

    if kind == "bar":
        fig, ax = plt.subplots(figsize=(max(6, n_params * 0.9 + 1), 4))
        w = 0.35
        ax.bar(x - w / 2, np.clip(S1,  0, None), w, label=r"First-order $S_i$",   alpha=0.85)
        ax.bar(x + w / 2, np.clip(ST,  0, None), w, label=r"Total effect $S_{Ti}$", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=20, ha="right")
        ax.set_ylabel("Sobol' index")
        ax.set_ylim(0, min(1.05, max(float(ST.max()) * 1.25 if len(ST) else 0.1, 0.1)))
        ax.set_title("Global sensitivity analysis (Sobol')")
        ax.legend()
        plt.tight_layout()

    else:  # heatmap
        n_obs = S1p.shape[1]
        obs_lbls = observable_labels or [f"Obs {i}" for i in range(n_obs)]
        fig, ax = plt.subplots(
            figsize=(max(8, n_obs * 0.45 + 1), max(3, n_params * 0.55 + 1))
        )
        im = ax.imshow(
            np.clip(S1p, 0, 1),
            aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
        )
        ax.set_xticks(range(n_obs))
        ax.set_xticklabels(obs_lbls, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(n_params))
        ax.set_yticklabels(param_labels, fontsize=8)
        ax.set_title("First-order Sobol' indices per observable")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        plt.tight_layout()

    return fig


def plot_local_sensitivity(
    local_result: dict,
    param_labels: list,
    observable_labels: Optional[list] = None,
    normalised: bool = True,
) -> plt.Figure:
    """Heatmap of the emulator Jacobian at a single parameter point.

    Parameters
    ----------
    local_result : dict
        Output of :func:`hic_bayes.sensitivity.local_sensitivity`.
    param_labels : list of str
    observable_labels : list of str, optional
    normalised : bool
        If True, plot ``jacobian_normalised`` (elasticity); otherwise the
        raw Jacobian (units: observable / parameter).

    Returns
    -------
    matplotlib.figure.Figure
    """
    J = local_result["jacobian_normalised" if normalised else "jacobian"]
    n_params, n_obs = J.shape
    obs_lbls = observable_labels or [f"Obs {i}" for i in range(n_obs)]

    fig, ax = plt.subplots(
        figsize=(max(8, n_obs * 0.45 + 1), max(3, n_params * 0.55 + 1))
    )
    vmax = np.abs(J).max() or 1.0
    im = ax.imshow(J, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_obs))
    ax.set_xticklabels(obs_lbls, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(param_labels, fontsize=8)
    title = "Normalised local sensitivity (Jacobian)" if normalised else "Local sensitivity (Jacobian)"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    return fig
