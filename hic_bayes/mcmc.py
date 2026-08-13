"""MCMC wrappers for emcee (ensemble sampler) and ptemcee (parallel-tempering).

Both samplers expose the same output convention:
    chain    : (n_walkers, n_steps, n_params)
    log_prob : (n_walkers, n_steps)
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Callable
import sys
import time

import h5py
import numpy as np


def _ensure_numpy_legacy_aliases() -> None:
    """Provide NumPy aliases removed in NumPy>=2 for legacy dependencies.

    ptemcee still references aliases like ``np.float`` internally.
    """
    legacy_aliases = {
        "float": float,
        "int": int,
        "bool": bool,
        "complex": complex,
    }
    for alias, target in legacy_aliases.items():
        if not hasattr(np, alias):
            setattr(np, alias, target)


def _progress_interval(n_steps: int) -> int:
    """Return a step interval that reports progress roughly every 10%."""
    return max(1, n_steps // 10)


def _use_tqdm_bar(progress: bool) -> bool:
    """Show tqdm only when the caller asked for progress and stderr is a TTY."""
    return progress and sys.stderr.isatty()


def _print_step_progress(step: int, n_steps: int, started_at: float, label: str) -> None:
    """Print a compact progress line that is friendly to log files."""
    elapsed = time.time() - started_at
    pct = 100.0 * step / n_steps
    print(
        f"[{label}] step {step:>6d}/{n_steps:<6d}  ({pct:5.1f}%)  elapsed={elapsed:7.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# emcee
# ---------------------------------------------------------------------------

def run_emcee(
    log_prob_fn: Callable[[np.ndarray], float],
    n_walkers: int,
    n_steps: int,
    n_params: int,
    p0: np.ndarray,
    output_file: str | None = None,
    n_jobs: int = 1,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the emcee ensemble sampler.

    Parameters
    ----------
    log_prob_fn : callable
        Takes a 1-D parameter array, returns a scalar log-probability.
    n_walkers : int
        Number of ensemble walkers (must be even and > 2 * n_params).
    n_steps : int
        Number of MCMC steps per walker.
    n_params : int
        Dimensionality of the parameter space.
    p0 : array, shape (n_walkers, n_params)
        Initial walker positions.
    output_file : str or None
        If given, save chain to HDF5.
    n_jobs : int
        Parallel processes (``1`` = serial).
    progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    chain : array, shape (n_walkers, n_steps, n_params)
    log_prob : array, shape (n_walkers, n_steps)
    """
    import emcee
    from tqdm import tqdm

    started_at = time.time()
    report_every = _progress_interval(n_steps)

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_params, log_prob_fn, pool=pool
            )
            if _use_tqdm_bar(progress):
                pbar = tqdm(
                    total=n_steps,
                    desc="emcee",
                    unit="step",
                    ncols=80,
                    position=0,
                    leave=False,
                    bar_format="{desc} | {percentage:.0f}% {bar} {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
                )
            else:
                pbar = None

            for step, _ in enumerate(sampler.sample(p0, iterations=n_steps), start=1):
                if step == 1 or step % report_every == 0 or step == n_steps:
                    _print_step_progress(step, n_steps, started_at, "emcee")
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob_fn)
        if _use_tqdm_bar(progress):
            pbar = tqdm(
                total=n_steps,
                desc="emcee",
                unit="step",
                ncols=80,
                position=0,
                leave=False,
                bar_format="{desc} | {percentage:.0f}% {bar} {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
            )
        else:
            pbar = None

        for step, _ in enumerate(sampler.sample(p0, iterations=n_steps), start=1):
            if step == 1 or step % report_every == 0 or step == n_steps:
                _print_step_progress(step, n_steps, started_at, "emcee")
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    # emcee returns (n_steps, n_walkers, n_params) — transpose to our convention
    chain = sampler.get_chain().transpose(1, 0, 2)
    log_prob = sampler.get_log_prob().T

    if output_file:
        _save_chain_hdf5(output_file, chain, log_prob)

    return chain, log_prob


# ---------------------------------------------------------------------------
# ptemcee
# ---------------------------------------------------------------------------

def run_ptemcee(
    log_likelihood_fn: Callable[[np.ndarray], float],
    log_prior_fn: Callable[[np.ndarray], float],
    n_walkers: int,
    n_steps: int,
    n_params: int,
    p0: np.ndarray,
    n_temps: int = 10,
    output_file: str | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the ptemcee parallel-tempering sampler with progress tracking.

    Returns only the samples at the lowest temperature (β = 1).

    Parameters
    ----------
    log_likelihood_fn : callable
        Log-likelihood function only (not including the prior).
    log_prior_fn : callable
        Log-prior function only.
    n_walkers : int
        Walkers per temperature.
    n_steps : int
        Number of steps.
    n_params : int
        Number of parameters.
    p0 : array, shape (n_temps, n_walkers, n_params)
        Initial positions for all temperatures.
    n_temps : int
        Number of temperatures.
    output_file : str or None
    n_jobs : int
        Parallel threads (workers for multiprocessing).

    Returns
    -------
    chain : array, shape (n_walkers, n_steps, n_params)  — β=1 only
    log_prob : array, shape (n_walkers, n_steps)
    """
    _ensure_numpy_legacy_aliases()
    import ptemcee
    from tqdm import tqdm

    started_at = time.time()
    report_every = _progress_interval(n_steps)

    print(f"[ptemcee] starting {n_steps} steps", flush=True)

    sampler = ptemcee.Sampler(
        nwalkers=n_walkers,
        dim=n_params,
        logl=log_likelihood_fn,
        logp=log_prior_fn,
        ntemps=n_temps,
        threads=n_jobs,
    )

    if _use_tqdm_bar(True):
        pbar = tqdm(
            total=n_steps,
            desc="sampling",
            unit="step",
            ncols=80,
            position=0,
            leave=False,
            bar_format="{desc} | {percentage:.0f}% {bar} {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        )
        for step, _ in enumerate(sampler.sample(p0, iterations=n_steps), start=1):
            if step == 1 or step % report_every == 0 or step == n_steps:
                _print_step_progress(step, n_steps, started_at, "ptemcee")
            pbar.update(1)
        pbar.close()
    else:
        for step, _ in enumerate(sampler.sample(p0, iterations=n_steps), start=1):
            if step == 1 or step % report_every == 0 or step == n_steps:
                _print_step_progress(step, n_steps, started_at, "ptemcee")

    chain = sampler.chain[0]           # (n_walkers, n_steps, n_params)
    log_prob = sampler.logprobability[0]  # (n_walkers, n_steps)

    if output_file:
        _save_chain_hdf5(output_file, chain, log_prob)

    return chain, log_prob


# ---------------------------------------------------------------------------
# Chain I/O
# ---------------------------------------------------------------------------

def _save_chain_hdf5(
    path: str,
    chain: np.ndarray,
    log_prob: np.ndarray,
) -> None:
    """Save MCMC chain to HDF5."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("chain", data=chain)
        f.create_dataset("log_prob", data=log_prob)


def load_chain_hdf5(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load MCMC chain from HDF5.

    Returns
    -------
    chain : array, shape (n_walkers, n_steps, n_params)
    log_prob : array, shape (n_walkers, n_steps)
    """
    with h5py.File(path, "r") as f:
        chain = f["chain"][:]
        log_prob = f["log_prob"][:]
    return chain, log_prob


def thin_chain(
    chain: np.ndarray,
    burnin: int,
    thin: int = 1,
) -> np.ndarray:
    """Discard burn-in steps and thin a chain.

    Parameters
    ----------
    chain : array, shape (n_walkers, n_steps, n_params)
    burnin : int
        Number of initial steps to discard.
    thin : int
        Keep every ``thin``-th step after burn-in.

    Returns
    -------
    flat_chain : array, shape (n_surviving_samples, n_params)
    """
    return chain[:, burnin::thin, :].reshape(-1, chain.shape[-1])
