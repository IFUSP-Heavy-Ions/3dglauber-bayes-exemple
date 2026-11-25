#!/usr/bin/env python3
"""Baseline likelihood + non-Gaussian systematic nuisance parameters for STAR v2(eta).

Everything is identical to the baseline (diagonal experimental covariance, same
emulators, same priors) EXCEPT the block v22_eta_cen_20_70_STAR of Au-Au-200:

    y_exp = mu(theta) + sum_a lambda_a vhat_a + eps,    eps ~ N(0, diag(sigma_stat^2))
    lambda_a = s_a (G_a - k_a)/sqrt(k_a),  G_a ~ Gamma(k_a),  k_a = 4/skew_a^2

(vhat_a, s_a, skew_a, sigma_stat) come from the Tensor PCA of the replicas
(study_nongaussian_systematics/data/inferred_sources.npz). sigma_stat^2 = sigma_total^2
- sum_a s_a^2 vhat_a^2, so the systematic part is not counted twice: it lives in the
lambda_a, which are sampled jointly with theta (Monte Carlo marginalization over the
Gamma prior). Marginalizing lambda gives back the baseline variance in every bin.

Usage (from 3dglauber-bayes-example):
    python bayes_mcmc_nuisance.py --check                       # timing + sanity, no MCMC
    python bayes_mcmc_nuisance.py 20000 --nwalkers 200 --nburnsteps 20000 --ncpus 16
"""
import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import stats

# Emulators saved by running emulator.py as a script resolve these through __main__.
from configurations import do_transform_design, transform_design  # noqa: F401

SYSTEM = "Au-Au-200"
OBS = "v22_eta_cen_20_70_STAR"
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "study_nongaussian_systematics" / "data" / "inferred_sources.npz"
NUIS_LABELS = ["lambda_norm", "lambda_asym", "lambda_edge"]


_SHARED_CHAIN = None


def _shared_chain():
    """Workers are forked after the chain is built; they unpickle the chain as this global."""
    return _SHARED_CHAIN


def gamma_logprior(lam, s, k):
    """log density of lambda = s (G-k)/sqrt(k), G ~ Gamma(k); -inf outside the support."""
    g = lam * np.sqrt(k) / s + k
    return np.sum(stats.gamma.logpdf(g, k) + np.log(np.sqrt(k) / s), axis=-1)


def make_chain_class():
    from bayes_mcmc_cs_model import Chain as GlobalCSChain, mvn_loglike
    from configurations import workdir

    class Chain(GlobalCSChain):
        def __init__(self, path=None, sources=DEFAULT_SOURCES):
            src = np.load(sources)
            self.sources_file = str(sources)
            self.sources_sha256 = hashlib.sha256(Path(sources).read_bytes()).hexdigest()
            self.v_hat = np.asarray(src["v_hat"], float)            # (21, 3), unit columns
            self.nuis_s = np.asarray(src["s"], float)               # rms of each lambda_a
            self.nuis_skew = np.asarray(src["skew"], float)
            self.nuis_k = 4.0 / self.nuis_skew**2
            self.sigma_stat = np.asarray(src["sigma_stat"], float)
            self.sigma_total = np.asarray(src["sigma_total"], float)
            # number of sources comes from the npz (original study: 3, detector_split: 4)
            self.nuis_labels = [str(x) for x in src["labels"]] if "labels" in src.files else list(NUIS_LABELS)
            if len(self.nuis_labels) != self.v_hat.shape[1]:
                raise ValueError("labels do not match the number of sources in v_hat")
            if path is None:
                path = workdir / "mcmc" / "02_10_nongaussian_nuisance" / (
                    f"chain-idf-0-nongauss-nuisance-{self.sources_sha256[:12]}.hdf")
            # parent builds the experimental covariance through the hook below (rho unused)
            super().__init__(path=path, cs_rho=0.0)

            self._star = dict(self._slices[SYSTEM])[OBS]
            self.theta_ndim = self.ndim
            k, s = self.nuis_k, self.nuis_s
            lo = -s * np.sqrt(k) * (1 - 1e-9)
            hi = s * (stats.gamma.ppf(1 - 1e-7, k) - k) / np.sqrt(k)
            self.min = np.concatenate([self.min, lo])
            self.max = np.concatenate([self.max, hi])
            self.range = np.array([self.min, self.max]).T
            self.labels = list(self.labels) + self.nuis_labels
            self.ndim = self.theta_ndim + len(s)

        def _build_experimental_covariance(self, system, experimental_std):
            cov = np.diag(experimental_std**2)                        # baseline: diagonal
            if system == SYSTEM:
                slc = dict(self._slices[system])[OBS]
                if not np.allclose(experimental_std[slc], self.sigma_total, rtol=1e-6, atol=0):
                    raise ValueError("inferred_sources sigma_total does not match the experimental data")
                cov[slc, slc] = np.diag(self.sigma_stat**2)          # systematic part -> lambda_a
            return cov

        def _covariance_metadata(self):
            return {
                "experimental_covariance_model": "diagonal_plus_nongaussian_nuisance",
                "experimental_nuisance_observable": f"{SYSTEM}/{OBS}",
                "experimental_nuisance_sources_sha256": self.sources_sha256,
                "experimental_nuisance_labels": json.dumps(self.nuis_labels),
                "experimental_nuisance_s": json.dumps(self.nuis_s.tolist()),
                "experimental_nuisance_skew": json.dumps(self.nuis_skew.tolist()),
            }

        def log_posterior(self, X, extra_std_prior_scale=0.001):
            X = np.array(X, copy=False, ndmin=2)
            lp = np.zeros(X.shape[0])
            inside = np.all((X > self.min) & (X < self.max), axis=1)
            inside &= (X[:, 0] <= X[:, 1]) & (X[:, 1] <= X[:, 2])    # same monotonicity cut as baseline
            lp[~inside] = -np.inf
            if not inside.any():
                return lp
            theta = X[inside, : self.theta_ndim].copy()
            lam = X[inside, self.theta_ndim:]
            pred = self._predict(theta, return_cov=True)
            for sys in self._expt_y:
                nobs = self._expt_y[sys].size
                n = theta.shape[0]
                dY = np.empty((n, nobs))
                cov = np.empty((n, nobs, nobs))
                Y_pred, cov_pred = pred[sys]
                for obs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = Y_pred[obs1] - self._expt_y[sys][slc1]
                    for obs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = cov_pred[obs1, obs2]
                if sys == SYSTEM:
                    dY[:, self._star] += lam @ self.v_hat.T             # residual of y = mu + V lambda + eps
                cov += self._expt_cov[sys]
                lp[inside] += np.array(list(map(mvn_loglike, dY, cov)))
            lp[inside] += gamma_logprior(lam, self.nuis_s, self.nuis_k)
            return lp

    def __reduce__(self):
        # emcee sends self.log_posterior to the pool for every task. Pickling the whole
        # object (experimental covariances, ~5 MB) ~200 times per step made the master
        # process the bottleneck; send a reference to the forked global instead.
        if _SHARED_CHAIN is not self:
            raise RuntimeError("register the chain with set_shared_chain() before sampling")
        return (_shared_chain, ())

    Chain.__reduce__ = __reduce__

    # The sampler pickles bound methods for multiprocessing: expose the class at module
    # level so it is found as <module>.Chain (a function-local class cannot be pickled).
    Chain.__qualname__ = "Chain"
    Chain.__module__ = __name__
    globals()["Chain"] = Chain
    return Chain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nsteps", type=int, nargs="?", help="production steps")
    ap.add_argument("--nwalkers", type=int)
    ap.add_argument("--nburnsteps", type=int)
    ap.add_argument("--ncpus", type=int, default=16)
    ap.add_argument("--status", type=int)
    ap.add_argument("--chain-path", type=Path)
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--check", action="store_true", help="evaluate the posterior at random points and exit")
    args = ap.parse_args()
    if args.nsteps is None and not args.check:
        ap.error("give nsteps or --check")

    Chain = make_chain_class()
    from emulator import _Covariance
    globals()["_Covariance"] = _Covariance                     # legacy dill namespace
    chain = Chain(path=args.chain_path, sources=args.sources)
    global _SHARED_CHAIN
    _SHARED_CHAIN = chain                                       # inherited by the forked workers

    if args.check:
        rng = np.random.default_rng(1)
        X = chain.random_pos(40)
        X[:, :3] = np.sort(X[:, :3], axis=1)
        t0 = time.time()
        lp = np.concatenate([chain.log_posterior(x) for x in X])
        dt = (time.time() - t0) / len(X)
        print(json.dumps(dict(
            ndim=chain.ndim, theta_ndim=chain.theta_ndim, labels=chain.nuis_labels,
            star_slice=[chain._star.start, chain._star.stop], n_bins={k: int(v.size) for k, v in chain._expt_y.items()},
            lambda_min=chain.min[chain.theta_ndim:].tolist(), lambda_max=chain.max[chain.theta_ndim:].tolist(),
            finite=int(np.isfinite(lp).sum()), n=len(lp), ms_per_eval=round(1e3 * dt, 2),
            chain_path=str(chain.path)), indent=1))
        return

    logging.info("chain: %s", chain.path)
    chain.run_mcmc(nsteps=args.nsteps, nwalkers=args.nwalkers, nburnsteps=args.nburnsteps,
                   status=args.status, ncpu=args.ncpus)


if __name__ == "__main__":
    main()
