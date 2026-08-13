"""hic-bayes: Bayesian parameter estimation for heavy-ion collision models.

Workflow
--------
1. Generate a design via :mod:`hic_bayes.design`
2. Compress model outputs via :mod:`hic_bayes.pca`
3. Train a GP emulator via :mod:`hic_bayes.emulator` (or :mod:`hic_bayes.emulator_mf`)
4. Define a prior via :mod:`hic_bayes.prior`
5. Run MCMC via :mod:`hic_bayes.mcmc`
6. Analyse the posterior via :mod:`hic_bayes.posterior`
7. Evaluate calibration via :mod:`hic_bayes.calibration`
8. Sensitivity analysis via :mod:`hic_bayes.sensitivity`
"""

from .design import lhs_design, sobol_design, load_design
from .pca import fit_pca, PCAResult
from .emulator import Emulator
from .likelihood import mvn_loglike, build_covariance
from .prior import UniformPrior
from .mcmc import run_emcee, run_ptemcee, thin_chain, load_chain_hdf5
from .posterior import (
    map_estimate,
    credible_interval,
    posterior_summary,
    corner_plot,
)
from .calibration import (
    mahalanobis_distance,
    energy_score,
    log_density_knn,
    calibration_report,
)
from .sensitivity import sobol_sensitivity, local_sensitivity
from .plots import (
    plot_variance_explained,
    plot_emulator_validation,
    plot_loo_validation,
    plot_trace,
    plot_calibration_report,
    plot_predictive_check,
    plot_sensitivity,
    plot_local_sensitivity,
)

__version__ = "0.1.0"

__all__ = [
    # design
    "lhs_design", "sobol_design", "load_design",
    # pca
    "fit_pca", "PCAResult",
    # emulator
    "Emulator",
    # likelihood
    "mvn_loglike", "build_covariance",
    # prior
    "UniformPrior",
    # mcmc
    "run_emcee", "run_ptemcee", "thin_chain", "load_chain_hdf5",
    # posterior
    "map_estimate", "credible_interval", "posterior_summary", "corner_plot",
    # calibration
    "mahalanobis_distance", "energy_score", "log_density_knn", "calibration_report",
    # sensitivity
    "sobol_sensitivity", "local_sensitivity",
    # plots
    "plot_variance_explained", "plot_emulator_validation", "plot_loo_validation",
    "plot_trace", "plot_calibration_report",
    "plot_predictive_check", "plot_sensitivity", "plot_local_sensitivity",
]
