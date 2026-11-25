"""Sampler for the global RBF study (module-level class so multiprocessing can pickle it)."""
import json
from pathlib import Path

from bayes_mcmc_cs_model import Chain as GlobalCSChain
from configurations import obs_cent_list, workdir
from experimental_global_covariance import global_rbf_covariance
from experimental_sector_covariance import covariance_signature


class Chain(GlobalCSChain):
    def __init__(self, lengths, acceptance, acceptance_path, path=None):
        self.rbf_lengths = dict(lengths)
        self.acceptance = acceptance
        self.acceptance_path = str(acceptance_path)
        super().__init__(path=path or workdir / "mcmc" / "02_02_global_rbf" / "pending.hdf", cs_rho=1.0)
        self.covariance_signature = covariance_signature(self._expt_cov, self._slices, obs_cent_list)
        if path is None:
            self.path = (workdir / "mcmc" / "02_02_global_rbf" /
                         f"{Path(self.filename).stem}-rbf-global-{self.covariance_signature[:12]}.hdf")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _build_experimental_covariance(self, system, experimental_std):
        return global_rbf_covariance(system, self._slices[system], experimental_std, obs_cent_list,
                                     self.rbf_lengths, self.acceptance)

    def _covariance_metadata(self):
        return {
            "experimental_covariance_model": "rbf",
            "experimental_covariance_scope": "global_system",
            "experimental_covariance_signature": self.covariance_signature,
            "experimental_rbf_kernel": "prod_axes exp(-((x_i-x_j)/L_axis)^2); axes pt, eta, centrality",
            "experimental_rbf_coordinates": "common physical space; differential axis = bin center; "
                                            "centrality = class center; integrated eta or pT = midpoint "
                                            "of the acceptance window",
            "experimental_rbf_lengths": json.dumps(self.rbf_lengths, sort_keys=True),
            "experimental_rbf_cross_observable_factor": "neutral (1)",
            "experimental_rbf_acceptance": json.dumps(self.acceptance, sort_keys=True),
        }
