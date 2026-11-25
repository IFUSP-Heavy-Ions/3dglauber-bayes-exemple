"""Sampler for the learned PHOBOS normalization study (module-level class so multiprocessing can pickle it)."""
from pathlib import Path

from bayes_mcmc_cs_model import Chain as GlobalCSChain
from configurations import obs_cent_list, workdir
from experimental_learned_covariance import learned_phobos_covariance
from experimental_sector_covariance import covariance_signature

OUTDIR = workdir / "mcmc" / "02_14_learned_phobos_normalization"


class Chain(GlobalCSChain):
    def __init__(self, u, slope, path=None):
        self.learned_u = float(u)
        self.learned_slope = float(slope)
        super().__init__(path=path or OUTDIR / "pending.hdf", cs_rho=1.0)
        self.covariance_signature = covariance_signature(self._expt_cov, self._slices, obs_cent_list)
        if path is None:
            self.path = OUTDIR / f"{Path(self.filename).stem}-learned-phobos-{self.covariance_signature[:12]}.hdf"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _build_experimental_covariance(self, system, experimental_std):
        return learned_phobos_covariance(system, self._slices[system], experimental_std,
                                         u=self.learned_u, slope=self.learned_slope)

    def _covariance_metadata(self):
        return {
            "experimental_covariance_model": "learned_phobos_normalization",
            "experimental_covariance_scope": "Au-Au PHOBOS dNdeta block (594 bins); every other bin diagonal",
            "experimental_covariance_signature": self.covariance_signature,
            "experimental_learned_source": "study_nongaussian_systematics/fatia2a (cross-detector comparisons)",
            "experimental_learned_u": self.learned_u,
            "experimental_learned_slope_per_100pct": self.learned_slope,
            "experimental_learned_form": "a_c a_c' (offset, flat in eta) + s^2 g_c g_c' (centrality slope) "
                                         "+ diag(rest); diagonal = calibration variance",
        }
