"""Common sampler for same-observable CS and physical-coordinate RBF studies."""
import json
from pathlib import Path
from bayes_mcmc_cs_model import Chain as GlobalCSChain
from configurations import workdir, obs_cent_list, system_strs
from experimental_sector_covariance import (
    sector_covariance, covariance_signature, validate_overrides, validate_regions,
)


class Chain(GlobalCSChain):
    def __init__(self, model, path=None, cs_rho=1., lengths=None, overrides=None, regions=None):
        if model not in ('cs', 'rbf'):
            raise ValueError('Model must be cs or rbf.')
        if regions is None:
            regions = json.loads(Path(__file__).with_name('rapidity_regions.json').read_text())
        self.rapidity_regions = regions
        validate_regions(self.rapidity_regions, {s: obs_cent_list[s] for s in system_strs})
        self.sector_model = model
        self.rbf_lengths = dict(lengths or {})
        self.rbf_overrides = overrides or {}
        validate_overrides(self.rbf_overrides, {s: obs_cent_list[s] for s in system_strs})
        # Sector construction dispatches through the parent hook during init.
        # Path is finalized only after the exact covariance signature is known.
        super().__init__(path=path or workdir / 'mcmc' / self.filename, cs_rho=cs_rho)
        self.covariance_signature = covariance_signature(self._expt_cov, self._slices, obs_cent_list)
        if path is None:
            suffix = ((f'cs-observable-region-rho-{self.cs_rho:g}-{self.covariance_signature[:12]}' if self.rapidity_regions else f'cs-observable-rho-{self.cs_rho:g}') if model == 'cs'
                      else f'rbf-observable-{self.covariance_signature[:12]}')
            self.path = workdir / 'mcmc' / f'{Path(self.filename).stem}-{suffix}.hdf'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _build_experimental_covariance(self, system, experimental_std):
        return sector_covariance(system, self._slices[system], experimental_std,
                                 obs_cent_list, self.sector_model, rho=self.cs_rho,
                                 lengths=self.rbf_lengths, overrides=self.rbf_overrides, regions=self.rapidity_regions)

    def _covariance_metadata(self):
        metadata = {
            'experimental_covariance_model': ('compound_symmetry' if self.sector_model == 'cs' else 'rbf'),
            'experimental_covariance_scope': 'observable_rapidity_region' if self.rapidity_regions else 'observable',
            'experimental_rapidity_regions': json.dumps(self.rapidity_regions,sort_keys=True),
            'experimental_covariance_signature': self.covariance_signature,
            'experimental_sector_definition': 'system + exact observable key + configured rapidity region',
        }
        if self.sector_model == 'cs':
            metadata.update(experimental_cs_scope=metadata['experimental_covariance_scope'], experimental_cs_rho=self.cs_rho)
        else:
            metadata.update(
                experimental_rbf_kernel='exp(-((x_i-x_j)/length)^2)',
                experimental_rbf_coordinates='physical bin centers; pt GeV/c; eta dimensionless; centrality percentage points',
                experimental_rbf_lengths=json.dumps(self.rbf_lengths, sort_keys=True),
                experimental_rbf_overrides=json.dumps(self.rbf_overrides, sort_keys=True),
            )
        return metadata
