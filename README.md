# hic-bayes

Bayesian parameter estimation for heavy-ion collision models.

A unified, installable Python library consolidating workflows from multiple analysis codebases (Bernhard-hic-param-est, Thiago\_statistical\_tools, Thiago\_Universality\_Github, Giuliano-Bayesian-Analysis, Andi-qgp-parameter-inference, Gabriel\_TLProject).

---

## Quickstart

```bash
# 1. Clone and install (editable mode)
cd hic-bayes
pip install -e ".[dev]"

# 2. Generate synthetic toy data to test the pipeline
python scripts/generate_toy_data.py

# 3. Run the full analysis from a config file
python -m pipelines.run_analysis config/example_PbPb2760.yaml

# 4. Or step through the tutorial notebooks
jupyter lab notebooks/
```

---

## Repository Structure

```
hic-bayes/
├── hic_bayes/            # Core library — import this in notebooks/scripts
│   ├── design.py         # LHS / Sobol design generation
│   ├── pca.py            # SVD-PCA compression of model outputs
│   ├── emulator.py       # Single-fidelity GP emulator (scikit-learn)
│   ├── emulator_mf.py    # Multi-fidelity GP emulator (emukit) [optional]
│   ├── likelihood.py     # Multivariate Gaussian log-likelihood (Cholesky)
│   ├── prior.py          # Uniform prior distributions
│   ├── mcmc.py           # emcee and ptemcee MCMC wrappers
│   ├── posterior.py      # MAP, credible intervals, corner plots
│   ├── calibration.py    # Mahalanobis, Energy Score, KNN log-density
│   └── plots.py          # Shared plotting utilities
├── config/
│   └── example_PbPb2760.yaml   # Template — one per analysis
├── pipelines/
│   ├── run_analysis.py         # CLI: standard single-fidelity analysis
│   └── run_tl_analysis.py      # CLI: multi-fidelity transfer-learning analysis
├── notebooks/
│   ├── 01_design.ipynb
│   ├── 02_emulator.ipynb
│   ├── 03_mcmc.ipynb
│   ├── 04_posterior.ipynb
│   └── 05_transfer_learning.ipynb
├── scripts/
│   └── generate_toy_data.py    # Create synthetic data for testing
├── tests/
│   ├── test_pca.py
│   ├── test_emulator.py
│   └── test_likelihood.py
└── pyproject.toml
```

---

## Standard Workflow

```
Design (LHS/Sobol)
      ↓
Physics simulations (run externally)
      ↓
SVD-PCA  →  PC scores
      ↓
GP Emulator training (one GP per PC)
      ↓
MCMC (emcee or ptemcee)
      ↓
Posterior analysis + calibration
```

### In Python

```python
import hic_bayes as hb
import numpy as np

# 1. Fit PCA
pca = hb.fit_pca(simulation_array, n_components=5)
Z_train = pca.transform(simulation_array)

# 2. Train emulator
em = hb.Emulator(n_restarts=4)
em.fit(design_array, Z_train, param_ranges)

# 3. Define log-probability
prior = hb.UniformPrior.from_config(config["parameters"])
y_exp_pc = pca.transform(y_exp.reshape(1, -1)).ravel()
exp_var_pc = pca.transform(y_err.reshape(1, -1)).ravel() ** 2

def log_prob(theta):
    lp = prior.log_prob(theta)
    if not np.isfinite(lp):
        return -np.inf
    y_pred, y_std = em.predict(theta.reshape(1, -1), return_std=True)
    cov = hb.build_covariance(y_std.ravel()**2, exp_var_pc)
    return lp + hb.mvn_loglike(y_exp_pc - y_pred.ravel(), cov)

# 4. Run MCMC
p0 = prior.sample(500)
chain, log_prob_arr = hb.run_emcee(log_prob, 500, 5000, prior.n_params, p0)

# 5. Analyse posterior
flat_chain = hb.thin_chain(chain, burnin=2000, thin=5)
fig = hb.corner_plot(flat_chain, param_labels)
```

---

## Multi-Fidelity (Transfer Learning)

For analyses with few expensive high-fidelity runs and many cheap low-fidelity runs:

```bash
pip install hic-bayes[multifidelity]   # installs emukit + GPy
python -m pipelines.run_tl_analysis config/example_tl.yaml
```

See `notebooks/05_transfer_learning.ipynb` for an interactive walkthrough.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dependencies

| Package | Role |
|---|---|
| scikit-learn | GP emulator, StandardScaler |
| emcee / ptemcee | MCMC samplers |
| scipy | Cholesky likelihood, design generation |
| corner | Corner plots |
| h5py | Chain file I/O |
| emukit + GPy | Multi-fidelity GP (optional) |

---

## Unique Capabilities vs Legacy Codebases

| Feature | Source repo |
|---|---|
| SVD-PCA + sklearn GP | All repos |
| ptemcee parallel-tempering | Bernhard, Gabriel |
| Multi-fidelity GP (TL) | Gabriel_TLProject |
| Multi-system support via YAML | Andi-qgp-parameter-inference |
| Mahalanobis / Energy Score calibration | Gabriel_TLProject |
| Installable package + CLI | New |



## Logging

[Setup info]
  ↓
[Dados carregados]
  ↓
[PCA fitting]
  ↓
[Emulators loaded/trained]
  ↓
╔════════════════════════════════════╗
║ MCMC CONFIG + ESTIMATIVA DE TEMPO  │  ← NOVO
║  • 4.4M likelihood evals expected
║  • ~1750 evals/sec (com 20 threads)
║  • ~41 minutos esperado
╚════════════════════════════════════╝
  ↓
╔════════════════════════════════════╗
║ BARRA DE PROGRESSO (tqdm)          │  ← NOVO
║ ptemcee sampling: 47%|██▌| 517/1100
╚════════════════════════════════════╝
  ↓
╔════════════════════════════════════╗
║ RESULTADO REAL vs ESTIMADO         │  ← NOVO
║  • 41m 35s (real)
║  • 41m 8s (estimado) → -1.1% erro
╚════════════════════════════════════╝
  ↓
[Posterior summary + samples]
  ↓
[Corner plot salvo]


# para definir o run + submeter apenas 1 task do estudo:
config/AtlasPbPb2760/emulator_params/submit_emulator_params.sh \
  --run-id run_001 \
  --only scan08_standard

  

