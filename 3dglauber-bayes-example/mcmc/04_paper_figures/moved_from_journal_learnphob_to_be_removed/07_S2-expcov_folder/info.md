# 07_S2-expcov

- Figura nossa: covariância (em cima) e correlação (embaixo) experimentais entre os 917 bins
- Especificação: `../../DESIGN_SYSTEM.md`, seção 2 · 07
- Modelos: baseline (diag) | CS sector | learned PHOBOS cov. (job 5079)
- Matriz aprendida reconstruída com `experimental_learned_covariance.py` (u = 0.0148, s = 0.025) e os `likelihood_std` de `bin_index.csv`
- Script: `fig_expcorr.py --proposal covmom_learned`. Script de execução: `mcmc/04_paper_figures/run_5078_5079.sh` (no servidor, venv `/storage/avilamrs/code-space/my-work/jetscape-emulator/.venv`); saída em `mcmc/04_paper_figures/out/{global_rbf,covmom_learned}/` e copiada para cá.
