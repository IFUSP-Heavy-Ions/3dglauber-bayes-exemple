# 02_14 — covariância do PHOBOS aprendida de dados (degrau 3)

Primeira cadeia em que a estrutura de correlação experimental é **aprendida de dados reais**, e não suposta.

- **Fonte:** `study_nongaussian_systematics/fatia2a` (comparações entre detectores PHOBOS, PHENIX, sPHENIX e BRAHMS em η≈0;
  cópia local em `IFUSP-Heavy-Ions/3dglauber-bayes-exemple/...`).
- **Bloco alterado:** só o Au-Au PHOBOS dN/dη (11 classes × 54 η = 594 bins). Todos os outros bins e o sistema d-Au continuam diagonais (= baseline).
- **A diagonal não muda:** cada bin mantém a variância da calibração. Só a correlação é aprendida.
- **Forma (em log):** `Σ = a_c a_c' (offset comum, ρ=1, plano em η) + s² g_c g_c' (inclinação em centralidade) + diag(resto)`.
  - `u = 0.0148`: parte classe a classe não correlacionada do PHOBOS (three-cornered hat, 1.48%).
  - `s = 0.025` por 100% de centralidade: rms das inclinações a 200 GeV envolvendo o PHOBOS, dividido por √2.
  - `a_c² = min_η σ² − u² − s² g_c²`.
- **Hipóteses declaradas:**
  - offset plano em η (a fatia 2b não foi feita);
  - variância extra em η forward não correlacionada (R3–R5 não feitas);
  - variância da calibração mantida (possível 90% C.L. usado como 1σ: ponto em aberto).
- **Checagem (`--check-covariance`):** correlação fora da diagonal no bloco: média 0.51, mínimo 0.07 (forward), máximo 0.94 (centro);
  positiva definida; diagonal igual à da calibração. Assinatura 7ec3735f96a8.
- **Código:** `experimental_learned_covariance.py`, `bayes_mcmc_learned_phobos.py`, `bayes_mcmc_learned_phobos_chain.py` (raiz do projeto).
- **Rodar:** `cd mcmc/02_14_learned_phobos_normalization && mkdir -p logs && condor_submit condor_learned_phobos.sub`
- **Saída:** `chain-idf-0-learned-phobos-<assinatura12>.hdf`.
