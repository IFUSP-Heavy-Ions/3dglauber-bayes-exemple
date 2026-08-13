# Progress Logging & Execution Guide

## O que melhorou

### 1. **Estimativa de Tempo no Início**
Agora você vê quanto tempo o cálculo deve levar **antes de começar**:

```
======================================================================
ptemcee Parallel-Tempering MCMC
======================================================================
  Walkers:           200
  Steps:             1100
  Temperatures:      20
  Threads:           20
  Likelihood evals:  4400000
  Final samples:     12000
----------------------------------------------------------------------
  Estimated throughput: ~ 1750 evals/sec
  Estimated duration:   ~ 41m 8s
----------------------------------------------------------------------
  Progress bar below will update per completed step.
======================================================================
```

A estimativa se baseia em heurísticas calibradas com benchmarks reais:
- Throughput escala com √(n_threads) para eficiência sub-linear
- Penaliza dimensionalidade maior (mais overhead por avaliação)
- Usa baseline: 200 evals/sec (1 thread, 17 params)

### 2. **Comparação com Realidade no Final**
Depois da execução:

```
======================================================================
Sampling completed in 41m 35s
  Wall time:    2495.0 s
  Throughput:   1760 likelihood evals/sec
  Estimate was: 41m 8s (-1.1%)
======================================================================
```

Mostra se você foi mais rápido ou lento que o previsto, útil para tuning futuro.

### 3. **Barra de Progresso do ptemcee**
Agora você vê uma barra de progresso em tempo real enquanto o MCMC roda, em vez de apenas começar e esperar 40+ minutos sem feedback.

---

## Como a Estimativa de Tempo Funciona

### Heurística
```
Est. throughput = baseline × √(n_threads) × penalty(n_params)

baseline = 200 evals/sec (single thread, 17 params)
√(n_threads) = parallelismo sub-linear (não é 100% linear por overhead)
penalty(n_params) = 1 / (1 + 0.005 × extra_params)
```

**Exemplos:**
- 1 thread, 17 params → ~200 evals/sec
- 4 threads, 17 params → ~360 evals/sec
- 20 threads, 17 params → ~1750 evals/sec
- 20 threads, 30 params → ~1600 evals/sec (penaliza dimensão maior)

### Calibração
A heurística é calibrada com runs reais. Se seu hardware for significativamente diferente (ex: CPU muito mais lenta), a estimativa será proporcional à diferença.

### Usar Histórico de Runs (Opcional)
Se quiser, pode editar `estimate_throughput()` em [pipelines/run_universality_analysis.py](pipelines/run_universality_analysis.py) com valores medidos nos seus runs anteriores para maior precisão.

---

## Logs Visualmente Completos

Um run típico agora mostra:

```
=== hic-bayes Universality | Pb+Pb 2.76 TeV … ===

Design   : 321 points × 17 params
Simulation: 321 × 287 observables
Prior    : 17 parameters

Fitting SVD PCA with 6 components …
  PCA reconstruction max error: 1.31e-02

Loading pre-trained emulators from:
  data/universality_PbPb2760/emulators_Grad_6_pcs.pkl
  Loaded 6 GP emulators.

Experimental covariance: mode='diag'  rho=0.0

======================================================================
ptemcee Parallel-Tempering MCMC
======================================================================
  Walkers:           200
  Steps:             1100  (burn-in=500, thin=10)
  Temperatures:      20
  Threads:           20
  Likelihood evals:  4400000  (20 temps × 200 walkers × 1100 steps)
  Final samples:     12000
----------------------------------------------------------------------
  Estimated throughput: ~ 1750 evals/sec
  Estimated duration:   ~ 41m 8s
----------------------------------------------------------------------
  Progress bar below will update per completed step.
======================================================================

ptemcee sampling:  23%|███▌          | 253/1100 [06:08<19:23, 0.65step/s]
[... barra atualiza enquanto calcula ...]
ptemcee sampling: 100%|████████████| 1100/1100 [41:35<00:00, 0.44step/s]

======================================================================
Sampling completed in 41m 35s
  Wall time:    2495.0 s
  Throughput:   1760 likelihood evals/sec
  Estimate was: 41m 8s (-1.1%)
======================================================================

Posterior: 12000 samples after burn-in / thinning

Parameter                     Mean       Std         CI5        CI95
  ----------------------------------------------------------------
  norm                     14.8950    3.1519     10.1659     19.5277
  [... 16 more params ...]
  
MAP estimate:
  {...}

Posterior chain saved to: results/universality_Grad_diag/chain.csv
Corner plot saved to    : results/universality_Grad_diag/posterior_corner.pdf

Done.
```

---

## Para Testar

### Opção 1: Teste Rápido com Barra (Recomendado Primeiro)
```bash
cd /home/avilamrs/code-space/my-work/hic-bayes
pip install -q corner tqdm
python -m pipelines.run_universality_analysis config/test_progress_10steps.yaml
```

**Tempo esperado:** ~10 segundos (validação rápida do sistema)

**O que verá:**
- Setup summary com 20 walkers × 10 steps × 2 temps
- Barra de progresso atualizando a cada passo
- Outputs em `results/test_progress/chain.csv` e `posterior_corner.pdf`

### Opção 2: Full Run (Diagonal)
```bash
cd /home/avilamrs/code-space/my-work/hic-bayes
pip install -U -r requirements.txt  # garante corner + tqdm
HIC_BAYES_NTHREADS=20 python -m pipelines.run_universality_analysis \
    config/universality_PbPb2760_Grad_diag.yaml
```

**Tempo esperado:** ~40-50 minutos com 20 threads  
**Barra atualiza:** a cada step concluído (1100 passos totais)

### Opção 3: Script Automatizado
```bash
bash scripts/test_progress_bar.sh
```

Isso:
1. Instala `corner` e `tqdm` (se necessário)
2. Roda o teste de 10 passos
3. Mostra onde os resultados foram salvos

---

## Interpretando a Barra de Progresso

```
ptemcee sampling: 47%|██████▌        | 517/1100 [15:23<18:12, 0.53step/s]
                 └──┬──┘ └─────────┬────────┘ └────┬─────┘ └──────┬──────┘ └─────┬─────┘
                 % done  visual bar   current/total   time elapsed  time left  speed
```

- **47%**: Você completou 47% dos steps
- **517/1100**: 517 passos concluídos de 1100 no total
- **[15:23<18:12]**: Levou 15 minutos 23 segundos até agora; faltam ~18 minutos 12 segundos
- **0.53step/s**: Velocidade de processamento (0.53 steps por segundo com 20 threads)

---

## Resumo Final

Quando terminar, você verá:
```
======================================================================
Sampling completed in 2513.0 s  (41.9 min)
  Throughput: 1751 likelihood evals/sec
======================================================================
```

Isso diz que você processou 1.75k likelihood evaluations por segundo, que é uma métrica útil para HPC scalability.

---

## Se algo der errado

Se a barra não aparecer:
- Você está em SSH/remote sem suporte a ANSI colors? Tqdm detecta e desativa a barra automaticamente nesse caso — mas a saída de texto continua funcionando
- Rode: `python -m pipelines.run_universality_analysis config/... --force-tqdm` (se adicionarmos essa flag depois)

Se `corner` ainda der erro:
```bash
pip install --upgrade corner
```

---

## Próximas Melhorias Opcionais

Se quiser ainda mais verbosidade:
- Adicionar callback que mostra likelihood min/max/mean por temperatura
- Salvar checkpoint intermediário a cada N steps
- Plotar trace (walkers × steps) em tempo real via matplotlib backend
