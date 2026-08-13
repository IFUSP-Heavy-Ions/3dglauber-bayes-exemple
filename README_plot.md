# Plot utilities

Execute os comandos abaixo a partir da raiz do repositorio `hic-bayes`.

## Script principal

O script baixo nivel continua sendo config-driven:

```bash
python3 hic_bayes/plot.py \
  --config config/AtlasPbPb2760/baseline/corner_plot_config.json \
  --mode all
```

Modos disponiveis:

- `--mode individual`: gera apenas plots individuais definidos no config.
- `--mode comparison`: gera apenas plots comparativos definidos no config.
- `--mode all`: gera ambos.

## Wrapper simples

Para uso normal, chame `plot_compare.py` passando:

```bash
python3 hic_bayes/plot_compare.py <experimento>/<estudo> <run> [outro_run] [--compare] [--all]
```

O wrapper procura automaticamente:

```bash
config/<experimento>/<estudo>/corner_plot_config.json
```

Tambem aceita o caminho completo, por exemplo:

```bash
python3 hic_bayes/plot_compare.py config/AtlasPbPb2760/baseline run_002
```

## Um Run

Com um unico run, o wrapper gera os jobs de `individual_plots` definidos no
`corner_plot_config.json`:

```bash
python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002
```

Output padrao:

```bash
config/AtlasPbPb2760/baseline/plots/run_002/individual/
```

## Dois Runs

Com dois runs sem flags, o wrapper compara cada plot individual de um run
contra o mesmo plot individual do outro run:

```bash
python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002 run_003
```

Output padrao:

```bash
config/AtlasPbPb2760/baseline/plots/run_002_vs_run_003/comparisons/
```

## Comparacoes Do Config

Com `--compare`, o wrapper usa os jobs pareados definidos em
`comparison_plots.jobs`, isto e, jobs com `left_experiment` e
`right_experiment`.

Dentro de um run:

```bash
python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002 --compare
```

Cruza a esquerda do primeiro run com a direita do segundo run:

```bash
python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002 run_003 --compare
```

Por exemplo, um job configurado como `baseline_Grad_diag` vs
`baseline_Grad_scaled_diag` vira:

```bash
run_002/baseline_Grad_diag vs run_003/baseline_Grad_scaled_diag
```

## Todos No Mesmo Grafico

Com `--all`, o wrapper usa os jobs multi-experimento definidos em
`comparison_plots.jobs` com `"experiments": [...]`. Se nao houver um job desse
tipo no config, ele cria um com todos os experimentos definidos em
`experiments`.

```bash
python3 hic_bayes/plot_compare.py AtlasPbPb2760/emulator_params run_002 --all
```

Output padrao:

```bash
config/AtlasPbPb2760/emulator_params/plots/run_002/comparisons/
```

## Dry Run

Para ver o JSON temporario que sera enviado ao `plot.py`, sem gerar plots:

```bash
python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002 run_003 --dry-run
```

## Config

As configuracoes de parametros, grupos, cores, labels, jobs individuais e jobs
comparativos continuam no arquivo:

```bash
config/<experimento>/<estudo>/corner_plot_config.json
```
