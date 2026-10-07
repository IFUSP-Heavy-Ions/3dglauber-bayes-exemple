# Configuracoes do exemplo 3D Glauber Bayes

Este documento descreve as configuracoes definidas em [configurations.py](configurations.py), organizadas pelas etapas representadas no fluxograma de `docs/fluxo_execucao.svg`. A cor entre colchetes indica a familia da etapa no fluxograma:

- **Cinza - configuracao geral:** tipos, caminhos, identificacao do modelo e controle do estudo.
- **Azul - dados para o emulador:** sistemas, design, limpeza dos pontos, transformacoes e PCA.
- **Verde - dados experimentais:** fonte dos dados, observaveis e incertezas.
- **Roxo - Bayes/MCMC:** prior, validacao, verossimilhanca e amostragem.
- **Amarelo - saidas:** identificacao das cadeias, parametros MAP e resultados derivados.

As referencias apontam para a linha da atribuicao em `configurations.py`. Quando uma configuracao e construida por varias linhas, a referencia aponta para a linha que inicia a definicao.

## Visao geral do fluxo

1. Configurar tipos, diretorios, sistemas e pontos a remover.
2. Reunir e estruturar os calculos dos sistemas definidos em `systems`.
3. Carregar o design, remover pontos invalidos e preparar a matriz de parametros.
4. Aplicar transformacoes de viscosidade e multiplicidades e treinar o emulador com PCA.
5. Ler os dados experimentais e selecionar os observaveis ativos.
6. Executar a inferencia Bayesiana, normalmente com `bayes_mcmc.py`.

## Cinza - configuracao geral

### Tipos numericos

Controla o tipo, o tamanho e o endianess dos arrays gravados e lidos pelo pipeline. Os tipos explicitos evitam incompatibilidades entre maquinas.

```python
float_t = "<f8"
int_t = "<i8"
complex_t = "<c16"
```

Referencia: [configurations.py:25](configurations.py#L25).

### Semente aleatoria

Fixa a semente usada nas operacoes aleatorias, inclusive na selecao de pontos para cross-validation quando essa opcao e ativada.

```python
np.random.seed(1)
```

Referencia: [configurations.py:30](configurations.py#L30).

### Diretorio de trabalho

Define a raiz dos caminhos de entrada e saida. Usa `WORKDIR` quando a variavel de ambiente existe e `.` caso contrario.

```python
workdir = Path(os.getenv("WORKDIR", "."))
```

Referencia: [configurations.py:33](configurations.py#L33).

### Diretorio do design de producao

Aponta para os arquivos de pontos, intervalos e labels usados no treinamento e na validacao.

```python
design_dir = str(workdir / "production_designs/ProductionMaxPro-Mod")
```

Referencia: [configurations.py:36](configurations.py#L36).

### Numero de modelos por ponto

Define quantas versoes do modelo sao armazenadas para cada ponto do design. O valor atual usa um unico modelo.

```python
number_of_models_per_run = 1
```

Referencia: [configurations.py:76](configurations.py#L76).

### Modelo de correcao viscosa

Seleciona a convencao de correcao `delta-f`. O valor `0` corresponde a `Grad`; `1` corresponderia a `Chapman-Enskog R.T.A.`.

```python
idf = 0
```

Os nomes usados para exibir esse valor sao definidos em [configurations.py:68](configurations.py#L68).

Referencia: [configurations.py:79](configurations.py#L79).

## Azul - DATA e diretorios

### Diretorio dos dados experimentais

Indica onde `bayes_exp.py` procura os dados experimentais ou pseudodados.

```python
dir_obs_exp = "HIC_experimental_data"
```

O valor ativo usa dados experimentais. As alternativas comentadas nas linhas seguintes permitem trocar para pseudodados de pontos de validacao.

Referencia: [configurations.py:39](configurations.py#L39).

### Teste de fechamento

Controla o ramo opcional que substitui os dados experimentais por um ponto simulado de validacao. O fluxo padrao nao executa esse teste.

```python
run_closure = False
closure_val_pt = 2
```

Referencia: [configurations.py:47](configurations.py#L47).

### Rotulo dos sistemas

`systems` e a configuracao principal dos sistemas de colisao. Cada tupla contem projetil, alvo e energia de centro de massa em GeV. O estado atual usa Au-Au-200 e d-Au-200.

```python
systems = [
    ('Au', 'Au', 200),
    ('d', 'Au', 200),
]
```

`system_strs` transforma as tuplas em nomes como `Au-Au-200`, e `num_systems` conta os sistemas ativos.

Referencia: [configurations.py:83](configurations.py#L83).

### Pontos invalidos do design

Agrupa pontos com NaN, eventos inacabados ou caracteristicas estranhas. O conjunto usado no estado atual vem dos pontos problematicos de `idf = 0`; os outros dois conjuntos estao vazios.

```python
nan_sets_by_deltaf = {0: set([65, 78, 207, 322, 210, 229, 256, 316])}
nan_design_pts_set = nan_sets_by_deltaf[idf]
unfinished_events_design_pts_set = set([])
strange_features_design_pts_set = set([])
delete_design_pts_set = nan_design_pts_set.union(
    unfinished_events_design_pts_set.union(strange_features_design_pts_set)
)
delete_design_pts_validation_set = []
```

Referencia: [configurations.py:94](configurations.py#L94).

### Metadados e arquivos por sistema

A classe `systems_setting` monta os caminhos dos arquivos de design, intervalos, labels, eventos e observaveis. Ao atribuir `run_id`, ela tambem cria os caminhos de `Events/main`, `Events/validation`, `Obs/main.dat` e `Obs/validation.dat`.

```python
class systems_setting(dict):
    ...
```

Referencia: [configurations.py:106](configurations.py#L106).

### Configuracao de d-Au-200

Define o run, o numero de pontos de design e validacao, os indices removidos e o numero de componentes principais para d-Au-200.

```python
SystemsInfo["d-Au-200"]["run_id"] = "production_414pts_d_Au_200"
SystemsInfo["d-Au-200"]["run_dir"] = "production_414pts_d_Au_200"
SystemsInfo["d-Au-200"]["n_design"] = 414
SystemsInfo["d-Au-200"]["n_validation"] = 5
SystemsInfo["d-Au-200"]["design_remove_idx"] = [22, 164, 186, 349, 15, 109, 199, 273, 194, 229]
SystemsInfo["d-Au-200"]["npc"] = 10
```

`MAP_obs_file` e `labels` sao definidos como `"None"` neste exemplo.

Referencia: [configurations.py:181](configurations.py#L181).

### Configuracao de Au-Au-200

Define os mesmos metadados para Au-Au-200. Os indices removidos sao especificos desse sistema.

```python
SystemsInfo["Au-Au-200"]["run_id"] = "production_414pts_Au_Au_200"
SystemsInfo["Au-Au-200"]["run_dir"] = "production_414pts_Au_Au_200"
SystemsInfo["Au-Au-200"]["n_design"] = 414
SystemsInfo["Au-Au-200"]["n_validation"] = 5
SystemsInfo["Au-Au-200"]["design_remove_idx"] = [15, 42, 164, 186, 70, 106, 209, 107, 273, 411, 235, 211]
SystemsInfo["Au-Au-200"]["npc"] = 10
```

`MAP_obs_file` e `labels` tambem sao definidos como `"None"`.

Referencia: [configurations.py:192](configurations.py#L192).

## Azul - DESIGN, TRANSFORMING e PCA

### Transformacao do design de viscosidade

Quando ativada, substitui os parametros de viscosidade por valores discretizados de `eta/s(T)` e `zeta/s(T)` antes do treinamento do emulador. O valor atual esta ligado.

```python
do_transform_design = True
```

Referencia: [configurations.py:310](configurations.py#L310).

### Parametros mantidos e parametros viscosos

O arquivo `design_points_main_AuAu-200.dat` possui uma coluna `idx` mais 20 parametros fisicos. A coluna `idx` nao e um parametro: `load_design` a remove com `design.drop("idx", axis=1)`. Portanto, depois dessa remocao, `X` possui 20 colunas numeradas de `0` a `19`. Ja o arquivo `design_ranges_main_AuAu-200.dat` tem tres colunas (`param`, `min`, `max`) e uma linha para cada um desses 20 parametros.

| Indices em `X` | Quantidade | Parametros no arquivo | Tratamento |
|---|---:|---|---|
| `0..10` | 11 | `ylossParam4At2` ate `stringTransverseShiftFrac` | Mantidos diretamente |
| `11..18` | 8 | Quatro parametros de `eta/s` e quatro de `zeta/s` | Substituidos por valores das curvas em 10 temperaturas |
| `19` | 1 | `eps_switch` | Mantido diretamente |

Assim, a frase "21 parametros originais" esta correta somente se contar a coluna auxiliar `idx` do arquivo de pontos. Para a transformacao numerica, a entrada tem **20 parametros fisicos**, nao 21.

```python
indices = [0,1,2,3,4,5,6,7,8,9,10]  # 11 parametros nao viscosos
extra_idx = [19]                       # eps_switch
```

Referencia: [configurations.py:422](configurations.py#L422) para a remocao de `idx` e [configurations.py:440](configurations.py#L440) para os indices preservados.

### Quais sao os oito parametros viscosos originais

Os indices `11..14` definem a parametrizacao de `eta/s(T)`:

1. `shear_viscosity_3_eta_over_s_T_kink_in_GeV` (`T_k`): temperatura do ponto de quebra;
2. `shear_viscosity_3_eta_over_s_low_T_slope_in_GeV` (`a_low`): inclinacao abaixo de `T_k`;
3. `shear_viscosity_3_eta_over_s_high_T_slope_in_GeV` (`a_high`): inclinacao acima de `T_k`;
4. `shear_viscosity_3_eta_over_s_at_kink` (`eta_s_k`): valor de `eta/s` em `T_k`.

Os indices `15..18` definem a parametrizacao de `zeta/s(T)`:

1. `bulk_viscosity_3_zeta_over_s_max` (`z_max`): altura maxima do pico;
2. `bulk_viscosity_3_zeta_over_s_T_peak_in_GeV` (`T_0`): temperatura do pico;
3. `bulk_viscosity_3_zeta_over_s_width_in_GeV` (`width`): largura do pico;
4. `bulk_viscosity_3_zeta_over_s_lambda_asymm` (`asym`): assimetria do pico.

Esses nomes e a ordem podem ser conferidos no cabecalho de [design_points_main_AuAu-200.dat](production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production/design_points_main_AuAu-200.dat) e nos intervalos de [design_ranges_main_AuAu-200.dat](production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production/design_ranges_main_AuAu-200.dat).

### Grade de temperatura da viscosidade

Define dez temperaturas igualmente espacadas entre `0.13` e `0.30`. Para cada temperatura, o design recebe um valor de `eta/s` e um de `zeta/s`.

```python
num_T = 10
Temperature_grid = np.linspace(0.13, 0.3, num_T)
```

Referencia: [configurations.py:446](configurations.py#L446).

### De 20 parametros fisicos para 32 atributos

Para cada ponto do design, os 8 parametros viscosos nao sao mantidos como colunas do design transformado. Eles sao avaliados em 10 temperaturas, produzindo 10 valores de `eta/s` e 10 valores de `zeta/s`. A montagem final e:

```text
11 parametros nao viscosos
+ 10 valores de eta/s(T)
+ 10 valores de zeta/s(T)
+ 1 eps_switch
= 32 atributos
```

Logo, a transformacao nao converte 21 parametros fisicos em 32. Ela converte **20 parametros fisicos de entrada** em **32 atributos de treinamento**, enquanto a coluna `idx` permanece apenas como identificador e e removida antes da transformacao.

### Limites transformados do design

Define os limites usados para os parametros de viscosidade discretizados: `eta/s` entre `0.01` e `0.65`, e `zeta/s` entre `0.01` e `0.2`.

```python
eta_min.append(0.01)
eta_max.append(0.65)
zeta_min.append(0.01)
zeta_max.append(0.2)
```

Referencia: [configurations.py:474](configurations.py#L474).

### Preparacao do design do emulador

`prepare_emu_design` carrega o design principal, aplica `transform_design` e transforma tambem os limites quando `do_transform_design` e verdadeiro. Se a opcao for desligada, usa o design original.

```python
if do_transform_design:
    design = transform_design(design.values)
    design_max, design_min = transform_design_min_max(design_max, design_min)
else:
    design = design.values
```

Referencia: [configurations.py:490](configurations.py#L490).

### Transformacao das multiplicidades

Controla se observaveis de multiplicidade sao treinados em escala logaritmica. O valor atual esta ligado; o comentario do codigo indica `log(1 + dY/dx)` ou, na analise padrao, `log(dY/dx)`.

```python
transform_multiplicities = True
```

Referencia: [configurations.py:325](configurations.py#L325).

### Funcoes de viscosidade

As funcoes abaixo definem como os parametros sao convertidos em `zeta/s(T)`, `eta/s(T)` e `tau_pi(T)` durante a transformacao do design.

```python
def zeta_over_s(T, zmax, T0, width, asym):
def eta_over_s(T, T_k, alow, ahigh, etas_k):
def taupi(T, T_k, alow, ahigh, etas_k, bpi):
```

Referencias: [configurations.py:383](configurations.py#L383), [configurations.py:390](configurations.py#L390) e [configurations.py:402](configurations.py#L402).

## Verde - BINS, observaveis e selecao

### Rotulos dos experimentos

Mapeia abreviacoes presentes nos arquivos para os nomes completos dos experimentos.

```python
expt_label_dict = {
    'BRAH': 'BRAHMS',
    'PHOB': 'PHOBOS',
    'STAR': 'STAR',
    'PHEN': 'PHENIX',
}
```

Referencia: [configurations.py:55](configurations.py#L55).

Os bins geometricos em si sao definidos em [bins_and_cuts.py](bins_and_cuts.py), que e importado no inicio de `configurations.py`.

### Formato Bayes dos observaveis

`bayes_dtype` define a estrutura dos dados para cada sistema, observavel e bin de centralidade. Cada observavel recebe campos `mean` e `err`, com o tipo numerico `float_t`.

```python
bayes_dtype = [
    (
        s,
        [
            (obs, [("mean", float_t, len(cent_list)), ("err", float_t, len(cent_list))])
            for obs, cent_list in obs_cent_list[s].items()
        ],
        number_of_models_per_run,
    )
    for s in system_strs
]
```

Referencia: [configurations.py:351](configurations.py#L351).

### Observaveis ativos

Por padrao, todos os observaveis presentes em `obs_cent_list` para cada sistema sao incluidos na calibracao MCMC.

```python
active_obs_list = {sys: list(obs_cent_list[sys].keys()) for sys in system_strs}
```

Referencia: [configurations.py:365](configurations.py#L365).

### Exclusao de observaveis

Quando ativada, permite retirar observaveis especificos de `active_obs_list`. O valor atual esta desligado, portanto nenhuma exclusao adicional e aplicada.

```python
exclude_data = False
```

Referencia: [configurations.py:369](configurations.py#L369).

## Roxo - EMULATOR e BAYES

### Sampler de parallel tempering

Seleciona o sampler de parallel tempering do `emcee`, usado para estimar evidencia Bayesiana. O estado atual usa o sampler MCMC convencional.

```python
usePTSampler = False
```

Referencia: [configurations.py:219](configurations.py#L219).

### Validacao do emulador

Controla se a analise deve validar o emulador em vez de usar os dados experimentais para estimar parametros. O caminho padrao esta desligado.

```python
validation = False
pseudovalidation = False
crossvalidation = False
fixed_validation_pt = 3
```

Com `validation = True`, pode-se usar pseudovalidacao, cross-validation ou um ponto independente definido por `fixed_validation_pt`.

Referencia: [configurations.py:224](configurations.py#L224).

### Parametros fixos

Permite manter parametros em valores especificos durante a estimativa. A lista esta vazia no estado atual.

```python
hold_parameters = False
hold_parameters_set = []
```

Referencia: [configurations.py:236](configurations.py#L236).

### Erro experimental nulo

Quando ativado, substitui todos os erros experimentais por zero. O estado atual preserva os erros.

```python
set_exp_error_to_zero = False
```

Referencia: [configurations.py:264](configurations.py#L264).

### Escala dos erros experimentais

Permite multiplicar erros experimentais por fatores definidos em `change_exp_error_vals`. A opcao esta desligada e os fatores definidos nao sao aplicados no estado atual.

```python
change_exp_error = False
change_exp_error_vals = {
    "Au-Au-200": {},
    "Pb-Pb-2760": {"dN_dy_proton": 1.0e-1, "mean_pT_proton": 1.0e-1},
}
```

Referencia: [configurations.py:269](configurations.py#L269).

### Intervalos do prior

Permite substituir os intervalos originais por um arquivo `restricted_prior_ranges/prior_range.dat` ou por valores definidos no codigo. A opcao esta desligada.

```python
change_parameters_range = False
```

Se for ativada, os intervalos alternativos codificados sao definidos em `change_parameters_range_set`.

Referencia: [configurations.py:280](configurations.py#L280).

### Correlacao dos erros experimentais

Controla a covariancia parametrizada entre bins de centralidade e grupos de observaveis. No estado atual, a covariancia experimental nao assume essas correlacoes.

```python
assume_corr_exp_error = False
cent_corr_length = 0.5
```

Os grupos de observaveis usados caso a opcao seja ligada sao definidos em `expt_obs_corr_group`.

Referencia: [configurations.py:329](configurations.py#L329) e [configurations.py:334](configurations.py#L334).

### Restricao do prior por sistema

`MAP_params` reserva dicionarios para parametros MAP salvos para uso posterior. Os valores nao sao preenchidos neste arquivo.

```python
MAP_params = {}
MAP_params["Pb-Pb-2760"] = {}
MAP_params["Au-Au-200"] = {}
```

Referencia: [configurations.py:516](configurations.py#L516).

## Amostragem MCMC

O numero de walkers, burn-in e passos de producao nao e definido em `configurations.py`; ele e passado ao executar `bayes_mcmc.py`, por exemplo:

```bash
./bayes_mcmc.py --nwalkers 200 --nburnsteps 1000 1000
```

As opcoes em `configurations.py` que alteram o MCMC sao `usePTSampler`, `hold_parameters`, `set_exp_error_to_zero`, `change_exp_error`, `assume_corr_exp_error`, `validation` e `active_obs_list`, documentadas acima.

## Observacoes sobre configuracoes comentadas

O arquivo preserva alternativas comentadas para:

- pseudodados em `dir_obs_exp`;
- sistemas adicionais, como Pb-Pb-5020, em `systems`;
- listas de parametros fixos em `hold_parameters_set`;
- remocao manual de observaveis em `exclude_data`;
- limites alternativos em `change_parameters_range_set`.

Essas alternativas nao estao ativas no estado atual e, portanto, nao alteram a execucao padrao.
