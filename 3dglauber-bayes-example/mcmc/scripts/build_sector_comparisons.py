"""Compare saved baseline and sector posteriors without rerunning MCMC."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import dill
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_sector_studies import (
    ROOT, STUDIES, embed, markdown, observable_panel, physical_predictions,
    save_figure, validate_reference, write_csv, write_json,
)
from diagnose_phase2_numerical import load_emulator_module
from plot_posterior_corner import DESIGN, LABELS, load_samples, plot

BASELINE = ROOT / 'mcmc/01_01_baseline_reproduction/chain-idf-0.hdf'
DESTINATIONS = {'cs': '02_06_baseline_vs_cs_sector',
                'rbf': '02_07_baseline_vs_rbf_sector'}
BLUE, RED = '#477fb5', '#df9393'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_attributes(path):
    with h5py.File(path, 'r') as handle:
        return {key: value.item() if isinstance(value, np.generic) else value
                for key, value in handle.attrs.items()}


def input_vectors(chain, path):
    with np.load(path) as archive:
        theta = archive['theta'].copy()
        indices = archive['walker_step_indices'].copy()
    if len(theta) != 1000 or indices.shape != (1000, 2):
        raise ValueError(f'Expected 1000 audited posterior inputs: {path}')
    with h5py.File(chain, 'r') as handle:
        for value, (walker, step) in zip(theta, indices):
            np.testing.assert_array_equal(value, handle['chain'][int(walker), int(step)])
    return theta, indices


def summarize_and_plot_marginals(baseline, sector, design, study, label):
    bq = np.quantile(baseline, [.05, .16, .5, .84, .95], axis=0)
    sq = np.quantile(sector, [.05, .16, .5, .84, .95], axis=0)
    rows = []
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    for index, (ax, parameter) in enumerate(zip(axes.flat, design)):
        bounds = (float(parameter['min']), float(parameter['max']))
        for values, quantiles, color, name in (
                (baseline, bq, RED, 'Baseline'), (sector, sq, BLUE, label)):
            ax.hist(values[:, index], bins=55, range=bounds, density=True,
                    histtype='step', color=color, linewidth=1.3, label=name)
            ax.axvline(quantiles[2, index], color=color, linewidth=1)
            for value in quantiles[[0, 4], index]:
                ax.axvline(value, color=color, linestyle='--', linewidth=.7)
        ax.set(xlim=bounds, xlabel=LABELS[index], ylabel='Density')
        row = dict(parameter=parameter['param'])
        for prefix, quantiles in (('baseline', bq), ('sector', sq)):
            for k, name in enumerate(('q05', 'q16', 'median', 'q84', 'q95')):
                row[f'{prefix}_{name}'] = float(quantiles[k, index])
        width = bq[4, index] - bq[0, index]
        if width <= 0:
            raise ValueError(f'Zero baseline interval width: {parameter["param"]}')
        row['median_shift'] = float(sq[2, index] - bq[2, index])
        row['shift_over_baseline_90_width'] = row['median_shift'] / width
        row['width_ratio_90'] = float((sq[4, index] - sq[0, index]) / width)
        rows.append(row)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(f'Baseline vs {label}: marginals; dashed = central 90% interval')
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure(fig, study / 'posterior_marginals_comparison')
    write_csv(study / 'posterior_parameter_comparison.csv', rows)
    return rows


def panel(ax, ratio_ax, config, system, obs, baseline, sector, label):
    rows = observable_panel(ax, config, system, obs, sector, BLUE)
    edges = np.asarray(config.obs_cent_list[system][obs])
    centers = edges.mean(axis=1)
    bq = np.quantile(baseline, [.05, .16, .5, .84, .95], axis=0)
    sq = np.quantile(sector, [.05, .16, .5, .84, .95], axis=0)
    if bq.shape != sq.shape or not np.isfinite(bq).all():
        raise ValueError(f'Invalid comparison predictions: {system}/{obs}')
    experiment = np.array([row['experiment'] for row in rows])
    errors = np.array([row['experimental_std'] for row in rows])
    if np.any(experiment == 0):
        raise ValueError(f'Model/data ratio undefined at zero data: {system}/{obs}')
    starts = np.r_[0, np.flatnonzero(edges[1:, 0] > edges[:-1, 1] + 1e-8) + 1]
    for start, stop in zip(starts, np.r_[starts[1:], len(edges)]):
        slc = slice(start, stop)
        ax.fill_between(centers[slc], bq[1, slc], bq[3, slc], color=RED, alpha=.25)
        ax.plot(centers[slc], bq[2, slc], color=RED, linestyle='--', linewidth=1.3)
        if ratio_ax is not None:
            for quantiles, color, style in ((bq, RED, '--'), (sq, BLUE, '-')):
                limits = quantiles[[1, 3], slc] / experiment[slc]
                ratio_ax.fill_between(centers[slc], limits.min(axis=0),
                                      limits.max(axis=0), color=color, alpha=.25)
                ratio_ax.plot(centers[slc], quantiles[2, slc] / experiment[slc],
                              color=color, linestyle=style, linewidth=1.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=RED, ls='--', label='Baseline (68%)'),
                       Line2D([], [], color=BLUE, label=f'{label} (68%)'),
                       Line2D([], [], color='black', marker='.', ls='', label='Experiment')],
              fontsize=7)
    if ratio_ax is not None:
        ratio_ax.errorbar(centers, np.ones_like(experiment), yerr=errors / abs(experiment),
                          fmt='.', color='black', markersize=3, elinewidth=.7)
        ratio_ax.axhline(1, color='gray', linewidth=.8)
        ratio_ax.set(xlabel=ax.get_xlabel(), ylabel='Model / data')
        ax.set_xlabel('')
    for index, row in enumerate(rows):
        for k, name in enumerate(('q05', 'q16', 'median', 'q84', 'q95')):
            row[f'baseline_{name}'] = float(bq[k, index])
            row[f'sector_{name}'] = row.pop(name)
        row['baseline_median_over_data'] = float(bq[2, index] / experiment[index])
        row['sector_median_over_data'] = float(sq[2, index] / experiment[index])
    return rows


def make_notebook(study, model, manifest, rows, pages):
    label = model.upper() + ' sector'
    ranked = sorted(rows, key=lambda row: abs(row['shift_over_baseline_90_width']),
                    reverse=True)
    table = '| Parametro | Baseline mediana | Setor mediana | Delta / largura baseline 90% | Razao larguras 90% |\n'
    table += '|---|---:|---:|---:|---:|\n'
    for row in rows:
        table += f'| {row["parameter"]} | {row["baseline_median"]:.5g} | {row["sector_median"]:.5g} | {row["shift_over_baseline_90_width"]:+.3f} | {row["width_ratio_90"]:.3f} |\n'
    cells = [
        markdown(f'# Baseline vs {label}\n\n'
                 'Comparacao das cadeias salvas, sem MCMC, reponderacao ou treinamento.\n\n'
                 '**Baseline vermelho tracejado; modelo setorizado azul.** '
                 'O baseline e a reproducao JETSCAPE, nao a cadeia de referencia externa. '
                 'Os caminhos, metadados e versoes constam em `comparison_manifest.json`.'),
        markdown('## Amostragem e modelos\n\n'
                 '200.000 vetores por cadeia para parametros, stride 50 e nenhum '
                 'descarte adicional de producao. O burn-in salvo separadamente nao '
                 'entra nos graficos. Todas as amostras retidas sao finitas e estao '
                 'dentro dos mesmos limites do design.\n\n'
                 'O modelo setorizado separa cada observavel/sistema e, nas regioes '
                 'eta configuradas, o centro das laterais combinadas. '
                 'CS usa rho=1; RBF usa comprimentos pt=0.5 GeV/c, eta=1, '
                 'centralidade=10 pontos percentuais. A covariancia do emulador '
                 'permanece no ajuste em ambos os casos.'),
        markdown('## Parametros: mudancas descritivas\n\n'
                 'Delta = mediana setorizada - mediana baseline. A normalizacao '
                 'usa a largura do intervalo central de 90% do baseline; nao e '
                 'um teste de significancia. A razao de larguras e setor/baseline.\n\n'
                 'Maiores deslocamentos relativos: ' +
                 ', '.join(f'`{row["parameter"]}` ({row["shift_over_baseline_90_width"]:+.3f})'
                           for row in ranked[:5]) + '.\n\n' + table),
        embed(study / 'posterior_corner_comparison.png', study,
              'Corner: marginais e contornos conjuntos sobrepostos'),
        markdown('O corner mostra **ambas as cadeias tambem nos paineis 2D**: '
                 'densidade/contornos azuis do setor e contornos vermelhos tracejados '
                 'do baseline, com massas conjuntas de 68% e 90%. Titulos: mediana e '
                 'intervalo central de 90% do setor.'),
        embed(study / 'posterior_marginals_comparison.png', study, 'Comparacao das marginais'),
        markdown('## Previsoes dos observaveis\n\n'
                 '1.000 vetores de parametros por cadeia, com os mesmos indices '
                 'walker/passo e seed de selecao 12345. As predicoes setorizadas '
                 'reutilizam os arquivos auditados do estudo individual. As do '
                 'baseline foram calculadas com os mesmos emuladores treinados. '
                 'Parametros e indices usados estao salvos para verificacao.\n\n'
                 'Bandas: percentis 16-84 das **medias do emulador**, nao intervalos '
                 'preditivos completos com ruido do emulador/experimento. '
                 'Pontos pretos: dados experimentais com desvios padrao. '
                 'Figuras individuais em `observables/` incluem painel model/data; '
                 'paginas agrupadas abaixo mostram todos os 40 observaveis.'),
    ]
    cells.extend(embed(path, study, path.stem) for path in pages)
    cells.append(markdown('## Validacao e limites\n\n'
                          'Os indices de predicao foram confrontados diretamente '
                          'com os HDF5 e as previsoes setorizadas com os CSVs existentes. '
                          'Os emuladores foram validados contra os 40 observaveis '
                          'historicos do estudo global CS (rtol=1e-6, atol=1e-10).\n\n'
                          'Os caches usam scikit-learn 1.6.1 e a leitura local 1.8.0. '
                          'A validacao cobre as medias usadas nos graficos. '
                          'O hash binario da covariancia local difere do registrado '
                          'nas chains setorizadas, conforme documentado nos estudos '
                          'originais; a causa permanece indeterminada. Nenhuma cadeia '
                          'foi alterada nem reponderada. '
                          'Nao foram calculados ESS/R-hat/autocorrelacao, nem '
                          'evidencias ou Bayes factors.\n\n'
                          'Reproduzir da raiz do projeto:\n\n```bash\n'
                          f'python mcmc/scripts/build_sector_comparisons.py --model {model}\n```'))
    for index, cell in enumerate(cells):
        cell['id'] = f'{model}-comparison-{index:03d}'
    write_json(study / f'baseline_vs_{model}_sector.ipynb', dict(
        nbformat=4, nbformat_minor=5,
        metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                'language': 'python'}, 'language_info': {'name': 'python'}},
        cells=cells))


def build(model, baseline, baseline_meta, baseline_predictions, baseline_indices, design):
    folder, filename = STUDIES[model]
    source = ROOT / 'mcmc' / folder
    chain = source / filename
    study = ROOT / 'mcmc' / DESTINATIONS[model]
    obsout = study / 'observables'
    obsout.mkdir(parents=True, exist_ok=True)
    label = model.upper() + ' sector'
    sector, metadata = load_samples(chain, 0, 50, 200000)
    theta, indices = input_vectors(chain, source / 'plots/observables/posterior_inputs.npz')
    np.testing.assert_array_equal(indices, baseline_indices)
    for values in (baseline, sector):
        if not np.isfinite(values).all() or values.shape[1] != len(design):
            raise ValueError('Invalid chain dimensions or nonfinite samples.')
        for index, parameter in enumerate(design):
            if np.any((values[:, index] < float(parameter['min'])) |
                      (values[:, index] > float(parameter['max']))):
                raise ValueError('Posterior lies outside the design range.')
    ranges = [(float(row['min']), float(row['max'])) for row in design]
    fig, _ = plot(sector, LABELS, ranges, .90, 45, 1., comparison=baseline,
                  primary_label=label, reference_label='Baseline', comparison_joint=True)
    save_figure(fig, study / 'posterior_corner_comparison')
    rows = summarize_and_plot_marginals(baseline, sector, design, study, label)
    np.savez_compressed(study / 'sector_posterior_inputs.npz',
                        theta=theta, walker_step_indices=indices)
    config = __import__('configurations')
    pages, all_records = [], []
    for system in config.system_strs:
        print(model, 'observable comparisons:', system, flush=True)
        with np.load(source / f'plots/observables/predictions_{system}.npz') as archive:
            predictions = {key: archive[key].copy() for key in archive}
        observables = config.active_obs_list[system]
        if set(predictions) != set(observables):
            raise ValueError('Saved observable keys differ from the current configuration.')
        for obs in observables:
            previous = pd.read_csv(source / f'plots/observables/{system}_{obs}.csv')
            np.testing.assert_allclose(
                np.quantile(predictions[obs], [.05, .16, .5, .84, .95], axis=0).T,
                previous[['q05', 'q16', 'median', 'q84', 'q95']].to_numpy(),
                rtol=1e-12, atol=1e-14)
            fig, (ax, ratio_ax) = plt.subplots(
                2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            records = panel(ax, ratio_ax, config, system, obs,
                            baseline_predictions[system][obs], predictions[obs], label)
            fig.tight_layout()
            save_figure(fig, obsout / f'{system}_{obs}')
            write_csv(obsout / f'{system}_{obs}.csv', records)
            all_records.extend(records)
        with PdfPages(obsout / f'posterior_comparison_{system}.pdf') as pdf:
            for page, start in enumerate(range(0, len(observables), 9), start=1):
                fig, axes = plt.subplots(3, 3, figsize=(16, 12))
                selected = observables[start:start + 9]
                for ax, obs in zip(axes.flat, selected):
                    panel(ax, None, config, system, obs,
                          baseline_predictions[system][obs], predictions[obs], label)
                for ax in axes.flat[len(selected):]:
                    ax.set_visible(False)
                fig.suptitle(f'Baseline vs {label}: {system}; posterior means (68%)')
                fig.tight_layout(rect=(0, 0, 1, .96))
                path = obsout / f'posterior_comparison_{system}_page{page:02d}.png'
                fig.savefig(path, dpi=160)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(path)
    write_csv(study / 'posterior_predictions_comparison.csv', all_records)
    manifest = dict(
        baseline=dict(path=str(BASELINE.relative_to(ROOT)), **baseline_meta,
                      attributes=read_attributes(BASELINE)),
        sector=dict(path=str(chain.relative_to(ROOT)), **metadata,
                    attributes=read_attributes(chain)),
        corner_scope='Both marginal densities and joint 68%/90% contours',
        n_prediction_samples=1000, seed=12345,
        baseline_prediction_source='../02_06_baseline_vs_cs_sector/baseline_predictions',
        sector_prediction_source=str((source / 'plots/observables').relative_to(ROOT)),
        bands='Posterior variation of emulator means; no emulator or experimental noise draws',
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256=file_hash(Path(__file__)),
        sector_study_manifest=json.loads((source / 'study_manifest.json').read_text()),
    )
    write_json(study / 'comparison_manifest.json', manifest)
    make_notebook(study, model, manifest, rows, pages)
    print('Completed:', study, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('cs', 'rbf', 'both'), default='both')
    args = parser.parse_args()
    baseline, baseline_meta = load_samples(BASELINE, 0, 50, 200000)
    with DESIGN.open() as handle:
        design = list(csv.DictReader(handle))
    source = ROOT / 'mcmc' / STUDIES['cs'][0] / 'plots/observables/posterior_inputs.npz'
    _, indices = input_vectors(ROOT / 'mcmc' / STUDIES['cs'][0] / STUDIES['cs'][1], source)
    with h5py.File(BASELINE, 'r') as handle:
        theta = np.array([handle['chain'][int(w), int(s)] for w, s in indices])
    baseline_out = ROOT / 'mcmc' / DESTINATIONS['cs'] / 'baseline_predictions'
    baseline_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(baseline_out / 'posterior_inputs.npz',
                        theta=theta, walker_step_indices=indices)
    baseline_predictions, emulator_sources = {}, []
    with threadpool_limits(limits=1):
        module = load_emulator_module()
        config = __import__('configurations')
        for system in config.system_strs:
            print('Baseline prediction:', system, flush=True)
            path = ROOT / f'emulator/emulator-{system}-idf-0-npc-10.dill'
            with path.open('rb') as handle:
                emulator = dill.load(handle)
            emulator.__class__ = module.Emulator
            validated = validate_reference(emulator, system)
            baseline_predictions[system] = physical_predictions(
                emulator, theta, config.active_obs_list[system])
            np.savez_compressed(baseline_out / f'predictions_{system}.npz',
                                **baseline_predictions[system])
            emulator_sources.append(dict(system=system, path=str(path.relative_to(ROOT)),
                                         sha256=file_hash(path), validated_observables=validated))
        write_json(baseline_out / 'manifest.json',
                   dict(chain=str(BASELINE.relative_to(ROOT)), emulators=emulator_sources,
                        n_prediction_samples=1000, seed=12345))
        for model in STUDIES if args.model == 'both' else (args.model,):
            build(model, baseline, baseline_meta, baseline_predictions, indices, design)


if __name__ == '__main__':
    main()
