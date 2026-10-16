"""Analyze saved sector chains without MCMC, training, or model-data writes."""
import argparse
import base64
import csv
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import dill
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits

from diagnose_phase2_numerical import load_emulator_module
from plot_posterior_corner import DESIGN, LABELS, load_samples

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDIES = {
    'cs': ('02_04_cs_sector', 'chain-idf-0-cs-observable-region-rho-1-a77902df73a5.hdf'),
    'rbf': ('02_05_rbf_sector', 'chain-idf-0-rbf-observable-a873ae558cc9.hdf'),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + extension), dpi=160, bbox_inches='tight')
    plt.close(fig)


def marginals(samples, design, out, title):
    quantiles = np.quantile(samples, [.05, .16, .5, .84, .95], axis=0)
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    rows = []
    for index, (ax, parameter) in enumerate(zip(axes.flat, design)):
        lo, hi = float(parameter['min']), float(parameter['max'])
        ax.hist(samples[:, index], bins=55, range=(lo, hi), density=True,
                histtype='stepfilled', color='#477fb5', alpha=.65)
        for value in quantiles[[0, 4], index]:
            ax.axvline(value, color='#285b8e', linestyle='--', linewidth=.8)
        ax.axvline(quantiles[2, index], color='#b45a45', linewidth=1)
        ax.set(xlim=(lo, hi), xlabel=LABELS[index], ylabel='Density')
        ax.set_title(f'Median: {quantiles[2, index]:.4g}', fontsize=10)
        rows.append(dict(parameter=parameter['param'],
                         mean=float(samples[:, index].mean()),
                         std=float(samples[:, index].std(ddof=1)),
                         **{name: float(quantiles[k, index]) for k, name in
                            enumerate(('q05', 'q16', 'median', 'q84', 'q95'))}))
    fig.suptitle(title + ' - marginals (dashed: central 90% interval)')
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure(fig, out / 'posterior_marginals')
    write_csv(out / 'posterior_parameters.csv', rows)
    return rows


def physical_predictions(model, theta, observables):
    prediction = model.predict(theta, return_cov=False)
    values = {}
    for obs in observables:
        value = np.asarray(prediction[obs])
        if 'dN' in obs or 'dET' in obs:
            value = np.exp(value)
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite emulator predictions: {obs}')
        values[obs] = value
    return values


def validate_reference(model, system):
    reference = ROOT / 'mcmc/02_01_global_cs/plots/observables/chain-idf-0-cs-system-rho-1'
    with np.load(reference / 'posterior_inputs.npz') as archive:
        theta = archive['theta']
    predictions = physical_predictions(model, theta, model.observables)
    for obs, value in predictions.items():
        frame = pd.read_csv(reference / f'{system}_{obs}.csv')
        np.testing.assert_allclose(
            np.quantile(value, [.16, .5, .84], axis=0).T,
            frame[['q16', 'median', 'q84']].to_numpy(), rtol=1e-6, atol=1e-10,
            err_msg=f'Rebound emulator disagrees with historical predictions: {system}/{obs}')
    return len(predictions)


def observable_panel(ax, config, system, obs, values, color):
    edges = np.asarray(config.obs_cent_list[system][obs])
    centers = edges.mean(axis=1)
    experiment_path = (ROOT / config.dir_obs_exp / system /
                       config.expt_label_dict[obs[-4:]] / (obs + '.dat'))
    frame = pd.read_csv(experiment_path, sep=' ', skiprows=2, escapechar='#').iloc[:len(edges)]
    experiment = frame['val'].to_numpy(dtype=float)
    variance = frame['err'].to_numpy(dtype=float)
    if len(frame) != len(edges) or not np.isfinite(experiment).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError(f'Invalid experimental data: {experiment_path}')
    std = np.sqrt(variance)
    quantiles = np.quantile(values, [.05, .16, .5, .84, .95], axis=0)
    if quantiles.shape[1] != len(edges):
        raise ValueError(f'Bin count mismatch: {system}/{obs}')
    # Do not draw curves through disconnected acceptance gaps.
    starts = np.r_[0, np.flatnonzero(edges[1:, 0] > edges[:-1, 1] + 1e-8) + 1]
    stops = np.r_[starts[1:], len(edges)]
    for k, (start, stop) in enumerate(zip(starts, stops)):
        slc = slice(start, stop)
        ax.fill_between(centers[slc], quantiles[1, slc], quantiles[3, slc],
                        color=color, alpha=.3, label='Posterior means: 68%' if k == 0 else None)
        ax.plot(centers[slc], quantiles[2, slc], color=color, linewidth=1.4)
    ax.errorbar(centers, experiment, yerr=std, fmt='.', color='black',
                markersize=3, elinewidth=.8, label='Experiment')
    from experimental_sector_covariance import observable_axis, AXIS_UNITS
    axis = observable_axis(obs)
    ax.set(title=f'{system}: {obs}', xlabel=f'{axis} [{AXIS_UNITS[axis]}]',
           ylabel='Observable (physical units)')
    if 'dN' in obs:
        if np.any(values <= 0) or np.any(experiment <= 0):
            raise ValueError(f'Nonpositive multiplicity: {system}/{obs}')
        ax.set_yscale('log')
    ax.tick_params(labelsize=8)
    ax.title.set_fontsize(9)
    return [dict(system=system, observable=obs, bin_index=index,
                 bin_low=float(edges[index, 0]), bin_high=float(edges[index, 1]),
                 center=float(centers[index]), experiment=float(experiment[index]),
                 experimental_std=float(std[index]),
                 **{name: float(quantiles[k, index]) for k, name in
                    enumerate(('q05', 'q16', 'median', 'q84', 'q95'))})
            for index in range(len(edges))]


def markdown(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(True))


def embed(path, study, caption):
    cell = markdown(f'### {caption}\n\n![{caption}](attachment:figure.png)\n\n'
                    f'Arquivo: `{path.relative_to(study)}`.')
    cell['attachments'] = {'figure.png': {
        'image/png': base64.b64encode(path.read_bytes()).decode('ascii')}}
    return cell


def notebook(study, model, inventory, parameter_rows, figures):
    title = model.upper() + ' setorizado'
    attrs = inventory['chain_attributes']
    kernel = ('CS: rho = 1; correlacao uniforme dentro de cada setor.' if model == 'cs'
              else 'RBF: exp(-((x_i-x_j)/length)^2), sem fator 1/2.\n\n'
                   f'Comprimentos registrados: `{attrs["experimental_rbf_lengths"]}`.')
    table = '| Parametro | Mediana | 68% inferior | 68% superior | 90% inferior | 90% superior |\n'
    table += '|---|---:|---:|---:|---:|---:|\n'
    for row in parameter_rows:
        table += '| ' + row['parameter'] + ' | ' + ' | '.join(
            f'{row[key]:.5g}' for key in ('median', 'q16', 'q84', 'q05', 'q95')) + ' |\n'
    cells = [
        markdown(f'# Estudo: {title}\n\n'
                 'Analise da cadeia finalizada, sem nova amostragem MCMC ou treinamento.\n\n'
                 f'Cadeia: `{inventory["chain"]}`.\n\n'
                 f'Walkers: **{inventory["walkers"]}**; burn-in salvo: **20.000**; '
                 'producao: **50.000 passos por walker**; **20 parametros**.\n\n'
                 'O burn-in esta no dataset separado `burnin`, portanto nao foi '
                 'descartado novamente da producao.'),
        markdown('## Modelo de correlacao\n\n' + kernel + '\n\n'
                 'Cada chave exata de observavel e sistema constitui um bloco. '
                 'Nos observaveis eta configurados, o centro e um setor e as duas '
                 'laterais, juntas, sao outro. Nao ha covariancia experimental entre '
                 'esses setores, nem entre observaveis ou sistemas distintos. No RBF, '
                 'as laterais compartilham o setor, mas a correlacao depende da '
                 'distancia fisica em eta. A cobertura continua PHOBOS dN/deta '
                 'permanece em um unico bloco.\n\n'
                 'A covariancia do emulador continua incluida no ajuste. CS com rho=1 '
                 'gera blocos experimentais semidefinidos, nao um erro por si so.\n\n'
                 'Detalhes e tamanhos de cada setor na reconstrucao local: '
                 '`covariance_check.json`. Os metadados salvos na cadeia sao a '
                 'referencia para identificar o ajuste original.'),
        markdown('## Posterior dos parametros\n\n'
                 '200.000 amostras para os graficos, com stride 50 nos passos de '
                 'producao. Isso limita memoria e custo grafico, mas **nao** certifica '
                 'independencia ou convergencia. O corner usa contornos conjuntos de '
                 '68%/90% e intervalos marginais centrais de 90%.\n\n' + table),
        embed(study / 'plots/posterior_marginals.png', study, 'Marginais dos 20 parametros'),
        embed(study / 'plots/posterior_corner.png', study, 'Corner completo'),
        markdown('## Observaveis preditos a partir da posterior\n\n'
                 f'{inventory["n_prediction_samples"]} vetores de parametros sorteados '
                 'sem reposicao da amostra retida, seed 12345. As figuras mostram a '
                 'mediana e o intervalo central de 68% das **medias do emulador**; '
                 'as tabelas tambem guardam o intervalo central de 90%. '
                 'Os dados sao mostrados com seus desvios padrao experimentais.\n\n'
                 'As bandas nao incluem sorteios do erro do emulador ou do erro '
                 'experimental: **nao sao intervalos preditivos completos de novas '
                 'medidas**. Esta e a mesma convencao dos estudos anteriores. '
                 'Multiplicidades foram reconvertidas de log para a escala fisica.\n\n'
                 'Os graficos nao ligam pontos atraves de lacunas de aceitacao. '
                 'Cada observavel tambem possui figuras individuais PNG/PDF e CSV '
                 'em `plots/observables/`.'),
    ]
    for figure in figures:
        cells.append(embed(figure, study, figure.stem))
    cells.extend([
        markdown('## Validacao e limites\n\n'
                 'As medias dos emuladores carregados foram comparadas com os '
                 'quantis salvos do estudo global CS: **40/40 observaveis** '
                 'coincidiram (rtol=1e-6, atol=1e-10). A classe atual de `emulator.py` '
                 'foi ligada ao estado treinado, sem retreinar, evitando depender '
                 'de bytecode Python legado do cache dill. As versoes usadas e os '
                 'hashes dos emuladores constam em `study_manifest.json`.\n\n'
                 'Os caches foram salvos com scikit-learn 1.6.1 e lidos com 1.8.0. '
                 'O teste historico valida as medias usadas nestes graficos, nao '
                 'uma equivalencia geral entre versoes ou a covariancia preditiva.\n\n'
                 f'Assinatura binaria da covariancia local igual a da cadeia: '
                 f'**{inventory["covariance_signature_matches"]}**. '
                 'Se diferente, isso nao permite certificar a reproducao bit a bit '
                 'da verossimilhanca remota; nao foi determinado se a causa e '
                 'arredondamento entre ambientes ou alguma diferenca nos inputs. '
                 'As regioes atuais foram comparadas com as salvas e coincidem. '
                 'Nenhum novo ajuste, reponderacao ou alteracao da cadeia foi feito. '
                 'As predicoes usam diretamente os parametros da cadeia salva.\n\n'
                 'A cadeia foi inspecionada quanto a forma e valores finitos da '
                 'amostra retida. Nao foram calculados ESS, tempo de autocorrelacao '
                 'ou R-hat; a finalizacao da cadeia nao demonstra convergencia.'),
        markdown('## Reproducao e arquivos\n\n'
                 'Na raiz do projeto, para recriar o corner:\n\n```bash\n'
                 f'python mcmc/scripts/plot_posterior_corner.py --chain mcmc/{study.name}/{inventory["chain"]} '
                 '--thin 50 --max-samples 200000\n```\n\n'
                 'Para recriar marginais, predicoes e este notebook:\n\n```bash\n'
                 f'python mcmc/scripts/build_sector_studies.py --model {model}\n```\n\n'
                 '`plots/posterior_parameters.csv`: resumo dos parametros.\n\n'
                 '`plots/observables/posterior_inputs.npz`: parametros e indices '
                 'originais walker/passo para auditoria.\n\n'
                 '`plots/observables/predictions_*.npz`: medias em escala fisica.\n\n'
                 'Todos os graficos exibidos estao incorporados neste notebook.'),
    ])
    for index, cell in enumerate(cells):
        cell['id'] = f'{model}-{index:03d}'
    write_json(study / f'{model}_sector_study.ipynb', dict(
        nbformat=4, nbformat_minor=5,
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}, cells=cells))


def build(model, module, n_prediction_samples):
    folder, filename = STUDIES[model]
    study = ROOT / 'mcmc' / folder
    chain = study / filename
    out = study / 'plots'
    obsout = out / 'observables'
    obsout.mkdir(parents=True, exist_ok=True)
    samples, inventory = load_samples(chain, 0, 50, 200000)
    if inventory['dropped_nonfinite']:
        raise ValueError('Nonfinite samples found in chain.')
    if (inventory['walkers'], inventory['steps'], samples.shape[1]) != (200, 50000, 20):
        raise ValueError('Unexpected finalized chain dimensions.')
    with h5py.File(chain, 'r') as handle:
        if handle['burnin'].shape != (200, 20000, 20):
            raise ValueError('Unexpected saved burn-in dimensions.')
        attrs = {key: value.item() if isinstance(value, np.generic) else value
                 for key, value in handle.attrs.items()}
    inventory.update(chain=filename, chain_attributes=attrs,
                     n_prediction_samples=n_prediction_samples, seed=12345)
    config = __import__('configurations')
    from sector_mcmc_cli import inspect_covariance
    covariance = inspect_covariance(
        config, model, attrs.get('experimental_cs_rho', 1.),
        json.loads(attrs.get('experimental_rbf_lengths', '{}')),
        json.loads(attrs.get('experimental_rbf_overrides', '{}')),
        json.loads(attrs['experimental_rapidity_regions']))
    inventory['covariance_signature_matches'] = (
        covariance['signature'] == attrs['experimental_covariance_signature'])
    if json.loads(attrs['experimental_rapidity_regions']) != json.loads((ROOT / 'rapidity_regions.json').read_text()):
        raise ValueError('Current rapidity partition differs from the saved chain.')
    if not inventory['covariance_signature_matches']:
        warnings.warn(
            f'{model}: local covariance fingerprint differs from the remote chain. '
            'Recording both; posterior plots use saved samples without refitting.',
            RuntimeWarning)
    covariance['saved_chain_signature'] = attrs['experimental_covariance_signature']
    covariance['matches_saved_chain_signature'] = inventory['covariance_signature_matches']
    write_json(study / 'covariance_check.json', covariance)
    with DESIGN.open() as handle:
        design = list(csv.DictReader(handle))
    parameter_rows = marginals(samples, design, out, model.upper() + ' sector')
    rng = np.random.default_rng(12345)
    selected = rng.choice(len(samples), size=n_prediction_samples, replace=False)
    theta = samples[selected]
    # load_samples flattens each 256-selected-step block in walker-major order.
    steps_kept = len(range(0, inventory['steps'], inventory['thin']))
    indices = []
    for index in selected:
        offset = 0
        for start in range(0, steps_kept, 256):
            count = min(256, steps_kept - start)
            if index < offset + inventory['walkers'] * count:
                local = index - offset
                indices.append((local // count, (start + local % count) * inventory['thin']))
                break
            offset += inventory['walkers'] * count
    with h5py.File(chain, 'r') as handle:
        for value, (walker, step) in zip(theta, indices):
            np.testing.assert_array_equal(value, handle['chain'][walker, step])
    np.savez_compressed(obsout / 'posterior_inputs.npz', theta=theta,
                        walker_step_indices=np.asarray(indices))
    figures, emulator_sources = [], []
    for system in config.system_strs:
        print(model, 'predicting', system, flush=True)
        cache = ROOT / f'emulator/emulator-{system}-idf-0-npc-10.dill'
        with cache.open('rb') as handle:
            emulator = dill.load(handle)
        emulator.__class__ = module.Emulator
        validated = validate_reference(emulator, system)
        observables = config.active_obs_list[system]
        predictions = physical_predictions(emulator, theta, observables)
        np.savez_compressed(obsout / f'predictions_{system}.npz', **predictions)
        rows = []
        for obs in observables:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            records = observable_panel(ax, config, system, obs, predictions[obs], '#d47f35')
            ax.legend(fontsize=8)
            fig.tight_layout()
            save_figure(fig, obsout / f'{system}_{obs}')
            write_csv(obsout / f'{system}_{obs}.csv', records)
            rows.extend(records)
        write_csv(obsout / f'posterior_predictions_{system}.csv', rows)
        with PdfPages(obsout / f'posterior_observables_{system}.pdf') as pdf:
            for page, start in enumerate(range(0, len(observables), 9), start=1):
                fig, axes = plt.subplots(3, 3, figsize=(16, 12))
                for ax, obs in zip(axes.flat, observables[start:start + 9]):
                    observable_panel(ax, config, system, obs, predictions[obs], '#d47f35')
                for ax in axes.flat[len(observables[start:start + 9]):]:
                    ax.set_visible(False)
                axes.flat[0].legend(fontsize=7)
                fig.suptitle(f'{model.upper()} sector - {system} - posterior emulator means (68%)')
                fig.tight_layout(rect=(0, 0, 1, .96))
                path = obsout / f'posterior_observables_{system}_page{page:02d}.png'
                fig.savefig(path, dpi=160)
                pdf.savefig(fig)
                plt.close(fig)
                figures.append(path)
        emulator_sources.append(dict(system=system, sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                                     reference_observables_validated=validated))
    inventory.update(emulators=emulator_sources,
                     bands='Posterior variation of emulator means; no emulator or experimental noise draws',
                     versions=dict(python=platform.python_version(), numpy=np.__version__,
                                   sklearn=sklearn.__version__, dill=dill.__version__),
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(study / 'study_manifest.json', inventory)
    notebook(study, model, inventory, parameter_rows, figures)
    print('Completed:', study, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('cs', 'rbf', 'both'), default='both')
    parser.add_argument('--prediction-samples', type=int, default=1000)
    args = parser.parse_args()
    if not 10 <= args.prediction_samples <= 200000:
        parser.error('--prediction-samples must be between 10 and 200000.')
    with threadpool_limits(limits=1):
        module = load_emulator_module()
        for model in STUDIES if args.model == 'both' else (args.model,):
            build(model, module, args.prediction_samples)


if __name__ == '__main__':
    main()
