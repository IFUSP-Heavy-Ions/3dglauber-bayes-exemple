#!/usr/bin/env python3
"""Export emulator observable covariance at the last production step of a chain.

Example: python covariancia_emulador.py --chain mcmc/my_simulation/chain-idf-0.hdf
Use --walker to choose the final-step input point (default: walker 0).
The 20 parameters are inputs; returned covariance is between observable bins.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parent


def correlacao(cov):
    sigma = np.sqrt(np.diag(cov))
    denominator = np.outer(sigma, sigma)
    return np.divide(cov, denominator, out=np.full_like(cov, np.nan), where=denominator > 0)


def painel(ax, system, obs, cov, physical_bins):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
    r = correlacao(cov)
    bins = np.asarray(physical_bins)
    if bins.shape != (len(r), 2):
        raise ValueError(f'Bin mapping mismatch for {system}/{obs}: {bins.shape} vs {r.shape}')
    positions = bins
    if '_eta_' in obs:
        # Match the reference's discrete-bin layout: equal cell widths and
        # small separators between detector regions, with physical tick labels.
        centers = bins.mean(axis=1)
        spacing = np.diff(centers)
        gaps = spacing > 1.8*np.median(spacing)
        starts = np.arange(len(bins), dtype=float)
        starts[1:] += .5*np.cumsum(gaps)
        positions = np.column_stack((starts,starts+1))
    vertices = [[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
                for y0,y1 in positions for x0,x1 in positions]
    cmap = LinearSegmentedColormap.from_list('article_correlation', ['#00234d','#465573','#a7a083','#fff13c'])
    vmin = -1 if np.nanmin(r) < -1e-8 else 0
    collection = PolyCollection(vertices, array=r.ravel(), cmap=cmap,
                                clim=(vmin,1), edgecolors='none', rasterized=True)
    ax.add_collection(collection)
    lo, hi = positions.min(), positions.max()
    ax.set_xlim(lo,hi)
    ax.set_ylim(hi,lo)
    ax.set_aspect('equal')
    if '_pt_' in obs:
        axis = r'$p_T$ (GeV/c)'
    elif '_eta_' in obs:
        axis = r'$\eta$'
    else:
        axis = 'Centrality (%)'
    ax.set_xlabel(axis)
    ax.set_ylabel(axis)
    if '_eta_' in obs:
        indices = np.unique(np.rint(np.linspace(0,len(bins)-1,6)).astype(int))
        ticks = positions.mean(axis=1)[indices]
        labels = [f'{value:.1f}' for value in bins.mean(axis=1)[indices]]
        ax.set_xticks(ticks,labels)
        ax.set_yticks(ticks,labels)
    elif len(r) <= 7:
        centers = bins.mean(axis=1)
        ax.set_xticks(centers)
        ax.set_yticks(centers)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    parts = obs.split('_')
    if 'cen' in parts:
        i = parts.index('cen')
        centrality = f'{int(parts[i+1])}–{int(parts[i+2])}%'
    else:
        centrality = ''
    symbol = (r'$dN/d\eta$' if obs.startswith('dNdeta') else
              r'$v_2$' if obs.startswith('v22') else
              r'$v_3$' if obs.startswith('v32') else parts[0])
    experiment = {'PHEN':'PHENIX','PHOB':'PHOBOS','STAR':'STAR','BRAH':'BRAHMS'}.get(parts[-1],parts[-1])
    ax.set_title(f'{system} GeV · {symbol} {centrality}\n{experiment}', fontsize=11)
    return collection, r


def salvar_plots(predictions, output, walker):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from bins_and_cuts import obs_cent_list
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':11})
    destination = output / 'plots'
    destination.mkdir(exist_ok=True)
    records = []
    with PdfPages(destination / 'all_observables.pdf') as pdf:
        for system, (mean, cov) in predictions.items():
            for obs in mean:
                matrix = cov[obs,obs][0]
                bins = obs_cent_list[system][obs]
                fig, ax = plt.subplots(figsize=(5.4,4.8), layout='constrained')
                artist, r = painel(ax, system, obs, matrix, bins)
                fig.colorbar(artist, ax=ax, fraction=.045, pad=.04, label=r'$C_{ij}/(\sigma_i\sigma_j)$')
                fig.suptitle(f'Emulator uncertainty · walker {walker}', fontsize=10)
                stem = f'{system}_{obs}'
                fig.savefig(destination / f'{stem}.png', dpi=200)
                fig.savefig(destination / f'{stem}.pdf')
                np.savetxt(destination / f'{stem}_correlation.csv', r, delimiter=',')
                pdf.savefig(fig)
                plt.close(fig)
                records.append((system,obs))
    reference = [('d-Au-200','dNdeta_eta_cen_00_05_PHEN'),
                 ('Au-Au-200','v22_eta_cen_03_15_PHOB'),
                 ('Au-Au-200','v22_pt_cen_20_30_PHEN')]
    selected = [pair for pair in reference if pair in records]
    if selected:
        fig, axes = plt.subplots(len(selected),1, figsize=(5.5,4.5*len(selected)), squeeze=False, layout='constrained')
        for ax, (system,obs) in zip(axes[:,0],selected):
            artist, _ = painel(ax,system,obs,predictions[system][1][obs,obs][0],obs_cent_list[system][obs])
            fig.colorbar(artist,ax=ax,fraction=.045,pad=.04,label=r'$C_{ij}/(\sigma_i\sigma_j)$')
        fig.savefig(destination / 'article_observables.png',dpi=200)
        fig.savefig(destination / 'article_observables.pdf')
        plt.close(fig)
    return {'directory':str(destination), 'walker_index':int(walker), 'observables':len(records),
            'eta_layout':'equal-width bin cells; physical center labels; large detector gaps compressed'}


def ultimo_step(path, walker=None):
    """Read only the last step, preserving the sample dimension."""
    with h5py.File(path, 'r') as f:
        d = f['chain']
        if d.ndim != 3 or d.shape[2] != 20 or min(d.shape) == 0:
            raise ValueError('Expected nonempty chain shaped (walkers, steps, 20).')
        shape = d.shape
        if walker is not None and not 0 <= walker < shape[0]:
            raise ValueError(f'walker must be between 0 and {shape[0]-1}.')
        indices = np.arange(shape[0]) if walker is None else np.array([walker])
        theta = d[:, shape[1]-1, :] if walker is None else d[walker:walker+1, shape[1]-1, :]
    if not np.isfinite(theta).all():
        raise ValueError('The selected final-step parameters contain nonfinite values.')
    return np.asarray(theta), indices, shape


def matriz_likelihood(mean, covariance, slices):
    """Assemble emulator-only covariance in the exact likelihood bin order."""
    nobs = max(s.stop for _, s in slices)
    nsamples = next(iter(mean.values())).shape[0]
    y = np.empty((nsamples, nobs))
    cov = np.empty((nsamples, nobs, nobs))
    bins = [None] * nobs
    for obs1, slc1 in slices:
        y[:, slc1] = mean[obs1]
        for i in range(slc1.stop - slc1.start):
            bins[slc1.start+i] = {'observable': str(obs1), 'bin_index': i}
        for obs2, slc2 in slices:
            cov[:, slc1, slc2] = covariance[obs1, obs2]
    if not np.isfinite(cov).all() or not np.isfinite(y).all():
        raise ValueError('Emulator prediction contains nonfinite values.')
    return y, cov, bins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', required=True, type=Path)
    parser.add_argument('--walker', type=int, default=0, help='Zero-based walker index (default: 0)')
    parser.add_argument('--output', type=Path, help='Default: <chain directory>/covariancia_emulador')
    args = parser.parse_args()
    path = args.chain.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else path.parent / 'covariancia_emulador'
    theta, walkers, shape = ultimo_step(path, args.walker)

    # Existing project modules use relative paths during import. Anchor those
    # paths to this project even when this script is called from elsewhere.
    os.environ['WORKDIR'] = str(PROJECT)
    os.chdir(PROJECT)
    from bayes_mcmc_cs_model import Chain
    from emulator import Emulator, Trained_Emulators
    from types import MethodType

    # Dill files can carry a prediction method with incomplete saved globals.
    # Use the project's prediction implementation with the unchanged trained
    # GP/PCA state. This restores configuration imports, without retraining.
    for emu in Trained_Emulators.values():
        emu.predict = MethodType(Emulator.predict, emu)
        emu._inverse_transform = MethodType(Emulator._inverse_transform, emu)

    model = Chain(path=path)
    if model.ndim != theta.shape[1]:
        raise ValueError(f'Emulator expects {model.ndim} parameters, chain has {theta.shape[1]}.')
    effective_theta = theta.copy()
    predictions = model._predict(effective_theta, return_cov=True)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'chain': str(path), 'chain_shape': list(shape), 'step_index': shape[1]-1,
        'walker_indices': walkers.tolist(), 'parameter_labels': list(map(str, model.labels)),
        'covariance': 'emulator only, conditional on each final-step parameter vector',
        'coordinates': 'same transformed observable coordinates as the likelihood',
        'emulator_configuration': str(PROJECT / 'configurations.py'),
        'prediction_implementation': 'emulator.Emulator.predict with loaded trained state',
        'systems': {},
    }
    for system, (mean, covariance) in predictions.items():
        y, cov, bins = matriz_likelihood(mean, covariance, model._slices[system])
        filename = f'{system}.npz'
        np.savez_compressed(output / filename, theta=theta, effective_theta=effective_theta,
                            walker_indices=walkers, step_index=shape[1]-1,
                            mean=y, covariance=cov)
        metadata['systems'][system] = {
            'file': filename, 'shape': list(cov.shape), 'bins': bins,
        }
        print(f'{system}: covariance {cov.shape} -> {output / filename}', flush=True)
    # Plots use the first selected walker, never an average of different theta.
    plot_predictions = model._predict(effective_theta[:1].copy(), return_cov=True)
    metadata['plots'] = salvar_plots(plot_predictions, output, walkers[0])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    print(f'Metadata: {output / "metadata.json"}')


if __name__ == '__main__':
    main()
