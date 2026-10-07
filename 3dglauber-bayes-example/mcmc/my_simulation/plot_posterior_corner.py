#!/usr/bin/env python3
"""Compact JETSCAPE-style posterior plot; requires numpy, scipy, matplotlib, h5py."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
PROJECT = next(parent for parent in HERE.parents if (parent / 'production_designs').is_dir())
DESIGN = PROJECT / 'production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production/design_ranges_main_AuAu-200.dat'
# Display labels follow the article reference; chain keys and units in data are unchanged.
LABELS = [
    r'$y_2$', r'$y_4$', r'$y_6$', r'$\sigma_{y_{\mathrm{loss}}}$',
    r'$\alpha_{\mathrm{rem}}$', r'$\alpha_{\mathrm{shadowing}}$',
    r'$\tau_{\mathrm{form}}$', r'$B_G\ [1/\mathrm{GeV}^2]$',
    r'$\mathrm{String}\ \sigma_x\ [\mathrm{fm}]$', r'$\mathrm{String}\ \sigma_\eta$',
    r'$\alpha_{\mathrm{shift}}$', r'$\frac{\eta}{s}\ T_{\mathrm{kink}}$',
    r'$\frac{\eta}{s}\ \mathrm{low}$-T slope',
    r'$\frac{\eta}{s}\ \mathrm{high}$-T slope',
    r'$\frac{\eta}{s}$ at kink', r'$\frac{\zeta}{s}$ max',
    r'$\frac{\zeta}{s}\ T_{\mathrm{peak}}$', r'$\frac{\zeta}{s}$ width',
    r'$\frac{\zeta}{s}\ \lambda$ assym.', r'$\mathrm{e\_switch}$'
]
GREEN, CONTOUR, RED = '#91bd80', '#71a962', '#d98178'


def load_samples(path, burnin, thin, max_samples):
    """Snapshot production length; stride steps before flattening walkers."""
    with h5py.File(path, 'r') as f:
        d = f['chain']
        if d.ndim != 3:
            raise ValueError('Expected chain shaped (walkers, steps, parameters).')
        walkers, steps, ndim = d.shape
        if not 0 <= burnin < steps:
            raise ValueError(f'burnin must be between 0 and {steps - 1}.')
        stride = max(thin, int(np.ceil(walkers * (steps - burnin) / max_samples)))
        # Read contiguous HDF chunks in bounded blocks instead of loading the 1.5 GB chain.
        blocks = []
        block_steps = 256 * stride
        for start in range(burnin, steps, block_steps):
            block = d[:, start:min(start + block_steps, steps):stride, :]
            blocks.append(block.reshape(-1, ndim))
        samples = np.concatenate(blocks)
    finite = np.isfinite(samples).all(axis=1)
    print(f'Chain: {walkers} walkers, {steps} production steps, {ndim} parameters')
    print(f'Discard: {burnin} steps; effective thinning: {stride}; finite samples: {finite.sum()}/{len(samples)}')
    samples = samples[finite]
    if len(samples) < 10:
        raise ValueError('At least 10 finite samples are required.')
    return samples, dict(walkers=walkers, steps=steps, burnin=burnin, thin=stride,
                         samples=len(samples), dropped_nonfinite=int((~finite).sum()))


def plot(samples, labels, ranges, ci, bins, smooth, logo=None):
    n = samples.shape[1]
    tail = (1 - ci) / 2
    bounds = np.quantile(samples, [tail, .5, 1 - tail], axis=0)
    cmap = LinearSegmentedColormap.from_list('posterior_green', ['white', '#e5f0df', '#b5d5a4', '#7fb36b'])
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'axes.linewidth': .55, 'axes.edgecolor': '#999999',
                         'font.size': 9, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(n, n, figsize=(3.3 + 1.25*n, 1.1 + 1.25*n), squeeze=False)
    fig.subplots_adjust(left=3.0/(3.3+1.25*n), right=.985, bottom=.095, top=.96,
                        wspace=.055, hspace=.075)
    dash = (0, (1.8, 1.5))
    for i in range(n):
        print(f'Plotting row {i+1}/{n}', flush=True)
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            ax.set_xlim(ranges[j])
            ax.spines[['top', 'right']].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
            ax.tick_params(direction='out', length=2, width=.5, colors='#555555', pad=2,
                           labelbottom=i == n-1, labelleft=j == 0 and i > 0)
            if i == j:
                hist, edges = np.histogram(samples[:, i], bins=bins, range=ranges[i], density=True)
                ax.stairs(hist, edges, color=GREEN, linewidth=1.0)
                ax.set_ylim(0, max(hist.max()*1.12, 1e-10))
                ax.set_yticks([])
                for value in bounds[[0, 2], i]:
                    ax.axvline(value, color=CONTOUR, ls=dash, lw=.85)
                lo, med, hi = bounds[:, i]
                ax.set_title(rf'${med:.2f}^{{+{hi-med:.2f}}}_{{-{med-lo:.2f}}}$', fontsize=11, pad=3)
            else:
                ax.set_ylim(ranges[i])
                ax.yaxis.set_major_locator(MaxNLocator(3, prune='both'))
                hist, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=bins,
                                             range=[ranges[j], ranges[i]])
                density = gaussian_filter(hist.T, smooth, mode='reflect')
                ax.pcolormesh(xe, ye, density, cmap=cmap, shading='flat', rasterized=True)
                ordered = np.sort(density.ravel())[::-1]
                mass = np.cumsum(ordered) / ordered.sum()
                levels = np.unique([ordered[min(np.searchsorted(mass, p), len(ordered)-1)]
                                    for p in (.90, .68)])
                levels = levels[(levels > density.min()) & (levels < density.max())]
                if levels.size:
                    ax.contour((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, density,
                               levels=levels, colors=CONTOUR, linewidths=.7, alpha=.85)
                ax.axhline(bounds[1, i], color=RED, ls=dash, lw=.85)
            ax.axvline(bounds[1, j], color=RED, ls=dash, lw=.85)
            if i == n-1:
                ax.set_xlabel(labels[j], fontsize=17, rotation=90, labelpad=8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if j == 0:
                plt.setp(ax.get_yticklabels(), rotation=45, va='center')
        position = axes[i, 0].get_position()
        fig.text(position.x0 - .65/fig.get_figwidth(), position.y0+position.height/2,
                 labels[i], fontsize=17, va='center', ha='right')
    if logo:
        if n < 3:
            raise ValueError('Logo requires at least three selected parameters.')
        pos = axes[max(0, n//3), min(n-1, 2*n//3)].get_position()
        logo_ax = fig.add_axes([pos.x0, pos.y0, min(.19, 1-pos.x0), .13])
        logo_ax.imshow(plt.imread(logo))
        logo_ax.axis('off')
    return fig, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=Path, default=HERE/'chain-idf-0_mine.hdf')
    parser.add_argument('--design', type=Path, default=DESIGN, help='CSV with param,min,max in chain order')
    parser.add_argument('--burnin', type=int, default=0, help='Additional production steps to discard per walker; original burn-in was not saved')
    parser.add_argument('--thin', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=200000, help='Bound sample count by increasing step stride')
    parser.add_argument('--parameters', nargs='+', help='Names or zero-based indices, in desired plot order')
    parser.add_argument('--ci', type=float, default=.90)
    parser.add_argument('--bins', type=int, default=45)
    parser.add_argument('--smooth', type=float, default=1., help='2D Gaussian smoothing in histogram bins')
    parser.add_argument('--logo', type=Path, help='Optional local PNG/JPEG logo in the upper triangle')
    parser.add_argument('--output', type=Path, default=HERE/'plots/posterior_corner', help='Output stem for PDF, PNG and JSON')
    args = parser.parse_args()
    if args.thin < 1 or args.max_samples < 10 or args.bins < 5 or args.smooth < 0 or not 0 < args.ci < 1:
        parser.error('Require thin>=1, max-samples>=10, bins>=5, smooth>=0 and 0<ci<1.')
    with args.design.open() as f:
        design = list(csv.DictReader(f))
    names = [row['param'] for row in design]
    samples, metadata = load_samples(args.chain, args.burnin, args.thin, args.max_samples)
    if samples.shape[1] != len(names):
        raise ValueError('Design parameter count does not match chain; supply the matching --design.')
    indices = list(range(len(names)))
    if args.parameters:
        indices = [int(p) if p.isdigit() else names.index(p) for p in args.parameters]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(names) for i in indices):
            parser.error('Parameter indices must be unique and within the chain dimensions.')
    selected = samples[:, indices]
    ranges = [(float(design[i]['min']), float(design[i]['max'])) for i in indices]
    if any(lo >= hi for lo, hi in ranges):
        raise ValueError('Design ranges must have min < max.')
    if any(np.any((selected[:, k] < lo) | (selected[:, k] > hi)) for k, (lo, hi) in enumerate(ranges)):
        raise ValueError('Samples outside design ranges; verify --design before plotting.')
    # Use mathematical labels only for the known original design schema.
    with DESIGN.open() as f:
        original_names = [row['param'] for row in csv.DictReader(f)]
    labels = [LABELS[i] if names == original_names else names[i] for i in indices]
    fig, bounds = plot(selected, labels, ranges, args.ci, args.bins, args.smooth, args.logo)
    stem = args.output.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        path = stem.with_suffix('.'+extension)
        fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=.15)
        print(f'Saved: {path}', flush=True)
    plt.close(fig)
    metadata.update(chain=str(args.chain.resolve()), design=str(args.design.resolve()),
                    credible_level=args.ci, contour_probabilities=[.68, .90], bins=args.bins,
                    smoothing_bins=args.smooth)
    metadata['posterior'] = {names[i]: dict(median=float(bounds[1,k]),
        ci_low=float(bounds[0,k]), ci_high=float(bounds[2,k])) for k,i in enumerate(indices)}
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
