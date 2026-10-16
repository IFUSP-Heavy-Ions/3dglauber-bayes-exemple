#!/usr/bin/env python3
"""Ensemble MCMC diagnostics. Outputs default to <chain directory>/plots/convergencia."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_posterior_corner import DESIGN, LABELS, GREEN, CONTOUR, RED


def autocorrelation(x):
    """Mean normalized walker ACF; FFT estimator and Sokal window c=5 (emcee)."""
    n = x.shape[1]
    size = 1 << (2*n - 1).bit_length()
    acf = np.zeros(n)
    valid = 0
    for walker in x:
        centered = walker - walker.mean()
        if np.var(centered) == 0:
            continue
        fft = np.fft.rfft(centered, n=size)
        a = np.fft.irfft(fft * fft.conj(), n=size)[:n]
        acf += a / a[0]
        valid += 1
    if valid != len(x):
        return np.full(n, np.nan), np.nan, 0, False
    acf /= valid
    taus = 2*np.cumsum(acf) - 1
    windows = np.flatnonzero(np.arange(n) >= 5*taus)
    window = int(windows[0]) if len(windows) else n-1
    tau = float(taus[window])
    if tau <= 0 or not np.isfinite(tau):
        tau = np.nan
    return acf, tau, window, bool(len(windows))


def grid(labels, title):
    cols = min(4, len(labels))
    rows = int(np.ceil(len(labels)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.55*rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        ax.set_title(label, fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=16)
    return fig, list(axes.flat)[:len(labels)]


def interval_bands(ax, time, quantiles):
    """Central empirical ensemble intervals: 90% = q05-q95; 60% = q20-q80."""
    ax.fill_between(time, quantiles[0], quantiles[4], color=GREEN, alpha=.30,
                    label='90% interval', zorder=1)
    ax.fill_between(time, quantiles[1], quantiles[3], color=CONTOUR, alpha=.55,
                    label='60% interval', zorder=2)
    ax.plot(time, quantiles[2], color=RED, lw=1, label='Median', zorder=4)


def save(fig, output, name):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .957),
                   ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .925 if handles else .965))
    for ext in ['pdf', 'png']:
        fig.savefig(output/f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--burnin', type=int, default=0, help='Discard saved steps per walker')
    parser.add_argument('--trace-walkers', type=int, default=20)
    parser.add_argument('--display-points', type=int, default=1500, help='Display decimation only; ACF uses all steps')
    args = parser.parse_args()
    chain = args.chain.expanduser().resolve()
    output = args.output or chain.parent/'plots/convergencia'
    output.mkdir(parents=True, exist_ok=True)
    with DESIGN.open() as f:
        design = list(csv.DictReader(f))
    names = [r['param'] for r in design]
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'axes.linewidth':.6})
    # Copy once to a temporary disk-backed array; preserve time and walker axes.
    with h5py.File(chain, 'r') as f, tempfile.TemporaryDirectory(prefix='mcmc-diagnostics-') as temp:
        d = f['chain']
        if d.ndim != 3 or d.shape[2] != len(names):
            parser.error('Expected (walkers, steps, 20 parameters) in the original design order.')
        walkers, total, ndim = d.shape
        if total - args.burnin < 16 and 0 <= args.burnin < total:
            snapshot = d[:, args.burnin:, :]
            fig, axes = grid(LABELS, 'Insufficient temporal samples — ensemble snapshot')
            for i, ax in enumerate(axes):
                ax.plot(np.arange(walkers), snapshot[:, -1, i], '.', color=CONTOUR, ms=3)
                ax.set_xlabel('Walker index (not time)')
            save(fig, output, 'ensemble_snapshot')
            summary = dict(chain=str(chain), shape=list(d.shape), burnin=args.burnin,
                status='insufficient_temporal_samples',
                reason='Fewer than 16 saved steps per walker; temporal convergence diagnostics not computed.')
            (output/'diagnostics.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(summary['reason'], flush=True)
            print(f'Saved snapshot: {output}', flush=True)
            return
        if not 0 <= args.burnin < total-15 or args.trace_walkers < 1 or args.display_points < 2:
            parser.error('Require at least 16 retained steps, positive trace-walkers and display-points>=2.')
        steps = total-args.burnin
        data = np.memmap(Path(temp)/'chain.dat', mode='w+', dtype='float64', shape=(walkers,steps,ndim))
        for start in range(0, steps, 512):
            block = d[:,args.burnin+start:args.burnin+min(steps,start+512),:]
            if not np.isfinite(block).all():
                parser.error('Chain contains nonfinite samples; diagnostics require uninterrupted trajectories.')
            data[:,start:start+block.shape[1],:] = block
        data.flush()
        print(f'{chain.name}: {walkers} walkers x {steps} retained steps', flush=True)
        figs = [grid(LABELS, title) for title in [
            'Walker traces — trajectories and central 60% / 90% intervals',
            'Time-window median and 60% / 90% intervals — check stationarity',
            'Autocorrelation — averaged over walkers',
            'Integrated autocorrelation time vs. retained chain length']]
        checkpoints = np.unique(np.linspace(max(16,steps//5), steps, 5, dtype=int))
        display_stride = max(1,int(np.ceil(steps/args.display_points)))
        selected_walkers = np.unique(np.linspace(0,walkers-1,min(walkers,args.trace_walkers),dtype=int))
        edges = np.unique(np.linspace(0,steps,min(40,steps//4)+1,dtype=int))
        records=[]
        for i,name in enumerate(names):
            x = np.array(data[:,:,i])
            ax=figs[0][1][i]
            t=np.arange(steps)[::display_stride]+args.burnin+1
            for w in selected_walkers:
                ax.plot(t,x[w,::display_stride],color=GREEN,alpha=.22,lw=.4,rasterized=True)
            interval_bands(ax, t, np.quantile(x[:,::display_stride], [.05,.20,.5,.80,.95], axis=0))
            ax.set_xlabel('Saved step')
            ax=figs[1][1][i]
            q=np.array([np.quantile(x[:,a:b],[.05,.20,.5,.80,.95]) for a,b in zip(edges[:-1],edges[1:])])
            centers=(edges[:-1]+edges[1:])/2+args.burnin
            interval_bands(ax, centers, q.T)
            ax.set_xlabel('Saved step (non-overlapping windows)')
            tau_values=[]
            for n in checkpoints:
                acf,tau,window,found=autocorrelation(x[:,:n])
                tau_values.append(tau)
            ax=figs[2][1][i]
            maxlag=min(steps//2, max(50,int(5*tau)) if np.isfinite(tau) else steps//2)
            ax.plot(np.arange(maxlag),acf[:maxlag],color=CONTOUR,lw=.9)
            ax.axhline(0,color=RED,ls='--',lw=.6)
            ax.set_xlabel('Lag (saved steps)')
            ax=figs[3][1][i]
            ax.plot(checkpoints,tau_values,'o-',color=CONTOUR,ms=3,lw=.9)
            ax.plot(checkpoints,checkpoints/50,color=RED,ls='--',lw=.8,label='N / 50')
            ax.set_xlabel('Retained steps per walker')
            ax.set_ylabel('Estimated tau')
            ratio=steps/tau if np.isfinite(tau) else np.nan
            relative=abs(tau_values[-1]-tau_values[-2])/abs(tau_values[-1]) if np.isfinite(tau) else np.nan
            reliable=bool(found and np.isfinite(ratio) and ratio>=50)
            record=dict(parameter=name,tau_steps=tau if np.isfinite(tau) else None,
                steps_over_tau=ratio if np.isfinite(ratio) else None,
                ess_approx=walkers*steps/tau if np.isfinite(tau) else None,
                window_lag=window,window_found=found,passes_50tau=reliable,
                relative_tau_change=relative if np.isfinite(relative) else None,
                tau_prefix_steps=checkpoints.tolist(),
                tau_prefix_values=[float(v) if np.isfinite(v) else None for v in tau_values])
            records.append(record)
            print(f'{i+1}/{ndim} {name}: tau={tau:.2f}, N/tau={ratio:.1f}, 50tau={reliable}',flush=True)
        for (fig,_),name in zip(figs,['walker_traces','window_stability','autocorrelation','tau_stability']):
            save(fig,output,name)
        summary=dict(chain=str(chain),shape=[walkers,total,ndim],burnin=args.burnin,
            retained_steps=steps,display_stride=display_stride,diagnostic_thinning=1,
            central_intervals=[.60,.90], interval_quantiles=[.05,.20,.5,.80,.95],
            method='Mean normalized walker FFT ACF; Sokal window c=5 (emcee approach)',
            notes=['Bands are empirical ensemble coverage intervals, not confidence intervals for the median.',
                   '50tau is a length heuristic, not proof of convergence.',
                   'ESS is approximate and assumes stationarity; cross-walker dependence can affect it.',
                   'No R-hat: ensemble walkers are not independent chains.',
                   'Tau is measured in saved steps; previous chain thinning is unknown.'],parameters=records)
        (output/'diagnostics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        fields=['parameter','tau_steps','steps_over_tau','ess_approx','window_found','passes_50tau','relative_tau_change']
        with (output/'diagnostics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
        print(f'Saved diagnostics: {output}',flush=True)
        del data


if __name__=='__main__':
    main()
