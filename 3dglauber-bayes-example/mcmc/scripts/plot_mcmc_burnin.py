#!/usr/bin/env python3
"""Plot the burnin HDF5 dataset separately from the production chain."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from plot_posterior_corner import DESIGN, LABELS, GREEN, CONTOUR, RED
from plot_mcmc_convergence import grid, save, interval_bands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain',type=Path,required=True,help='HDF5 file containing burnin')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--trace-walkers',type=int,default=20)
    parser.add_argument('--display-points',type=int,default=1500)
    parser.add_argument('--resampling-step',type=int,help='Saved-step boundary where walkers were resampled')
    args=parser.parse_args()
    path=args.chain.expanduser().resolve()
    output=args.output or path.parent/'plots/convergencia'
    with h5py.File(path,'r') as f:
        if 'burnin' not in f: parser.error('HDF5 file has no burnin dataset.')
        data=f['burnin'][:]
    if data.ndim!=3 or data.shape[2]!=len(LABELS) or data.shape[1]<2:
        parser.error('Expected burnin shaped (walkers, steps, 20 parameters), with at least two steps.')
    if not np.isfinite(data).all(): parser.error('Burn-in contains nonfinite samples.')
    walkers,steps,ndim=data.shape
    if args.trace_walkers<1 or args.display_points<2: parser.error('Invalid display settings.')
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','axes.linewidth':.6})
    trace,axes=grid(LABELS,'Burn-in — trajectories and central 60% / 90% intervals')
    stability,window_axes=grid(LABELS,'Burn-in — time-window median and 60% / 90% intervals')
    stride=max(1,int(np.ceil(steps/args.display_points)))
    selected=np.unique(np.linspace(0,walkers-1,min(walkers,args.trace_walkers),dtype=int))
    t=np.arange(steps)[::stride]+1
    edges=np.unique(np.linspace(0,steps,min(40,steps)+1,dtype=int))
    with DESIGN.open() as f: names=[r['param'] for r in csv.DictReader(f)]
    records=[]
    for i in range(ndim):
        x=data[:,:,i]
        for w in selected: axes[i].plot(t,x[w,::stride],color=GREEN,lw=.4,alpha=.22,rasterized=True)
        interval_bands(axes[i], t, np.quantile(x[:,::stride], [.05,.20,.5,.80,.95], axis=0))
        axes[i].set_xlabel('Burn-in step')
        q=np.array([np.quantile(x[:,a:b],[.05,.20,.5,.80,.95]) for a,b in zip(edges[:-1],edges[1:])])
        center=(edges[:-1]+edges[1:])/2
        interval_bands(window_axes[i], center, q.T)
        window_axes[i].set_xlabel('Burn-in step (time windows)')
        if args.resampling_step is not None:
            for ax in [axes[i],window_axes[i]]:
                ax.axvline(args.resampling_step,color='#777777',ls='--',lw=.7)
        records.append(dict(parameter=names[i],initial_median=float(np.median(x[:,0])),
                            final_median=float(np.median(x[:,-1]))))
    save(trace,output,'burnin_traces')
    save(stability,output,'burnin_window_stability')
    summary=dict(chain=str(path),dataset='burnin',shape=list(data.shape),display_stride=stride,
        resampling_step=args.resampling_step,parameters=records,
        central_intervals=[.60,.90], interval_quantiles=[.05,.20,.5,.80,.95],
        notes=['Burn-in is transient; no stationary autocorrelation time or ESS is estimated.',
               'Resampling can break continuity of individual walker identities at the marked boundary.'])
    (output/'burnin_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Saved burn-in plots: {output}',flush=True)


if __name__=='__main__':main()
