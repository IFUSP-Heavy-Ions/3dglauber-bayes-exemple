"""Overlay reference and baseline posteriors and update the baseline notebook."""
from pathlib import Path
import csv,json,base64,uuid
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from plot_posterior_corner import load_samples,plot,LABELS,DESIGN,plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'01_01_baseline_reproduction/plots'

def main():
    baseline,bmeta=load_samples(ROOT/'01_01_baseline_reproduction/chain-idf-0.hdf',0,50,200000)
    reference,rmeta=load_samples(ROOT/'01_00_reference_posterior/posterior-chain-default.hdf',0,50,200000)
    with DESIGN.open() as f: design=list(csv.DictReader(f))
    assert baseline.shape[1]==reference.shape[1]==len(design)==20
    ranges=[(float(p['min']),float(p['max'])) for p in design]
    fig,bounds=plot(baseline,LABELS,ranges,.9,45,1.)
    axes=np.array(fig.axes).reshape(20,20)
    rbounds=np.quantile(reference,[.05,.5,.95],axis=0)
    for i in range(20):
        hist,edges=np.histogram(reference[:,i],bins=45,range=ranges[i],density=True)
        ax=axes[i,i]
        ax.stairs(hist,edges,color='#8546a8',linewidth=1.2)
        ax.set_ylim(0,max(ax.get_ylim()[1],hist.max()*1.12))
        for v in rbounds[[0,2],i]:ax.axvline(v,color='#8546a8',ls=':',lw=.8)
        for j in range(i):
            h,xe,ye=np.histogram2d(reference[:,j],reference[:,i],bins=45,range=[ranges[j],ranges[i]])
            density=gaussian_filter(h.T,1.,mode='reflect')
            ordered=np.sort(density.ravel())[::-1];mass=np.cumsum(ordered)/ordered.sum()
            levels=np.unique([ordered[min(np.searchsorted(mass,p),len(ordered)-1)] for p in [.9,.68]])
            levels=levels[(levels>density.min())&(levels<density.max())]
            if len(levels):axes[i,j].contour((xe[1:]+xe[:-1])/2,(ye[1:]+ye[:-1])/2,density,levels=levels,colors='#8546a8',linewidths=1.,linestyles='--')
    fig.legend(handles=[Line2D([],[],color='#71a962',label='Baseline reproduction'),Line2D([],[],color='#8546a8',ls='--',label='Reference posterior')],loc='upper right',bbox_to_anchor=(.88,.91),frameon=False,fontsize=22)
    fig.text(.58,.79,'Both chains: 68% and 90% joint contours\nDiagonal titles: baseline median and 90% interval\nDashed purple contours: reference',fontsize=16)
    stem=OUT/'reference_vs_baseline_corner'
    for ext in ['png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
    rows=[]
    for i,p in enumerate(design):
        lo=max(bounds[0,i],rbounds[0,i]);hi=min(bounds[2,i],rbounds[2,i])
        rows.append(dict(parameter=p['param'],baseline_median=float(bounds[1,i]),reference_median=float(rbounds[1,i]),baseline_90=bounds[[0,2],i].tolist(),reference_90=rbounds[[0,2],i].tolist(),intervals_overlap=bool(hi>=lo),median_shift_over_reference_90_width=float((bounds[1,i]-rbounds[1,i])/(rbounds[2,i]-rbounds[0,i]))))
    metadata=dict(baseline=bmeta,reference=rmeta,parameters=rows,contour_probabilities=[.68,.90],bins=45,smoothing_bins=1.,parameter_order='Assumed common order from design_ranges_main_AuAu-200.dat; HDF datasets do not independently establish parameter names.')
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    table='| Parameter | Reference median | Baseline median | Reference 90% interval | Baseline 90% interval |\n| --- | --- | --- | --- | --- |\n'
    for row in rows:
        fmt=lambda v:f'{v:.4g}'
        table+='| '+ ' | '.join([row['parameter'],fmt(row['reference_median']),fmt(row['baseline_median']),' – '.join(map(fmt,row['reference_90'])),' – '.join(map(fmt,row['baseline_90']))])+' |\n'
    count=sum(row['intervals_overlap'] for row in rows)
    text=f'''## Reference vs. baseline posterior comparison

The corner plot overlays all 20 parameters from the saved reference and baseline production chains. Green shows the baseline; dashed purple contours show the reference. The 2D contours enclose 68% and 90% probability, estimated with the same histogram bins and smoothing. Diagonal panels compare marginal densities; titles summarize the baseline median and central 90% interval.

Both chains are sampled with a step stride of at least 50 and a cap of approximately 200,000 samples each. No additional production burn-in is removed. The chain dimensions and retained sample counts are recorded in `plots/reference_vs_baseline_corner.json`. A common parameter order is assumed from the design-range file, as in the existing plots.

The marginal 90% intervals overlap for **{count} of 20 parameters**. The table below allows direct comparison of the medians and interval widths. Overlap is evidence of compatible marginal ranges; it does not alone prove that the joint posteriors are identical or that either chain is converged. Use the overlaid joint contours to assess differences in correlations and shape.

![Reference and baseline corner plot](attachment:reference_vs_baseline.png)

{table}

Recreate this comparison from the project directory with `python mcmc/scripts/compare_reference_baseline.py`. The PDF, PNG and numerical summary are saved in this study's `plots/` directory.
'''
    cell=dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(True),attachments={'reference_vs_baseline.png':{'image/png':base64.b64encode(stem.with_suffix('.png').read_bytes()).decode()}})
    path=OUT.parent/'baseline_reproduction.ipynb';nb=json.loads(path.read_text())
    for c in nb['cells']:
        if c['cell_type']=='markdown':
            c['source']=[s.replace('they do not include a numerical comparison against the paper, so agreement with the reference is not established here.','the reference-chain comparison below evaluates reproduction of the supplied reference posterior. A direct comparison with published paper values is not included.') for s in c['source']]
    nb['cells']=[c for c in nb['cells'] if not ''.join(c.get('source',[])).startswith('## Reference vs. baseline posterior comparison')]
    nb['cells'].insert(1,cell);path.write_text(json.dumps(nb,indent=1)+'\n')
    print('Overlapping intervals:',count,'/20; maximum absolute normalized median shift:',max(abs(r['median_shift_over_reference_90_width']) for r in rows),flush=True)
    print('Updated',path,flush=True)
if __name__=='__main__':main()
