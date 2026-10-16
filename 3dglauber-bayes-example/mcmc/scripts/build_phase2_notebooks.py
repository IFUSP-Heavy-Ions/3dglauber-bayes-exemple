"""Build publication notebooks from the saved Phase 2 figures and summaries."""
from pathlib import Path
import json,base64,uuid,csv
ROOT=Path(__file__).resolve().parents[1]
def md(t):return dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=t.splitlines(True))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def figure(p,d,caption):
 c=md('### '+p.stem.replace('_',' ')+'\n\n'+caption+'\n\n![Saved result](attachment:figure.png)\n\nSource: `'+str(p.relative_to(d))+'`.')
 c['attachments']={'figure.png':{'image/png':base64.b64encode(p.read_bytes()).decode()}}
 return c
def write(d,name,cells):
 nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},cells=cells)
 p=d/name
 if p.exists() and p.stat().st_size:
  previous=json.loads(p.read_text())
  numerical=[c for c in previous.get('cells',[]) if ''.join(c.get('source',[])).startswith('## Numerical covariance and uncertainty diagnostics')]
  nb['cells'][2:2]=numerical
 p.write_text(json.dumps(nb,indent=1)+'\n')
 check=json.loads(p.read_text());count=0
 for c in check['cells']:
  for a in c.get('attachments',{}).values():assert base64.b64decode(a['image/png']).startswith(b'\x89PNG');count+=1
 print(p.name,len(cells),'cells;',count,'embedded figures;',p.stat().st_size,'bytes')
model='''## Correlation model and run

The experimental covariance uses a compound-symmetry (CS) correlation structure with **ρ = 1**. All experimental bins within a collision system share the same correlation block, across observables and detectors. Au–Au and d–Au are separate blocks. In likelihood coordinates, Cᵢⱼ = σᵢσⱼ at this value of ρ. For multiplicities, σ is converted to the logarithmic coordinate convention used by the likelihood.

This is the fully correlated limiting case, not an inferred correlation coefficient. Its experimental covariance block is rank one; the likelihood also includes emulator covariance. The name “global” refers to the scope within each system, not a correlation across both collision systems.

The saved production chain has 200 walkers, 50,000 steps per walker, and 20 parameters. Corner summaries retain 200,000 samples with stride 50 and no additional production burn-in removal. These reports summarize saved outputs; they do not establish chain convergence or rerun the fit.
'''
bands='''## Observable predictions: what the bands mean

The saved observable figures use **250 posterior parameter samples**, seed 12345, and show the median and the 16th–84th percentiles (central **68%** interval). These bands describe the posterior variation of emulator means. They do **not** include emulator-noise draws or experimental-noise draws, so they are not full predictive uncertainty intervals for a new measurement.

Experimental points have their saved error bars. The lower panel shows model/data: unity means the median prediction matches the measured value. Similar curves can arise from different parameter posteriors, so assess parameter shifts as well as observable agreement. No goodness-of-fit statistic is inferred from visual overlap.
'''
d=ROOT/'02_01_global_cs';summary=json.loads((d/'plots/posterior_corner.json').read_text());post=summary['posterior']
cells=[md('# Phase 2 · Global Compound-Symmetry Correlations\n\nThis study examines the posterior obtained when experimental errors are fully correlated within each collision system. It collects the parameter posterior and all saved observable figures for Au–Au and d–Au at 200 GeV.'),md(model),md('## Parameter posterior\n\nDiagonal corner panels show one-dimensional parameter densities. Lower panels show parameter-pair distributions, with 68% and 90% joint contours. The table lists medians and central 90% credible intervals. Broad ranges indicate remaining parameter uncertainty under this covariance assumption.\n\n'+table(['Parameter','Median','90% lower','90% upper'],[(k,*[f'{v[x]:.5g}' for x in ['median','ci_low','ci_high']]) for k,v in post.items()])),figure(d/'plots/posterior_corner.png',d,'Posterior of the 20 model parameters under global CS experimental correlations (ρ = 1).'),md(bands)]
obs=d/'plots/observables/chain-idf-0-cs-system-rho-1'
for system in ['Au-Au-200','d-Au-200']:
 cells.append(md('## '+system+' observables\n\nThe individual figures below cover all saved observables for this collision system. Centrality, detector and observable are identified in each figure.'))
 for p in sorted(obs.glob(system+'_*.png')):
  if not p.name.startswith('._'):cells.append(figure(p,d,'Orange shows the CS posterior median and central 68% band of emulator means. Black points show the experiment. The ratio panel makes relative offsets visible across bins.'))
cells.append(md('## Combined observable pages\n\nThese pages repeat the individual results in compact layouts for comparison across observables.'))
for p in sorted(obs.glob('posterior_observables_*page*.png')):cells.append(figure(p,d,'Combined view of the same saved CS observable predictions. Bands are central 68% intervals of emulator means.'))
cells.append(md('## Source and interpretation\n\nThe numerical posterior summary is `plots/posterior_corner.json`; observable CSVs, `posterior_inputs.npz`, and `manifest.json` are stored alongside the observable figures. Historical paths inside the manifest refer to earlier folder names. All displayed images are embedded in this notebook.\n\nThis run represents one strong correlation assumption. Compare it with the baseline in study 02_03 to assess sensitivity; it does not establish that ρ = 1 is favored by the data.'))
write(d,'cs_correlation_model.ipynb',cells)
d=ROOT/'02_03_baseline_vs_global_cs';summary=json.loads((d/'posterior_corner_comparison.json').read_text());cs=summary['posterior'];baseline=summary['comparison']['posterior'];rows=[]
for k,v in cs.items():
 b=baseline[k];bw=b['ci_high']-b['ci_low'];cw=v['ci_high']-v['ci_low'];delta=v['median']-b['median']
 rows.append(dict(parameter=k,baseline_median=b['median'],cs_median=v['median'],delta=delta,normalized_shift=delta/bw,width_ratio=cw/bw,baseline_interval=[b['ci_low'],b['ci_high']],cs_interval=[v['ci_low'],v['ci_high']]))
ranked=sorted(rows,key=lambda r:abs(r['normalized_shift']),reverse=True)
intro='## Main changes in the saved posterior\n\nThe largest median shifts relative to the baseline 90% interval width are:\n\n'
for r in ranked[:5]:intro+=f"- `{r['parameter']}`: median {r['baseline_median']:.4g} → {r['cs_median']:.4g}; shift / baseline interval width = {r['normalized_shift']:+.3f}; CS/baseline interval-width ratio = {r['width_ratio']:.3f}.\n"
intro+='\nThese are descriptive sensitivity measures, not significance tests. A width ratio above one means a broader CS marginal interval; below one means a narrower interval. Parameter shifts can coexist with similar observable curves.\n\n'
intro+=table(['Parameter','Baseline median','CS median','Δ median','Δ / baseline 90% width','CS/baseline 90% width'],[(r['parameter'],f"{r['baseline_median']:.5g}",f"{r['cs_median']:.5g}",f"{r['delta']:+.4g}",f"{r['normalized_shift']:+.3f}",f"{r['width_ratio']:.3f}") for r in rows])
(rootout:=d/'posterior_sensitivity_summary.json').write_text(json.dumps({'definition':'delta = CS median - baseline median; normalized shift uses baseline central 90% interval width; width ratio = CS width / baseline width','parameters':rows},indent=2)+'\n')
cells=[md('# Phase 2 · Baseline vs. Global CS\n\nThis study compares the saved baseline posterior with the posterior obtained using fully correlated experimental errors within each collision system. It reports parameter shifts, changes in marginal credible intervals, and observable predictions.'),md(model),md(intro),md('## Marginal credible intervals\n\n'+table(['Parameter','Baseline 90% interval','CS 90% interval'],[(r['parameter'],' – '.join(f'{v:.5g}' for v in r['baseline_interval']),' – '.join(f'{v:.5g}' for v in r['cs_interval'])) for r in rows])),figure(d/'posterior_corner_comparison.png',d,'In this saved figure, blue shows CS and pink shows baseline **marginal densities only**. Titles and all two-dimensional panels belong to CS. The figure does not overlay baseline joint contours and therefore cannot establish changes in pairwise correlations.'),md(bands+'\nIn the observable comparisons, **orange is CS** and **blue (`my_simulation`) is the baseline**. Each uses 250 posterior parameter samples. The legacy label `my_simulation` refers to study 01_01_baseline_reproduction, not to another correlation model.')]
obs=d/'observables/chain-idf-0-cs-system-rho-1'
for system in ['Au-Au-200','d-Au-200']:
 cells.append(md('## '+system+' observable comparisons'))
 for p in sorted(obs.glob(system+'_*.png')):
  if not p.name.startswith('._'):cells.append(figure(p,d,'Orange: CS (ρ = 1). Blue: baseline (`my_simulation`). Shading shows the central 68% posterior variation of emulator means. Black points show measurements. Compare the model/data curves to locate changes across bins.'))
cells.append(md('## Combined comparison pages\n\nThese pages repeat the individual comparisons in compact layouts.'))
for p in sorted(obs.glob('posterior_observables_*page*.png')):cells.append(figure(p,d,'Combined baseline and CS observable comparisons with central 68% bands of emulator means.'))
cells.append(md('## Interpretation and sources\n\nThe parameter summaries show how the posterior changes under this correlation assumption. The observable figures show whether those changes visibly alter the predicted curves. Neither similar curves nor overlapping credible intervals establish that the covariance models are equivalent or identify a preferred model. Pairwise correlation changes require a comparison of both joint posteriors, which is not provided by the saved marginal-overlay corner figure.\n\nNumerical sources: `posterior_corner_comparison.json`, `posterior_sensitivity_summary.json`, and the observable CSVs and manifest in `observables/chain-idf-0-cs-system-rho-1/`. The manifest retains historical directory names. All displayed figures are embedded, so they remain available when the notebook is published.'))
write(d,'comparison_global_correlation_models.ipynb',cells)
