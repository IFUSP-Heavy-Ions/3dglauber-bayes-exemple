#!/usr/bin/env python3
"""Conditional trained-GP surfaces and full-calibration marginalized posterior.
No GP training, MCMC, optimization or writes to original analysis inputs.
Run --prepare-only, review outputs/selection_report.md, then --plot-only.
"""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
# Optional isolated local dependency location; normal environments need no override.
if os.environ.get('HIC_PYTHON_PACKAGES'):
    sys.path.insert(0,os.environ['HIC_PYTHON_PACKAGES'])
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'mcmc/scripts'),str(ROOT)]
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'logs/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(HERE/'logs/cache'))
import argparse,ast,csv,hashlib,json,platform,shutil,warnings
import re
import numpy as np
import pandas as pd
import h5py,dill,sklearn,scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits
from plot_posterior_corner import load_samples
from diagnose_phase2_numerical import load_emulator_module
from build_sector_studies import validate_reference

OUT=HERE/'outputs';INP=HERE/'inputs'
KEYS=['shear_viscosity_3_eta_over_s_at_kink','bulk_viscosity_3_zeta_over_s_max']
XLABEL=r'$(\eta/s)_{\mathrm{kink}}$';YLABEL=r'$(\zeta/s)_{\max}$'

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def copied_source(path):
    """Small immutable inputs copied; large original chain/emulators stay read-only."""
    target=INP/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
    return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),snapshot=str(target.relative_to(HERE)))

def data_and_covariance(config,emu,system):
    # Same source function as baseline, extracted to avoid import-time loads/writes.
    tree=ast.parse((ROOT/'bayes_mcmc.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_cov')
    namespace=dict(vars(config));exec(compile(ast.Module(body=[fn],type_ignores=[]),'bayes_mcmc.py','exec'),namespace)
    compute_cov=namespace['compute_cov']
    assert not config.run_closure and not config.assume_corr_exp_error
    assert not config.set_exp_error_to_zero and not config.change_exp_error
    assert not config.hold_parameters and not config.change_parameters_range
    records=[];source_files=[]
    for obs in config.active_obs_list[system]:
        sl=emu._slices[obs];edges=np.asarray(config.obs_cent_list[system][obs])
        assert sl.stop-sl.start==len(edges)
        path=ROOT/config.dir_obs_exp/system/config.expt_label_dict[obs[-4:]]/(obs+'.dat')
        frame=pd.read_csv(path,sep=' ',skiprows=2,escapechar='#').iloc[:len(edges)]
        assert len(frame)==len(edges) and np.isfinite(frame[['val','err']]).all().all()
        if (frame['err']<0).any():raise ValueError('Negative experimental variance')
        std=np.sqrt(frame['err'].to_numpy(float)) # bayes_exp.py: err column is VARIANCE
        covariance=compute_cov(obs,obs,std,std)
        # Only nonmultiplicity observables are candidates: no inverse-log approximation.
        if 'dN' not in obs and 'dET' not in obs:
            np.testing.assert_allclose(np.diag(covariance),frame['err'],rtol=1e-14)
            for k,(edge,val,sigma) in enumerate(zip(edges,frame['val'],std)):
                records.append(dict(observable=obs,bin_index=k,emulator_output_index=sl.start+k,
                    bin_low=float(edge[0]),bin_high=float(edge[1]),experiment=float(val),
                    experimental_std=float(sigma),file=str(path.relative_to(ROOT))))
        source_files.append(path)
    return records,source_files,compute_cov

def prepare(args):
    os.chdir(ROOT);OUT.mkdir(exist_ok=True);INP.mkdir(exist_ok=True)
    import configurations as config
    if config.idf!=0:raise ValueError('This saved baseline uses idf=0')
    designpath=Path(config.SystemsInfo[args.system]['main_range_file']).resolve()
    with designpath.open() as f:design=list(csv.DictReader(f))
    names=[d['param'] for d in design];ids=[names.index(k) for k in KEYS]
    ranges=np.asarray([[float(design[i]['min']),float(design[i]['max'])] for i in ids])
    diagnostic=ROOT/'mcmc/01_01_baseline_reproduction/plots/posterior_corner.json'
    defaults=json.loads(diagnostic.read_text())
    burnin=defaults['burnin'] if args.burnin is None else args.burnin
    thin=defaults['thin'] if args.thin is None else args.thin
    chainpath=(ROOT/args.chain).resolve()
    with h5py.File(chainpath,'r') as f:
        schema={k:dict(shape=list(v.shape),attrs={key:str(val) for key,val in v.attrs.items()}) for k,v in f.items() if isinstance(v,h5py.Dataset)}
        attrs={k:str(v) for k,v in f.attrs.items()}
        if f['chain'].shape[2]!=len(design):raise ValueError('Chain/design dimensional mismatch')
        # Use a saved MAP only if its semantics and coordinates are explicitly available.
        saved_map=None
        if 'map' in f and f['map'].shape==(len(names),):saved_map=np.asarray(f['map'])
    samples,meta=load_samples(chainpath,burnin,thin,200000)
    if meta['dropped_nonfinite']:
        raise ValueError('Chain contains nonfinite samples; inspect it before plotting')
    assert np.isfinite(samples).all()
    saved_medians=np.array([defaults['posterior'][name]['median'] for name in names])
    median_error=None
    if chainpath==(ROOT/'mcmc/01_01_baseline_reproduction/chain-idf-0.hdf') and burnin==defaults['burnin'] and meta['thin']==defaults['thin']:
        median_error=float(np.max(np.abs(np.median(samples,axis=0)-saved_medians)))
        np.testing.assert_allclose(np.median(samples,axis=0),saved_medians,rtol=1e-10,atol=1e-12,err_msg='Saved baseline summary and chain disagree')
    lo=np.array([float(d['min']) for d in design]);hi=np.array([float(d['max']) for d in design])
    if np.any(samples<lo) or np.any(samples>hi):raise ValueError('Samples outside configured priors')
    anchor=np.median(samples,axis=0) if saved_map is None else saved_map.copy()
    if not anchor[0]<=anchor[1]<=anchor[2]:raise ValueError('Representative point violates energy-loss ordering')
    module=load_emulator_module()
    path=ROOT/f'emulator/emulator-{args.system}-idf-0-npc-{config.SystemsInfo[args.system]["npc"]}.dill'
    with path.open('rb') as f:emu=dill.load(f)
    emu.__class__=module.Emulator # retain trained state; source transformations/predict/PCA
    if sklearn.__version__!='1.6.1':raise ValueError('Use sklearn 1.6.1, matching cached GP state')
    validated=validate_reference(emu,args.system)
    print('Historical emulator-mean validation passed for',validated,'observables',flush=True)
    records,files,_=data_and_covariance(config,emu,args.system)
    for obs in config.active_obs_list[args.system]:
        assert obs in emu._slices
    # Local uncertainty-weighted gradients plus a global coarse-grid coverage check.
    test=np.tile(anchor,(5,1));span=np.diff(ranges,axis=1).ravel()
    for j in range(2):
        test[1+2*j,ids[j]]-=0.02*span[j];test[2+2*j,ids[j]]+=0.02*span[j]
    if np.any(test<lo) or np.any(test>hi):raise ValueError('Derivative probes outside prior')
    pred,cov=emu.predict(test,return_cov=True,extra_std=np.array(0.))
    pilot_x,pilot_y=np.meshgrid(*[np.linspace(*r,13) for r in ranges])
    pilot=np.tile(anchor,(pilot_x.size,1));pilot[:,ids[0]]=pilot_x.ravel();pilot[:,ids[1]]=pilot_y.ravel()
    coarse=emu.predict(pilot)
    for record in records:
        obs=record['observable'];k=record['bin_index']
        record['gp_std_at_anchor']=float(np.sqrt(max(cov[obs,obs][0,k,k],0)))
        total=np.hypot(record['experimental_std'],record['gp_std_at_anchor'])
        g=np.array([pred[obs][2,k]-pred[obs][1,k],pred[obs][4,k]-pred[obs][3,k]])/0.04
        record['gradient_per_prior_span']=g.tolist();record['weighted_gradient']=(g/total).tolist()
        z=coarse[obs][:,k];record['pilot_mean_range']=[float(z.min()),float(z.max())]
        record['pilot_compatible_fraction']=float(np.mean(np.abs(z-record['experiment'])<=total))
    flows=[r for r in records if r['observable'].startswith(('v22','v32')) and r['pilot_compatible_fraction']>0]
    pts=[r for r in records if r['observable'].startswith('meanpT') and r['pilot_compatible_fraction']>0]
    if not flows or not pts:raise ValueError('No flow/mean-pT pair compatible on this conditional slice')
    pairs=[]
    for a in flows:
        for b in pts:
            g=np.array([a['weighted_gradient'],b['weighted_gradient']]);score=abs(np.linalg.det(g))
            if np.linalg.norm(g[0])*np.linalg.norm(g[1])>0:
                complement=score/(np.linalg.norm(g[0])*np.linalg.norm(g[1]))
                pairs.append((score,complement,a,b))
    score,complement,a,b=max(pairs,key=lambda x:x[0])
    selected=[a,b]
    for r in selected:
        obs=r['observable'];k=r['bin_index'];ref=ROOT/'mcmc/02_01_global_cs/plots/observables/chain-idf-0-cs-system-rho-1'/f'{args.system}_{obs}.csv'
        row=pd.read_csv(ref).iloc[k]
        np.testing.assert_allclose([r['experiment'],r['experimental_std'],r['bin_low'],r['bin_high']],row[['experiment','experimental_std','bin_low','bin_high']].to_numpy(float),rtol=1e-12)
    diagpath=ROOT/'mcmc/01_01_baseline_reproduction/plots/convergencia/diagnostics.json'
    diagnostics=json.loads(diagpath.read_text())
    convergence=[p for p in diagnostics['parameters'] if p['parameter'] in KEYS] if median_error is not None else []
    report=dict(parameters=[dict(index=i,**design[i]) for i in ids],system=args.system,
        chain=str(chainpath.relative_to(ROOT)),chain_size_bytes=chainpath.stat().st_size,baseline_saved_medians_max_abs_error=median_error,
        chain_mtime_ns=chainpath.stat().st_mtime_ns,chain_schema=schema,chain_attributes=attrs,sampling=meta,
        chain_processing='Project load_samples(): discard production steps then stride steps before flattening; burn-in saved separately by run_mcmc().',
        anchor_method='saved MAP' if saved_map is not None else 'coordinatewise posterior median (not MAP)',
        MAP_available=saved_map is not None,anchor=anchor.tolist(),all_parameter_names=names,
        observables=selected,selection_method='Maximize absolute determinant of uncertainty-weighted local gradients: one flow bin and one mean-pT bin, requiring coarse-grid measurement compatibility. Selected to illustrate complementarity, not discovery significance.',
        sensitivity_determinant=float(score),gradient_sine_angle=float(complement),
        historical_validation=dict(observable_groups=validated,tolerance_rtol=1e-6,tolerance_atol=1e-10,sklearn_cache_version='1.6.1'),
        experimental_covariance='Baseline: diagonal. The err input column is variance; std=sqrt(err), as in bayes_exp.py. No separate statistical/systematic decomposition supplied.',
        transformed_coordinates='Native Emulator.predict() applies transform_design(), source GP prediction and PCA/scaler inverse reconstruction. Selected outputs are non-logarithmic.',
        posterior_scope='Full saved calibration, all active observables and both systems Au-Au-200 and d-Au-200; 2D marginal over other 18 parameters, NOT a new two-observable posterior.',
        compatibility='Abs(GP mean - measurement) <= sqrt(experimental marginal variance + GP marginal variance). A pointwise 1-sigma diagnostic, NOT joint likelihood or posterior.',
        convergence_diagnostics=convergence,
        limitations=['No saved MAP or log-posterior values found; median used if MAP unavailable.','Thinning reduces display size, not proof of independent draws.','Published existing convergence diagnostics include finite-chain limitations; contours are estimates from saved samples.','Covariance off-diagonals within a single observed bin do not enter the diagnostic; full chain inference includes the baseline multi-bin likelihood.'],
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,dill=dill.__version__,matplotlib=matplotlib.__version__),
        input_snapshots=[copied_source(p) for p in [designpath,ROOT/'configurations.py',ROOT/'bins_and_cuts.py',ROOT/'bayes_mcmc.py',ROOT/'bayes_exp.py',ROOT/'emulator.py',ROOT/'mcmc/scripts/plot_posterior_corner.py',ROOT/'mcmc/scripts/diagnose_phase2_numerical.py',ROOT/'mcmc/scripts/build_sector_studies.py',diagnostic,diagpath]+[ROOT/r['file'] for r in selected]],
        original_emulator=dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    np.savez_compressed(INP/'posterior_inputs.npz',posterior_samples=samples,anchor=anchor,parameter_indices=ids,prior_ranges=ranges)
    report['posterior_snapshot_sha256']=sha(INP/'posterior_inputs.npz')
    write_json(OUT/'selection.json',report)
    write_json(OUT/'observable_sensitivity_scan.json',records)
    lines=['# Selection before plotting','',f'Chain: `{report["chain"]}`',f'Sampling: {meta}',f'Representative point: {report["anchor_method"]}','', 'Parameters:']
    for r in report['parameters']:lines.append(f'- {r["param"]}: [{r["min"]}, {r["max"]}], index {r["index"]}')
    lines+=['','Selected measurements:']
    for r in selected:lines.append(f'- {r["observable"]}, bin {r["bin_index"]} [{r["bin_low"]}, {r["bin_high"]}]: {r["experiment"]} +/- {r["experimental_std"]}; emulator output {r["emulator_output_index"]}')
    lines+=['',report['posterior_scope'],'',report['experimental_covariance'],'','MAP available: '+str(report['MAP_available']),'','Existing convergence diagnostics:']
    for d in convergence:lines.append(f'- {d["parameter"]}: tau={d.get("tau_steps")}, N/tau={d.get("steps_over_tau")}, passes 50tau={d.get("passes_50tau")}')
    (OUT/'selection_report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return emu,report

def grids(args,emu,report):
    snap=np.load(INP/'posterior_inputs.npz');anchor=snap['anchor'];ids=snap['parameter_indices'];ranges=snap['prior_ranges']
    x,y=np.meshgrid(*[np.linspace(*r,args.grid) for r in ranges])
    theta=np.tile(anchor,(x.size,1));theta[:,ids[0]]=x.ravel();theta[:,ids[1]]=y.ravel()
    means=[[],[]];variances=[[],[]];cross=[]
    selected=report['observables']
    for start in range(0,len(theta),args.batch):
        pred,cov=emu.predict(theta[start:start+args.batch],return_cov=True,extra_std=np.array(0.))
        for j,r in enumerate(selected):
            obs=r['observable'];k=r['bin_index'];means[j].extend(pred[obs][:,k]);variances[j].extend(cov[obs,obs][:,k,k])
        cross.extend(cov[selected[0]['observable'],selected[1]['observable']][:,selected[0]['bin_index'],selected[1]['bin_index']])
    arrays=dict(x=x,y=y,theta=theta,gp_mean=np.asarray(means).reshape(2,*x.shape),gp_variance=np.asarray(variances).reshape(2,*x.shape),gp_cross_covariance=np.asarray(cross).reshape(x.shape))
    for value in arrays.values():
        if not np.isfinite(value).all():raise ValueError('Nonfinite grid prediction')
    if arrays['gp_variance'].min() < -1e-10:raise ValueError('Negative GP variance')
    np.savez_compressed(OUT/'prediction_grid.npz',**arrays)
    return arrays

def make_figure(args,report,arrays):
    snap=np.load(INP/'posterior_inputs.npz');ranges=snap['prior_ranges'];ids=snap['parameter_indices'];samples=snap['posterior_samples'][:,ids];anchor=snap['anchor'][ids]
    x=arrays['x'];y=arrays['y'];obs=report['observables']
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix','font.size':12,'axes.titlesize':14,'axes.labelsize':15,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(16,9),facecolor='white')
    axes=[fig.add_axes([.005,.25,.285,.54],projection='3d'),fig.add_axes([.34,.25,.285,.54],projection='3d'),fig.add_axes([.745,.285,.23,.46])]
    gpcols=['#487e9b','#a77650'];compatible='#33845b';expcolor='#596573'
    titles=[]
    for j,ax in enumerate(axes[:2]):
        r=obs[j];name=r['observable'];z=arrays['gp_mean'][j];std=np.sqrt(np.maximum(arrays['gp_variance'][j],0));value=r['experiment'];expstd=r['experimental_std'];compatible_mask=np.abs(z-value)<=np.hypot(std,expstd)
        if 'meanpT_pi' in name:title=r'$\langle p_T\rangle_\pi$';zlab=r'$\langle p_T\rangle_\pi\;[\mathrm{GeV}/c]$'
        elif 'meanpT_k' in name:title=r'$\langle p_T\rangle_K$';zlab=r'$\langle p_T\rangle_K\;[\mathrm{GeV}/c]$'
        else:title=r'$v_2\{2\}$' if name.startswith('v22') else r'$v_3\{2\}$';zlab=title
        expt={'STAR':'STAR','PHEN':'PHENIX','PHOB':'PHOBOS','BRAH':'BRAHMS'}[name[-4:]]
        if '_int_' in name or name.startswith('meanpT'):bintext=f'{r["bin_low"]:g}–{r["bin_high"]:g}% centrality'
        elif '_pt_' in name:bintext=rf'$p_T={r["bin_low"]:g}$–${r["bin_high"]:g}$ GeV/$c$'
        else:bintext=rf'$\eta={r["bin_low"]:g}$–${r["bin_high"]:g}$'
        centrality=re.search(r'_cen_(\d+)_(\d+)_',name)
        if centrality and '_pt_' in name:
            bintext=f'{int(centrality[1])}–{int(centrality[2])}% centrality; '+bintext
        titles.append(f'{title}  |  {expt}\n{bintext}\n'+rf'Experiment: ${value:g}\pm{expstd:.2g}$')
        # Neutral experimental plane with upper/lower faces of the marginal 1-sigma slab.
        px,py=np.meshgrid(ranges[0],ranges[1]);flat=np.full_like(px,value)
        ax.plot_surface(px,py,flat,color=expcolor,alpha=.13,shade=False,linewidth=0)
        for sign in [-1,1]:
            ax.plot_surface(px,py,flat+sign*expstd,color=expcolor,alpha=.055,shade=False,linewidth=0)
        for xi,yi in [(0,0),(0,1),(1,0),(1,1)]:
            ax.plot([px[xi,yi]]*2,[py[xi,yi]]*2,[value-expstd,value+expstd],color=expcolor,alpha=.30,lw=.7)
        ax.plot_surface(x,y,z,rcount=args.grid,ccount=args.grid,color=gpcols[j],alpha=.83,linewidth=0,antialiased=True,rasterized=True)
        for sign in [-1,1]:ax.plot_wireframe(x,y,z+sign*std,rstride=max(1,args.grid//9),cstride=max(1,args.grid//9),color=gpcols[j],linewidth=.45,alpha=.30)
        ax.scatter(x[compatible_mask],y[compatible_mask],z[compatible_mask],s=3,color=compatible,alpha=.8,depthshade=False,rasterized=True)
        # Mean-surface/measurement intersection where it actually exists.
        if z.min()<value<z.max():ax.contour(x,y,z,levels=[value],colors=[compatible],linewidths=1.8)
        ax.view_init(elev=26,azim=-56);ax.set_box_aspect((1,1,.75));ax.set_xlim(*ranges[0]);ax.set_ylim(*ranges[1]);ax.set_xlabel(XLABEL,labelpad=9);ax.set_ylabel(YLABEL,labelpad=9);ax.set_zlabel(zlab,labelpad=7,fontsize=13)
        lims=[np.min(z-std),np.max(z+std),value-1.25*expstd,value+1.25*expstd];zmin,zmax=min(lims),max(lims);pad=.04*(zmax-zmin);ax.set_zlim(zmin-pad,zmax+pad)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(3));axis.pane.fill=False;axis._axinfo['grid'].update(color=(.8,.82,.84,.35),linewidth=.5)
        ax.tick_params(labelsize=10,pad=1);ax.zaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_title(titles[j],fontsize=12,pad=18,linespacing=1.5)
        report['observables'][j]['full_grid_compatible_fraction']=float(compatible_mask.mean())
    # Empirical marginalized density, smoothed in histogram bins, with HPD mass contours.
    ax=axes[2];hist,xe,ye=np.histogram2d(samples[:,0],samples[:,1],bins=100,range=ranges)
    density=gaussian_filter(hist.T,sigma=1.35,mode='reflect');density/=density.sum()*np.diff(xe)[0]*np.diff(ye)[0]
    ordered=np.sort(density.ravel())[::-1];mass=np.cumsum(ordered)/ordered.sum()
    thresholds={str(p):float(ordered[min(np.searchsorted(mass,p),len(ordered)-1)]) for p in [.68,.95]}
    cx=(xe[1:]+xe[:-1])/2;cy=(ye[1:]+ye[:-1])/2
    ax.pcolormesh(xe,ye,density,cmap='Blues',shading='flat',rasterized=True)
    levels=sorted(thresholds.values())
    cs=ax.contour(cx,cy,density,levels=levels,colors=['#466980','#113e61'],linewidths=[1.35,1.8]);ax.clabel(cs,fmt={thresholds['0.95']:'95%',thresholds['0.68']:'68%'},fontsize=11,inline=True)
    markerlabel='MAP' if report['MAP_available'] else 'Posterior median'
    ax.plot(*anchor,marker='*' if report['MAP_available'] else '+',color='#a4503a',ms=12,mew=1.8,label=markerlabel)
    ax.set(xlim=ranges[0],ylim=ranges[1],xlabel=XLABEL,ylabel=YLABEL);ax.set_title('Marginalized posterior\nAll calibration observables',fontsize=14,pad=14);ax.set_aspect('equal',adjustable='box');ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',fontsize=10,frameon=False)
    fig.suptitle('Bayesian inference of QGP transport parameters',x=.5,y=.955,fontsize=22,fontweight='medium')
    fig.text(.5,.897,rf'Au–Au, $\sqrt{{s_{{\mathrm{{NN}}}}}}=200$ GeV  |  Baseline calibration (Au–Au + d–Au)  |  Existing trained GP',ha='center',fontsize=12,color='#45515a')
    for x0,label in [(.02,'(a)'),(.355,'(b)'),(.71,'(c)')]:fig.text(x0,.81,label,fontsize=15,fontweight='bold')
    fig.legend(handles=[Patch(facecolor='#648da4',label='GP mean (a: blue; b: ochre)'),Line2D([],[],color='#648da4',lw=1,alpha=.5,label=r'GP $\pm1\sigma$ mesh'),Patch(facecolor=expcolor,alpha=.16,label=r'Experiment $\pm1\sigma$ slab'),Patch(facecolor=compatible,label=r'Compatible within $1\sigma_{\mathrm{exp+GP}}$')],loc='lower center',bbox_to_anchor=(.49,.155),ncol=4,fontsize=10,frameon=False,columnspacing=1.4)
    anchor_text='MAP' if report['MAP_available'] else 'posterior median'
    fig.text(.5,.115,f'(a–b) Conditional emulator slices: other 18 parameters fixed at the {anchor_text}.\n(c) Full MCMC marginal: the surfaces and their intersections are not a posterior.',ha='center',va='center',fontsize=11,color='#394854',linespacing=1.6)
    fig.text(.5,.059,f'Production chain: additional discard={report["sampling"]["burnin"]}, thinning={report["sampling"]["thin"]}, N={len(samples):,}.  HPD estimates: 68% / 95%.',ha='center',fontsize=10,color='#69737b')
    convergence=report['convergence_diagnostics']
    warning='Finite-chain caution: saved diagnostics fail the 50 autocorrelation-time criterion for both parameters.' if convergence and all(not d['passes_50tau'] for d in convergence) else 'Convergence is not certified by this visualization; inspect chain diagnostics before interpreting contours.'
    fig.text(.5,.027,warning,ha='center',fontsize=10,color='#8b593e')
    for ext in ['pdf','png']:fig.savefig(OUT/f'bayesian_inference_three_panel.{ext}',dpi=300)
    plt.close(fig)
    report['posterior_density']=dict(method='100x100 histogram; Gaussian smoothing 1.35 bins, reflect boundary; HPD thresholds by cumulative density mass',thresholds=thresholds,integral=float(density.sum()*np.diff(xe)[0]*np.diff(ye)[0]),mass_above_threshold={key:float(density[density>=val].sum()/density.sum()) for key,val in thresholds.items()})
    np.savez_compressed(OUT/'posterior_density.npz',xedges=xe,yedges=ye,density=density)
    report['figure']=dict(inches=[16,9],png_dpi=300,png_pixels=[4800,2700],conditional_grid=args.grid,experimental_band='marginal 1 sigma',GP_uncertainty='native marginal predictive variance including PCA truncation')
    write_json(OUT/'manifest.json',report)
    return report

def validate_outputs(report):
    from PIL import Image
    with Image.open(OUT/'bayesian_inference_three_panel.png') as image:
        if image.size!=(4800,2700) or not np.allclose(image.info['dpi'],[300,300],atol=.01):
            raise ValueError('PNG must be 4800x2700 at 300 dpi')
    density=report['posterior_density']
    np.testing.assert_allclose(density['integral'],1,atol=1e-12)
    for target,mass in density['mass_above_threshold'].items():
        if not 0<=mass-float(target)<.005:
            raise ValueError(f'HPD mass {mass} does not match target {target}')
    pdf=OUT/'bayesian_inference_three_panel.pdf'
    if not pdf.read_bytes().startswith(b'%PDF-'):
        raise ValueError('Invalid PDF export')
    report['output_validation']=dict(png_pixels=[4800,2700],png_dpi=300,
        density_normalized=True,HPD_mass_tolerance=.005,
        files={path.name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path in
               [pdf,OUT/'bayesian_inference_three_panel.png',OUT/'prediction_grid.npz',OUT/'posterior_density.npz']},
        plotting_script_sha256=sha(Path(__file__)))
    write_json(OUT/'manifest.json',report)
    print('Validated PNG dimensions/DPI, PDF signature, density normalization and 68%/95% HPD masses.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--chain',default='mcmc/01_01_baseline_reproduction/chain-idf-0.hdf');parser.add_argument('--system',default='Au-Au-200')
    parser.add_argument('--burnin',type=int);parser.add_argument('--thin',type=int);parser.add_argument('--grid',type=int,default=45);parser.add_argument('--batch',type=int,default=12)
    args=parser.parse_args()
    if args.grid<5 or args.batch<1:parser.error('grid >=5, batch >=1')
    if args.burnin is not None and args.burnin<0:parser.error('burnin must be nonnegative')
    if args.thin is not None and args.thin<1:parser.error('thin must be positive')
    if args.prepare_only and args.plot_only:parser.error('choose one phase')
    if args.plot_only and (args.burnin is not None or args.thin is not None):
        parser.error('Sampling changes require preparation, not --plot-only')
    os.chdir(ROOT)
    with threadpool_limits(limits=2):
        if args.plot_only:
            report=json.loads((OUT/'selection.json').read_text())
            if sklearn.__version__!=report['historical_validation']['sklearn_cache_version']:
                raise ValueError('Use sklearn 1.6.1, matching cached GP state')
            for source in report['input_snapshots']:
                if sha(ROOT/source['path'])!=source['sha256']:
                    raise ValueError(f'Input changed after preparation: {source["path"]}')
            module=load_emulator_module()
            with (ROOT/report['original_emulator']['path']).open('rb') as f:emu=dill.load(f)
            emu.__class__=module.Emulator
            if sha(ROOT/report['original_emulator']['path'])!=report['original_emulator']['sha256']:raise ValueError('Emulator changed after preparation')
            if sha(INP/'posterior_inputs.npz')!=report['posterior_snapshot_sha256']:raise ValueError('Posterior snapshot changed')
        else:emu,report=prepare(args)
        if args.prepare_only:return
        print((OUT/'selection_report.md').read_text(),flush=True)
        if not report['convergence_diagnostics'] or any(not d['passes_50tau'] for d in report['convergence_diagnostics']):
            warnings.warn('MCMC convergence is not certified: contours estimate the saved finite-chain distribution.',RuntimeWarning)
        arrays=grids(args,emu,report);make_figure(args,report,arrays);validate_outputs(report)
        print('Saved PDF and 300-dpi PNG in',OUT,flush=True)
if __name__=='__main__':main()
