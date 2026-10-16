#!/usr/bin/env python3
"""Numerical baseline/CS diagnostics. No MCMC, fitting, plots, or cache writes.

Adapted from diagnose_covariance_models.py and
 diagnose_uncertainty_correlation.py in the AllDetectorsAuAu200 study.
"""
from pathlib import Path
import argparse,ast,base64,csv,hashlib,json,sys,types,uuid,platform
import numpy as np
import h5py
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
MCMC=ROOT/'mcmc'
STUDY=MCMC/'02_03_baseline_vs_global_cs'
OBS=STUDY/'observables/chain-idf-0-cs-system-rho-1'
DESIGN=ROOT/'production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production/design_ranges_main_AuAu-200.dat'
CHAINS={'baseline':MCMC/'01_01_baseline_reproduction/chain-idf-0.hdf','cs':MCMC/'02_01_global_cs/chain-idf-0-cs-system-rho-1.hdf'}

def load_samples(path,burnin,thin,max_samples):
    with h5py.File(path) as f:
        d=f['chain'];walkers,steps,ndim=d.shape
        stride=max(thin,int(np.ceil(walkers*(steps-burnin)/max_samples)))
        blocks=[d[:,start:min(start+256*stride,steps):stride,:].reshape(-1,ndim) for start in range(burnin,steps,256*stride)]
        samples=np.concatenate(blocks)
    finite=np.isfinite(samples).all(axis=1)
    if not finite.all():raise ValueError('Nonfinite posterior samples found; inspect the chain.')
    print(path.parent.name,walkers,steps,ndim,'retained',len(samples),flush=True)
    return samples,dict(walkers=walkers,steps=steps,burnin=burnin,thin=stride,samples=len(samples),dropped_nonfinite=0)

def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return dict(n_valid=0,mean=None,median=None,min=None,max=None)
    return dict(n_valid=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))

def divide(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b!=0)

def matrix_stats(c):
    eig=np.linalg.eigvalsh(c);tol=len(eig)*np.finfo(float).eps*max(abs(eig).max(),np.finfo(float).tiny)
    positive=eig>tol;rank=int(positive.sum());diag=np.diag(c)
    corr=divide(c,np.sqrt(np.outer(diag,diag)));off=corr[~np.eye(len(c),dtype=bool)]
    return dict(n_bins=len(c),trace=float(np.trace(c)),diag_mean=float(diag.mean()),frobenius=float(np.linalg.norm(c)),min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1]),rank=rank,eigenvalue_tolerance=float(tol),positive_semidefinite=bool(eig[0]>=-tol),positive_definite=bool(rank==len(c)),condition_number=float(eig[-1]/eig[0]) if rank==len(c) else None,logdet=float(np.log(eig).sum()) if rank==len(c) else None,symmetry_error=float(abs(c-c.T).max()),mean_abs_offdiag_correlation=float(np.nanmean(abs(off))),max_abs_offdiag_correlation=float(np.nanmax(abs(off))))

def gaussian_terms(residual,c):
    cf=cho_factor(c,lower=True,check_finite=True)
    chi=float(residual@cho_solve(cf,residual));ld=float(2*np.log(np.diag(cf[0])).sum())
    return dict(chi2=chi,chi2_per_bin=chi/len(residual),logdet=ld,gaussian_loglike=-.5*(chi+ld+len(residual)*np.log(2*np.pi)))

def load_emulator_module():
    # Import only source imports/classes: no training, calculation loaders or
    # module-level dill loading. Rebind loaded state to current source methods
    # so serialized Python bytecode is not used across Python versions.
    source=ast.parse((ROOT/'emulator.py').read_text())
    body=[n for n in source.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef)) and not (isinstance(n,ast.ImportFrom) and n.module=='calculations_load') and not (isinstance(n,ast.Import) and any(a.name=='matplotlib.pyplot' for a in n.names))]
    m=types.ModuleType('emulator');m.__file__=str(ROOT/'emulator.py');sys.modules['emulator']=m
    exec(compile(ast.Module(body=body,type_ignores=[]),m.__file__,'exec'),m.__dict__)
    assert m.do_transform_design and m.transform_multiplicities
    return m

def posterior_diagnostics(out):
    with DESIGN.open() as f:design=list(csv.DictReader(f))
    samples={};inventory=[];summaries=[];windows=[]
    for run,path in CHAINS.items():
        a,meta=load_samples(path,0,50,200000)
        assert a.shape[1]==len(design)==20
        samples[run]=a;inventory.append(dict(run=run,chain=str(path.relative_to(ROOT)),**meta))
        q=np.quantile(a,[.05,.16,.2,.5,.8,.84,.95],axis=0)
        for i,p in enumerate(design):
            assert np.all((a[:,i]>=float(p['min']))&(a[:,i]<=float(p['max'])))
            summaries.append(dict(run=run,parameter=p['param'],mean=float(a[:,i].mean()),std=float(a[:,i].std(ddof=1)),q05=float(q[0,i]),q16=float(q[1,i]),q20=float(q[2,i]),median=float(q[3,i]),q80=float(q[4,i]),q84=float(q[5,i]),q95=float(q[6,i]),n_samples=len(a)))
        with h5py.File(path) as f:
            steps=f['chain'].shape[1]
            for start in range(0,steps,steps//10):
                block=f['chain'][:,start:min(start+steps//10,steps):50,:].reshape(-1,20)
                quant=np.quantile(block,[.05,.2,.5,.8,.95],axis=0)
                for i,p in enumerate(design):windows.append(dict(run=run,parameter=p['param'],start_saved_step=start,end_saved_step_exclusive=min(start+steps//10,steps),q05=float(quant[0,i]),q20=float(quant[1,i]),median=float(quant[2,i]),q80=float(quant[3,i]),q95=float(quant[4,i])))
    sensitivity=[]
    for i,p in enumerate(design):
        a=samples['baseline'][:,i];b=samples['cs'][:,i]
        ha,edges=np.histogram(a,bins=80,range=(float(p['min']),float(p['max'])))
        hb,_=np.histogram(b,bins=edges);pa=ha+1e-12;pb=hb+1e-12;pa/=pa.sum();pb/=pb.sum()
        qa=np.quantile(a,[.05,.16,.5,.84,.95]);qb=np.quantile(b,[.05,.16,.5,.84,.95])
        sensitivity.append(dict(parameter=p['param'],js_divergence_bits=float(jensenshannon(pa,pb,base=2)**2),wasserstein=float(wasserstein_distance(a,b)),median_shift=float(qb[2]-qa[2]),normalized_median_shift=float((qb[2]-qa[2])/np.sqrt(a.var(ddof=1)+b.var(ddof=1))),width_ratio_68=float((qb[3]-qb[1])/(qa[3]-qa[1])),width_ratio_90=float((qb[4]-qb[0])/(qa[4]-qa[0]))))
    correlations=[]
    ca=np.corrcoef(samples['baseline'],rowvar=False);cb=np.corrcoef(samples['cs'],rowvar=False)
    for i in range(20):
        for j in range(i):correlations.append(dict(parameter_1=design[j]['param'],parameter_2=design[i]['param'],baseline_pearson=float(ca[i,j]),cs_pearson=float(cb[i,j]),delta_pearson=float(cb[i,j]-ca[i,j])))
    save_csv(out/'run_inventory.csv',inventory);save_csv(out/'posterior_parameter_summary.csv',summaries)
    save_csv(out/'posterior_sensitivity.csv',sensitivity);save_csv(out/'parameter_sensitivity_ranking.csv',sorted(sensitivity,key=lambda r:r['js_divergence_bits'],reverse=True))
    save_csv(out/'parameter_pair_correlations.csv',correlations);save_csv(out/'time_window_intervals.csv',windows)
    return inventory,sensitivity,correlations

def predictive_diagnostics(out,batch_size):
    module=load_emulator_module()
    manifest=json.loads((OBS/'manifest.json').read_text());assert manifest['chain_attributes']['experimental_cs_rho']==1
    theta={};sources={}
    for run,name in [('baseline','my_simulation_inputs.npz'),('cs','posterior_inputs.npz')]:
        with np.load(OBS/name) as f:theta[run]=f['theta'].copy();indices=f['walker_step_indices'].copy()
        with h5py.File(CHAINS[run]) as f:
            for x,(w,s) in zip(theta[run],indices):np.testing.assert_allclose(x,f['chain'][int(w),int(s)],rtol=0,atol=0)
        sources[run]=dict(n_samples=len(theta[run]),input_file=str((OBS/name).relative_to(ROOT)),sha256=hashlib.sha256((OBS/name).read_bytes()).hexdigest())
    covariance=[];differences=[];perbin=[];uncertainty=[];perobs=[];likelihood=[];cross=[];validation=[];emu_sources=[]
    for system,observables in manifest['observables'].items():
        print('Predicting',system,flush=True)
        path=ROOT/f'emulator/emulator-{system}-idf-0-npc-10.dill'
        import dill
        with path.open('rb') as f:emu=dill.load(f)
        emu.__class__=module.Emulator
        emu_sources.append(dict(system=system,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert set(observables)==set(emu.observables)
        metadata=[];experiment=[];experimental_std=[];multiplicity=[];offset=0;records={}
        for obs in observables:
            with (OBS/f'{system}_{obs}.csv').open() as f:rows=list(csv.DictReader(f))
            records[obs]=rows
            assert len(rows)==emu._slices[obs].stop-emu._slices[obs].start
            for i,row in enumerate(rows):
                metadata.append(dict(system=system,observable=obs,bin_index=i))
                experiment.append(float(row['experiment']));experimental_std.append(float(row['experimental_std']));multiplicity.append(('dN' in obs) or ('dET' in obs))
        experiment=np.array(experiment);experimental_std=np.array(experimental_std);mult=np.array(multiplicity)
        y=experiment.copy();y[mult]=np.log(y[mult])
        sig=experimental_std.copy();sig[mult]=np.sqrt(np.log1p((sig[mult]/experiment[mult])**2))
        assert np.isfinite(y).all() and np.all(sig>0)
        ce={'baseline':np.diag(sig**2),'cs':np.outer(sig,sig)}
        np.testing.assert_allclose(np.diag(ce['baseline']),np.diag(ce['cs']))
        for run in ce:covariance.append(dict(system=system,component='experimental',run=run,**matrix_stats(ce[run])))
        diff=ce['cs']-ce['baseline']
        differences.append(dict(system=system,relative_frobenius=float(np.linalg.norm(diff)/np.linalg.norm(ce['baseline'])),max_abs_difference=float(abs(diff).max()),trace_difference=float(np.trace(diff))))
        for run in ['baseline','cs']:
            means=[];physical=[];sum_cov=np.zeros_like(ce[run]);cross_rows=[]
            for start in range(0,len(theta[run]),batch_size):
                x=theta[run][start:start+batch_size]
                mean,cov=emu.predict(x,return_cov=True,extra_std=np.array(0.))
                mu=np.concatenate([mean[obs] for obs in observables],axis=1)
                # Use the same explicit observable order for means and covariances.
                order=np.concatenate([np.arange(emu._slices[obs].start,emu._slices[obs].stop) for obs in observables])
                c=cov.array[:,order][:,:,order]
                assert np.isfinite(mu).all() and np.isfinite(c).all()
                ph=mu.copy();ph[:,mult]=np.exp(ph[:,mult]);means.append(mu);physical.append(ph);sum_cov+=c.sum(axis=0)
                for k in range(len(x)):
                    for model in ce:
                        terms=gaussian_terms(y-mu[k],ce[model]+c[k])
                        row=dict(system=system,posterior_source=run,evaluated_covariance=model,sample_index=start+k,**terms)
                        cross.append(row);cross_rows.append(row)
            mu=np.concatenate(means);ph=np.concatenate(physical);avg_cov=sum_cov/len(mu)
            covariance.append(dict(system=system,component='mean_emulator',run=run,**matrix_stats(avg_cov)))
            total=ce[run]+avg_cov
            covariance.append(dict(system=system,component='experimental_plus_mean_emulator',run=run,**matrix_stats(total)))
            residual=y-np.median(mu,axis=0)
            terms=gaussian_terms(residual,total)
            likelihood.append(dict(system=system,run=run,definition='coordinatewise posterior median prediction; mean emulator covariance',n_prediction_samples=len(mu),**terms))
            q16,q50,q84=np.quantile(ph,[.16,.5,.84],axis=0);std=ph.std(axis=0,ddof=1)
            ratio=divide(std,experimental_std);cvpost=divide(std,abs(q50));cvexp=divide(experimental_std,abs(experiment));cvratio=divide(cvpost,cvexp)
            valid=ratio[np.isfinite(ratio)]
            uncertainty.append(dict(system=system,run=run,**{f'ratio_{k}':v for k,v in stats(ratio).items()},fraction_posterior_std_below_exp=float(np.mean(valid<1)),**{f'cv_ratio_{k}':v for k,v in stats(cvratio).items()}))
            offset=0
            for obs in observables:
                n=len(records[obs]);sl=slice(offset,offset+n)
                perobs.append(dict(system=system,run=run,observable=obs,n_bins=n,median_std_ratio=float(np.nanmedian(ratio[sl])),min_std_ratio=float(np.nanmin(ratio[sl])),max_std_ratio=float(np.nanmax(ratio[sl])),fraction_posterior_std_below_exp=float(np.mean(ratio[sl]<1)),median_cv_ratio=float(np.nanmedian(cvratio[sl]))))
                saved=np.array([[float(r[k]) for k in (['q16','median','q84'] if run=='cs' else ['my_simulation_q16','my_simulation_median','my_simulation_q84'])] for r in records[obs]])
                calculated=np.stack([q16[sl],q50[sl],q84[sl]],axis=1)
                error=float(np.max(abs(calculated-saved)/np.maximum(abs(saved),1e-12)))
                validation.append(dict(system=system,run=run,observable=obs,max_relative_error_saved_quantiles=error,matches_saved_quantiles=bool(np.allclose(calculated,saved,rtol=1e-6,atol=1e-10))))
                offset+=n
            for i,meta in enumerate(metadata):
                perbin.append(dict(**meta,run=run,experiment=float(experiment[i]),experimental_std=float(experimental_std[i]),posterior_q16=float(q16[i]),posterior_median=float(q50[i]),posterior_q84=float(q84[i]),posterior_std=float(std[i]),posterior_over_experimental_std=float(ratio[i]),posterior_cv=float(cvpost[i]),experimental_cv=float(cvexp[i]),posterior_cv_over_experimental_cv=float(cvratio[i]),likelihood_residual=float(residual[i]),residual_over_total_diag_std=float(residual[i]/np.sqrt(total[i,i])),emulator_variance=float(avg_cov[i,i]),experimental_variance=float(ce[run][i,i])) )
            np.savez_compressed(out/f'{system}_{run}_diagnostic_arrays.npz',prediction_means_likelihood=mu,prediction_means_physical=ph,mean_emulator_covariance=avg_cov,experimental_covariance=ce[run],experiment_likelihood=y,experimental_std_physical=experimental_std,observable=np.array([m['observable'] for m in metadata]),bin_index=np.array([m['bin_index'] for m in metadata]))
            print(system,run,'done; saved quantile matches',sum(r['matches_saved_quantiles'] for r in validation if r['system']==system and r['run']==run),'/',len(observables),flush=True)
    for name,rows in [('covariance_metrics',covariance),('covariance_vs_baseline',differences),('posterior_predictive_summary',likelihood),('uncertainty_summary',uncertainty),('uncertainty_by_observable',perobs),('posterior_predictive_per_bin',perbin),('cross_covariance_likelihoods',cross),('saved_quantile_validation',validation)]:save_csv(out/(name+'.csv'),rows)
    return dict(covariance=covariance,differences=differences,likelihood=likelihood,uncertainty=uncertainty,per_observable=perobs,validation=validation,prediction_sources=sources,emulator_sources=emu_sources)

def fmt(v):
    if v is None:return 'singular / undefined'
    if isinstance(v,(float,np.floating)):return f'{v:.5g}'
    return str(v)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(fmt,row))+' |' for row in rows]])
def update_notebook(data,sensitivity,correlations,out):
    rank=sorted(sensitivity,key=lambda r:r['js_divergence_bits'],reverse=True)
    section='''## Numerical covariance and uncertainty diagnostics

This section adapts the numerical analyses in `diagnose_covariance_models.py` and `diagnose_uncertainty_correlation.py` from the hic-bayes AllDetectorsAuAu200 study to the saved **joint Au–Au + d–Au** baseline and CS runs. The parameters are shared by the joint fit; parameter sensitivity is therefore reported once, not separately for each system. All new outputs are numerical tables, with full CSVs in `numerical_diagnostics/`.

### Definitions and reproducibility

Posterior comparisons use up to 200,000 samples per chain (saved-step stride 50, no extra burn-in). Jensen–Shannon divergence uses 80 shared bins over the prior range and a 10⁻¹² count pseudocount, as in the original script. It is reported in bits (0 means identical binned marginals; 1 is maximal separation). Wasserstein distance is in parameter units. The normalized median shift is (CS median − baseline median) / √(s²_baseline + s²_CS); this differs from the earlier normalization by the baseline 90% interval width. Width ratios use central 68% intervals.

Prediction diagnostics reuse the exact 250 parameter vectors per chain stored with the observable plots. Walker/step indices are checked against the HDF chains. Loaded emulator state is evaluated with the current `emulator.py` class methods, without retraining or using serialized Python method bytecode. Source hashes, package versions and checks against saved observable quantiles are recorded. These sample-based quantities have Monte Carlo uncertainty and are descriptive, not hypothesis tests.

### Parameter sensitivity

'''
    section+=table(['Parameter','JS (bits)','Wasserstein','Normalized Δ median','CS/baseline 68% width'],[(r['parameter'],r['js_divergence_bits'],r['wasserstein'],r['normalized_median_shift'],r['width_ratio_68']) for r in rank])
    r=rank[0];section+=f"\n\nThe largest binned marginal change is `{r['parameter']}` (JS = {r['js_divergence_bits']:.4g} bits). A shape change need not be captured by the median shift alone.\n\n### Changes in parameter correlations\n\nPearson correlations are computed from the two sampled joint posteriors. These are posterior parameter correlations, not the experimental error correlation ρ. The ten largest absolute changes are:\n\n"
    section+=table(['Parameter 1','Parameter 2','Baseline Pearson r','CS Pearson r','Δr'],[(r['parameter_1'],r['parameter_2'],r['baseline_pearson'],r['cs_pearson'],r['delta_pearson']) for r in sorted(correlations,key=lambda r:abs(r['delta_pearson']),reverse=True)[:10]])
    section+='''\n\n### Experimental covariance structure

Both models retain the same diagonal variances. Baseline experimental errors are diagonal; CS experimental errors form one fully correlated block per collision system. At ρ = 1, the experimental CS matrix is rank one and cannot define an ordinary multivariate Gaussian likelihood on its own. Tiny negative eigenvalues within the reported tolerance are numerical roundoff. Singular matrices have undefined ordinary log determinant and infinite condition number; they are shown as “singular / undefined”, rather than regularized silently. Trace and norm depend on the observable coordinates and units.

'''
    section+=table(['System','Model','Bins','Rank','Trace','Frobenius norm','Min eigenvalue','Condition number','Log determinant'],[(r['system'],r['run'],r['n_bins'],r['rank'],r['trace'],r['frobenius'],r['min_eigenvalue'],r['condition_number'],r['logdet']) for r in data['covariance'] if r['component']=='experimental'])
    section+='\n\n'+table(['System','‖CS − baseline‖F / ‖baseline‖F','Max absolute difference','Trace difference'],[(r['system'],r['relative_frobenius'],r['max_abs_difference'],r['trace_difference']) for r in data['differences']])
    section+='''\n\n### Posterior variation versus experimental uncertainty

For each observable bin, σ_post is the sample standard deviation of emulator **means** over the 250 posterior parameter vectors, in physical observable coordinates. The ratio σ_post / σ_exp compares parameter-induced variation with the saved experimental standard deviation. It excludes emulator-noise and experimental-noise draws. CV ratio = |σ_post / posterior median| / |σ_exp / experiment|. Bins are summarized with equal weight; a ratio below one is not evidence of a better fit or calibrated coverage.

'''
    section+=table(['System','Posterior','Mean σ ratio','Median σ ratio','Min','Max','Fraction below 1','Median CV ratio'],[(r['system'],r['run'],r['ratio_mean'],r['ratio_median'],r['ratio_min'],r['ratio_max'],r['fraction_posterior_std_below_exp'],r['cv_ratio_median']) for r in data['uncertainty']])
    section+='\n\nPer-observable breakdown:\n\n'+table(['System','Observable','Baseline median σ ratio','CS median σ ratio'],[(r['system'],r['observable'],next(b['median_std_ratio'] for b in data['per_observable'] if b['system']==r['system'] and b['observable']==r['observable'] and b['run']=='baseline'),r['median_std_ratio']) for r in data['per_observable'] if r['run']=='cs'])
    section+='''\n\n### Residual and Gaussian likelihood terms

The residual is experiment minus the coordinatewise median emulator prediction in **likelihood coordinates** (log multiplicity where applicable). C_total = C_exp + average posterior emulator covariance. This matches the summary-level construction in the source analysis: the coordinatewise median is not necessarily the prediction at a single physical parameter vector. χ²/bin is a descriptive scaling, not reduced χ² with fitted degrees of freedom. The Gaussian log likelihood includes −½ log|C_total| and the normalizing constant; it excludes priors and is not Bayesian evidence.

'''
    section+=table(['System','Posterior/model','χ²','χ²/bin','Log determinant','Gaussian log likelihood'],[(r['system'],r['run'],r['chi2'],r['chi2_per_bin'],r['logdet'],r['gaussian_loglike']) for r in data['likelihood']])
    section+='''\n\nThe raw table `cross_covariance_likelihoods.csv` additionally evaluates **both** experimental covariance choices at each of the same posterior draws, using that draw's emulator covariance. This isolates the covariance change at fixed parameters; it is not a reweighted posterior or a model-selection calculation. `covariance_metrics.csv` also includes mean emulator covariance and total covariance metrics. `posterior_predictive_per_bin.csv` provides residuals, marginally standardized residuals, emulator/experimental variances and uncertainty ratios. Marginally standardized residuals are not whitened residuals when the covariance is correlated.

### Temporal interval checks

`time_window_intervals.csv` lists the median and central 60% / 90% sample intervals in ten non-overlapping windows for each chain. These describe temporal changes in the sampled distributions and do not prove convergence. No τ, ESS or walker-based R-hat is added to this report.

### Validation and rerun

'''
    with (out/'time_window_intervals.csv').open() as f:window_rows=list(csv.DictReader(f))
    temporal=[]
    for parameter in [r['parameter'] for r in sensitivity]:
        result=[parameter]
        for run in ['baseline','cs']:
            values=[r for r in window_rows if r['run']==run and r['parameter']==parameter]
            medians=np.array([float(r['median']) for r in values])
            widths=np.array([float(r['q95'])-float(r['q05']) for r in values])
            result.append(float(np.ptp(medians)/np.median(widths)))
        temporal.append(result)
    temporal_text='\n\nThe table reports the range of window medians divided by the median window 90% interval width. It measures temporal movement relative to the sampled spread; it has no convergence pass/fail threshold.\n\n'+table(['Parameter','Baseline temporal ratio','CS temporal ratio'],temporal)
    section=section.replace('### Validation and rerun',temporal_text+'\n\n### Validation and rerun')
    with (out/'cross_covariance_likelihoods.csv').open() as f:cross_rows=list(csv.DictReader(f))
    cross_summary=[]
    for system in ['Au-Au-200','d-Au-200']:
        for source in ['baseline','cs']:
            for evaluated in ['baseline','cs']:
                subset=[r for r in cross_rows if r['system']==system and r['posterior_source']==source and r['evaluated_covariance']==evaluated]
                chi=np.array([float(r['chi2']) for r in subset]);ll=np.array([float(r['gaussian_loglike']) for r in subset])
                cross_summary.append([system,source,evaluated,float(np.median(chi)),float(np.quantile(ll,.16)),float(np.median(ll)),float(np.quantile(ll,.84))])
    extra='### Likelihood comparison at the same posterior draws\n\nEach source uses the same 250 parameter vectors under both covariance evaluations. The log-likelihood interval below is the central 68% spread across those draws; it is not a confidence interval or evidence ratio.\n\n'+table(['System','Posterior source','Evaluated C_exp','Median χ²','LogL q16','Median logL','LogL q84'],cross_summary)
    with (out/'posterior_predictive_per_bin.csv').open() as f:bin_rows=list(csv.DictReader(f))
    ratios=[]
    for r in data['covariance']:
        if r['component']!='experimental_plus_mean_emulator':continue
        subset=[b for b in bin_rows if b['system']==r['system'] and b['run']==r['run']]
        variance_ratios=[float(b['emulator_variance'])/float(b['experimental_variance']) for b in subset]
        ratios.append([r['system'],r['run'],r['rank'],r['n_bins'],r['condition_number'],float(np.median(variance_ratios))])
    extra+='\n\n### Total covariance and emulator contribution\n\nThe variance ratio uses the diagonal of the posterior-averaged emulator covariance divided by the experimental diagonal variance, in likelihood coordinates. It measures variance magnitude, not off-diagonal structure.\n\n'+table(['System','Posterior/model','Total rank','Bins','Total condition number','Median emulator/experimental variance'],ratios)
    section=section.replace('### Temporal interval checks',extra+'\n\n### Temporal interval checks')
    section=section.replace('### Validation and rerun','### Validation and rerun\n\nThe emulator cache was saved with scikit-learn 1.6.1 and evaluated here with scikit-learn '+__import__('sklearn').__version__+'. Agreement with saved quantiles checks the predicted means; it does not independently validate historical emulator covariance values. Package versions are recorded in diagnostics.json.')
    valid=data['validation'];section+=f"Recomputed observable quantiles match the saved plot CSVs within rtol 10⁻⁶ and atol 10⁻¹⁰ in **{sum(r['matches_saved_quantiles'] for r in valid)} of {len(valid)}** run/observable comparisons. See `saved_quantile_validation.csv` for errors.\n\nRun from the project directory:\n\n```bash\npython mcmc/scripts/diagnose_phase2_numerical.py\n```\n\nThis updates the numerical tables and this notebook section without generating new graphics or running MCMC. The original Phase 2 notebook builder preserves this section on regeneration."
    path=STUDY/'comparison_global_correlation_models.ipynb';nb=json.loads(path.read_text());heading='## Numerical covariance and uncertainty diagnostics'
    nb['cells']=[c for c in nb['cells'] if not ''.join(c.get('source',[])).startswith(heading)]
    nb['cells'].insert(2,dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=section.splitlines(True)))
    path.write_text(json.dumps(nb,indent=1)+'\n')
    (out/'report.md').write_text(section+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--refresh-notebook',action='store_true',help='Rebuild notebook tables from already saved diagnostics, without recomputing predictions.');args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    sys.path.insert(0,str(ROOT))
    out=STUDY/'numerical_diagnostics';out.mkdir(exist_ok=True)
    if args.refresh_notebook:
        data=json.loads((out/'diagnostics.json').read_text())
        def read_numeric(name,text_columns):
            with (out/name).open() as f:rows=list(csv.DictReader(f))
            for row in rows:
                for k in row:
                    if k not in text_columns:row[k]=float(row[k])
            return rows
        sensitivity=read_numeric('posterior_sensitivity.csv',{'parameter'})
        correlations=read_numeric('parameter_pair_correlations.csv',{'parameter_1','parameter_2'})
        update_notebook(data,sensitivity,correlations,out)
        print('Refreshed notebook from saved numerical results.',flush=True)
        return
    with threadpool_limits(limits=1):
        inventory,sensitivity,correlations=posterior_diagnostics(out)
        data=predictive_diagnostics(out,args.batch_size)
    import scipy,sklearn,dill
    provenance=dict(source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['emulator.py','configurations.py','bayes_mcmc.py','bayes_mcmc_cs_model.py','experimental_cs_covariance.py']},python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,dill=dill.__version__,run_inventory=inventory,definitions='No MCMC or plots. 250 fixed saved draws per posterior for predictions; full 200k thinned samples for parameter metrics.',**data)
    # All missing/singular quantities are explicit null, never nonstandard JSON NaN.
    (out/'diagnostics.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    update_notebook(data,sensitivity,correlations,out)
    print('Updated numerical diagnostics and notebook:',out,flush=True)
if __name__=='__main__':main()
