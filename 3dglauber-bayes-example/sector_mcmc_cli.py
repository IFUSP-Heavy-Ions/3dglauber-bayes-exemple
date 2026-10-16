"""Lightweight CLI and read-only covariance check; sampler imports are lazy."""
import argparse
import json
from pathlib import Path
import numpy as np
from experimental_sector_covariance import (
    AXIS_UNITS, observable_axis, bin_centers, likelihood_std, resolve_length,
    sector_covariance, validate_overrides, covariance_signature, validate_regions, rapidity_region_labels,
)


def parser_for(model):
    parser = argparse.ArgumentParser(description=(
        f'MCMC with {model.upper()} experimental correlations only within each exact observable key.'))
    parser.add_argument('nsteps', type=int, nargs='?', help='Additional production steps')
    parser.add_argument('--nwalkers', type=int)
    parser.add_argument('--nburnsteps', type=int)
    parser.add_argument('--ntemps', type=int, default=1)
    parser.add_argument('--status', type=int)
    parser.add_argument('--ncpus', type=int, default=None)
    parser.add_argument('--chain-path', type=Path)
    parser.add_argument('--check-covariance', action='store_true',
                        help='Read experimental data and validate blocks without loading emulators or running MCMC')
    parser.add_argument('--rapidity-regions-json',type=Path,default=Path(__file__).with_name('rapidity_regions.json'),help='Eta regions JSON (default: rapidity_regions.json, separating clear acceptance gaps); zero experimental correlation between regions')
    if model == 'cs':
        parser.add_argument('--cs-rho', type=float, default=1.,
                            help='Within-observable Pearson correlation (default: 1)')
    else:
        for axis, unit in AXIS_UNITS.items():
            parser.add_argument(f'--rbf-length-{axis}', type=float,
                                help=f'RBF e-folding length in {unit}; required unless overridden per observable')
        parser.add_argument('--rbf-lengths-json', type=Path,
                            help='Optional nested {system: {exact_observable_key: length}} overrides')
    return parser


def inspect_covariance(config, model, rho, lengths, overrides, regions=None):
    import pandas as pd
    if config.run_closure or config.pseudovalidation:
        raise ValueError('--check-covariance reads experimental data; disable closure and pseudovalidation.')
    if config.change_exp_error or config.set_exp_error_to_zero or config.number_of_models_per_run != 1:
        raise ValueError('--check-covariance currently requires unchanged errors and one model per run.')
    bins = {s: config.obs_cent_list[s] for s in config.system_strs}
    matrices, slices, report = {}, {}, []
    for system in config.system_strs:
        slices[system] = [];sigmas=[];blocks=[];offset=0
        for obs in config.active_obs_list[system]:
            edges=bins[system][obs];n=len(edges)
            file=Path(config.dir_obs_exp)/system/config.expt_label_dict[obs[-4:]]/(obs+'.dat')
            raw=pd.read_csv(file,sep=' ',skiprows=2,escapechar='#')
            mean=raw['val'].iloc[:n].to_numpy(dtype=float)
            variance=raw['err'].iloc[:n].to_numpy(dtype=float)
            if len(mean)!=n or np.any(variance<0):
                raise ValueError(f'Invalid experimental bins or error variances: {system}/{obs}')
            std=likelihood_std(obs,mean,np.sqrt(variance),config.transform_multiplicities)
            axis=observable_axis(obs);centers=bin_centers(edges,n)
            slices[system].append((obs,slice(offset,offset+n)));sigmas.extend(std);offset+=n
            labels=rapidity_region_labels(system,obs,edges,regions)
            blocks.append(dict(observable=obs,n_bins=n,coordinate=axis,unit=AXIS_UNITS[axis],
                               rapidity_regions={label:int(np.sum(labels==label)) for label in np.unique(labels)},
                               min_bin_center=float(centers.min()),max_bin_center=float(centers.max()),
                               length=resolve_length(system,obs,lengths,overrides) if model=='rbf' else None))
        matrix=sector_covariance(system,slices[system],np.array(sigmas),bins,model,
                                 rho=rho,lengths=lengths,overrides=overrides,regions=regions)
        eig=np.linalg.eigvalsh(matrix);tol=len(eig)*np.finfo(float).eps*max(abs(eig).max(),np.finfo(float).tiny)
        if eig.min() < -tol:
            raise ValueError(f'Covariance is not positive semidefinite: {system}')
        np.testing.assert_allclose(np.diag(matrix),np.square(sigmas),rtol=1e-14,atol=0)
        matrices[system]=matrix
        report.append(dict(system=system,n_bins=len(sigmas),n_sectors=sum(len(b['rapidity_regions']) for b in blocks),
                           numerical_rank=int(np.sum(eig>tol)),min_eigenvalue=float(eig.min()),
                           psd_tolerance=float(tol),blocks=blocks))
    return dict(model=model,scope='observable_rapidity_region' if regions else 'observable',rapidity_configuration=regions or {},rho=rho if model=='cs' else None,
                signature=covariance_signature(matrices,slices,bins),systems=report)


def main(model):
    parser=parser_for(model);args=parser.parse_args()
    if args.nsteps is None and not args.check_covariance:
        parser.error('Specify nsteps or --check-covariance.')
    if args.nsteps is not None and args.nsteps<=0:parser.error('nsteps must be positive.')
    for key in ['nwalkers','ncpus','status','ntemps','nburnsteps']:
        value=getattr(args,key)
        if value is not None and value<=0:parser.error(f'--{key} must be positive.')
    if args.nburnsteps is not None and args.nburnsteps<2:parser.error('--nburnsteps must be at least 2.')
    rho=getattr(args,'cs_rho',1.)
    if not np.isfinite(rho) or not -1<=rho<=1:parser.error('--cs-rho must be finite and in [-1,1].')
    lengths={axis:getattr(args,'rbf_length_'+axis) for axis in AXIS_UNITS
             if getattr(args,'rbf_length_'+axis,None) is not None}
    overrides={}
    regions={}
    try:
        for value in lengths.values():
            if not np.isfinite(value) or value<=0:raise ValueError('RBF lengths must be finite and positive.')
        if getattr(args,'rbf_lengths_json',None):overrides=json.loads(args.rbf_lengths_json.read_text())
        if args.rapidity_regions_json:regions=json.loads(args.rapidity_regions_json.read_text())
        import configurations as config
        validate_overrides(overrides,{s:config.obs_cent_list[s] for s in config.system_strs})
        validate_regions(regions,{s:config.obs_cent_list[s] for s in config.system_strs})
        if model=='rbf':
            for system in config.system_strs:
                for obs in config.active_obs_list[system]:resolve_length(system,obs,lengths,overrides)
        if args.check_covariance:
            result=inspect_covariance(config,model,rho,lengths,overrides,regions)
            print(json.dumps(result,indent=2,allow_nan=False));return
    except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))
    from bayes_mcmc_sector import Chain
    kwargs=dict(nsteps=args.nsteps,nwalkers=args.nwalkers,nburnsteps=args.nburnsteps,
                ntemps=args.ntemps,status=args.status)
    if args.ncpus is not None:kwargs['ncpu']=args.ncpus
    Chain(model=model,path=args.chain_path,cs_rho=rho,lengths=lengths,overrides=overrides,regions=regions).run_mcmc(**kwargs)
