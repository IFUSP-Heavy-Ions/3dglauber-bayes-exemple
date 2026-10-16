"""Scientific checks for same-observable covariance and chain resume isolation."""
import ast
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import logging
from experimental_sector_covariance import (
    sector_covariance, likelihood_std, bin_centers, observable_axis,
    resolve_length, validate_overrides, covariance_signature, validate_regions, rapidity_region_labels,
)


class SectorCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.bins={'S':{'v22_pt_cen_00_10_PHEN':[[.2,.4],[.4,.8],[1.4,1.8]],
                        'v22_pt_cen_10_20_PHEN':[[.2,.4],[.4,.8]]}}
        self.slices=[('v22_pt_cen_00_10_PHEN',slice(0,3)),
                     ('v22_pt_cen_10_20_PHEN',slice(3,5))]
        self.std=np.array([.1,.2,.3,.4,.5])

    def build(self,model='rbf',bins=None,lengths=None,rho=1.):
        return sector_covariance('S',self.slices,self.std,bins or self.bins,model,
                                 rho=rho,lengths=lengths or {'pt':.5})

    def test_physical_distance_gaps_and_block_independence(self):
        cov=self.build()
        x=bin_centers(self.bins['S'][self.slices[0][0]])
        expected=np.outer(self.std[:3],self.std[:3])*np.exp(-((x[:,None]-x[None,:])/.5)**2)
        np.testing.assert_allclose(cov[:3,:3],expected)
        np.testing.assert_array_equal(cov[:3,3:],0)
        np.testing.assert_allclose(np.diag(cov),self.std**2)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-12)
        # A physical gap must suppress correlation more than adjacent-bin spacing.
        corr=cov/np.outer(self.std,self.std)
        self.assertLess(corr[1,2],corr[0,1])

    def test_change_of_units_preserves_correlation(self):
        bins={'S':{k:np.asarray(v)*1000 for k,v in self.bins['S'].items()}}
        np.testing.assert_allclose(self.build(),self.build(bins=bins,lengths={'pt':500.}))

    def test_cs_limits_and_negative_bound_is_per_sector(self):
        cov=self.build(model='cs')
        np.testing.assert_allclose(cov[:3,:3],np.outer(self.std[:3],self.std[:3]))
        np.testing.assert_array_equal(cov[:3,3:],0)
        np.testing.assert_array_equal(self.build(model='cs',rho=0),np.diag(self.std**2))
        # -0.4 is allowed for sectors of sizes 3 and 2, even though it is
        # invalid for one global block of size 5.
        self.assertGreaterEqual(np.linalg.eigvalsh(self.build(model='cs',rho=-.4)).min(),-1e-12)
        with self.assertRaises(ValueError):self.build(model='cs',rho=-.6)

    def test_coordinate_is_bin_axis_not_observable_units(self):
        self.assertEqual(observable_axis('meanpT_pi_STAR'),'centrality')
        self.assertEqual(observable_axis('v22_int_STAR'),'centrality')
        self.assertEqual(observable_axis('dNdeta_eta_cen_00_03_PHOB'),'eta')
        self.assertEqual(observable_axis('v22_pt_cen_00_10_PHEN'),'pt')
        with self.assertRaises(ValueError):observable_axis('unknown')

    def test_log_error_convention_and_missing_lengths(self):
        mean=np.array([10.,20.]);std=np.array([1.,4.])
        np.testing.assert_allclose(likelihood_std('dNdeta_eta',mean,std),np.sqrt(np.log1p((std/mean)**2)))
        np.testing.assert_allclose(likelihood_std('v22_pt',mean,std),std)
        obs=self.slices[0][0]
        self.assertEqual(resolve_length('S',obs,{'pt':1.},{'S':{obs:.2}}),.2)
        for lengths in [{},{'pt':0},{'pt':float('nan')}]:
            with self.assertRaises(ValueError):resolve_length('S',obs,lengths,{})
        with self.assertRaises(ValueError):validate_overrides({'S':{'misspelled':1}},self.bins)

    def test_signature_changes_for_scales_or_bin_definitions(self):
        slices={'S':self.slices}
        a=covariance_signature({'S':self.build()},slices,self.bins)
        b=covariance_signature({'S':self.build(lengths={'pt':1.})},slices,self.bins)
        self.assertNotEqual(a,b)

    def test_sector_chain_dispatch_paths_and_metadata(self):
        from experimental_sector_covariance import covariance_signature
        root=Path(__file__).parent
        tree=ast.parse((root/'bayes_mcmc_sector.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Chain')
        bins=self.bins;slices=self.slices;std=self.std
        class ParentStub:
            filename='chain-idf-0.hdf'
            def __init__(self,path,cs_rho):
                self.path=Path(path);self.cs_rho=cs_rho
                self._slices={'S':slices}
                self._expt_cov={'S':self._build_experimental_covariance('S',std)}
        import json
        with tempfile.TemporaryDirectory() as temp:
            ns=dict(GlobalCSChain=ParentStub,Path=Path,json=json,workdir=Path(temp),
                    obs_cent_list=bins,system_strs=['S'],sector_covariance=sector_covariance,
                    covariance_signature=covariance_signature,validate_overrides=validate_overrides,validate_regions=validate_regions)
            exec(compile(ast.Module(body=[cls],type_ignores=[]),'<sector-chain>','exec'),ns)
            cs=ns['Chain']('cs',regions={});rbf=ns['Chain']('rbf',lengths={'pt':.5},regions={})
            np.testing.assert_array_equal(cs._expt_cov['S'][:3,3:],0)
            np.testing.assert_allclose(rbf._expt_cov['S'],self.build())
            self.assertIn('cs-observable',cs.path.name)
            self.assertIn('rbf-observable',rbf.path.name)
            self.assertEqual(cs._covariance_metadata()['experimental_cs_scope'],'observable')
            self.assertEqual(rbf._covariance_metadata()['experimental_covariance_model'],'rbf')
            changed=ns['Chain']('rbf',lengths={'pt':1.},regions={})
            self.assertNotEqual(rbf._covariance_metadata(),changed._covariance_metadata())

    def test_central_and_combined_laterals_are_sectors_for_both_models(self):
        obs='dNdeta_eta_cen_00_05_PHEN'
        bins={'S':{obs:[[-2.6,-2.5],[-2.5,-2.4],[-.4,-.3],[.3,.4],[1.3,1.4],[1.4,1.5]]}}
        regions={'S':{obs:{'boundaries':[-1,1],'labels':['lateral','central','lateral']}}}
        labels=rapidity_region_labels('S',obs,bins['S'][obs],regions)
        self.assertEqual(labels.tolist(),['lateral','lateral','central','central','lateral','lateral'])
        for model in ['cs','rbf']:
            cov=sector_covariance('S',[(obs,slice(0,6))],np.ones(6),bins,model,
                                  lengths={'eta':10},regions=regions)
            np.testing.assert_array_equal(cov[labels[:,None]!=labels[None,:]],0)
            np.testing.assert_array_equal(np.diag(cov),1)
            self.assertGreater(cov[0,1],0)
            self.assertGreater(cov[0,4],0)
            self.assertEqual(cov[0,2],0)
            self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-12)
        # The lateral sector contains all four outer bins; bounds use that group.
        sector_covariance('S',[(obs,slice(0,6))],np.ones(6),bins,'cs',rho=-.3,regions=regions)
        with self.assertRaises(ValueError):
            sector_covariance('S',[(obs,slice(0,6))],np.ones(6),bins,'cs',rho=-.4,regions=regions)
        # A cut cannot silently divide an experimental bin.
        bad={'S':{obs:{'boundaries':[-.35],'labels':['left','right']}}}
        with self.assertRaises(ValueError):validate_regions(bad,bins)
        # Continuous coverage alone must not invent rapidity-sector boundaries.
        self.assertEqual(set(rapidity_region_labels('S',obs,bins['S'][obs])),{'all'})

    def test_sector_entrypoints_expose_legacy_emulator_globals(self):
        root=Path(__file__).parent
        for filename in ('bayes_mcmc_cs_sector.py','bayes_mcmc_rbf_sector.py'):
            tree=ast.parse((root/filename).read_text())
            imports=[
                node for node in tree.body
                if isinstance(node,ast.ImportFrom) and node.module=='configurations'
            ]
            imported={alias.name for node in imports for alias in node.names}
            self.assertTrue(
                {'do_transform_design','transform_design'} <= imported,
                f'{filename} must bind legacy dill globals in __main__',
            )
            self.assertTrue(any(
                isinstance(node,ast.Import) and any(
                    alias.name=='numpy' and alias.asname=='np' for alias in node.names
                ) for node in tree.body
            ))
            self.assertTrue(any(
                isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id=='main' and any(
                    keyword.arg=='legacy_emulator_namespace'
                    for keyword in node.keywords
                ) for node in ast.walk(tree)
            ))

    def test_empty_existing_chain_restarts_with_burnin(self):
        path=Path(__file__).with_name('bayes_mcmc_cs_model.py')
        tree=ast.parse(path.read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Chain')
        run_mcmc=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='run_mcmc')

        class PoolStub:
            def __init__(self,workers): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass

        class SamplerStub:
            def __init__(self,nwalkers,ndim,log_prob_fn,pool):
                self.nwalkers=nwalkers;self.ndim=ndim
                self.chain=np.empty((nwalkers,0,ndim))
            def run_mcmc(self,positions,nsteps,status=None):
                steps=np.repeat(np.asarray(positions)[:,None,:],nsteps,axis=1)
                self.chain=np.concatenate((self.chain,steps),axis=1)
                self.flatchain=self.chain.reshape(-1,self.ndim)
                self.flatlnprobability=np.arange(len(self.flatchain))
                return self.chain[:,-1,:]
            def reset(self):
                self.chain=np.empty((self.nwalkers,0,self.ndim))

        ns={'np':np,'logging':logging,'Pool':PoolStub,'cpu_count':lambda:2,
            'LoggingEnsembleSampler':SamplerStub,'usePTSampler':False}
        exec(compile(ast.Module(body=[run_mcmc],type_ignores=[]),str(path),'exec'),ns)

        class ChainStub:
            ndim=3
            log_posterior=None
            def __init__(self,filepath): self.filepath=filepath
            def open(self,mode): return h5py.File(self.filepath,mode)
            def _validate_covariance_metadata(self,handle): pass
            def random_pos(self,nwalkers): return np.zeros((nwalkers,self.ndim))

        with tempfile.TemporaryDirectory() as temp:
            filepath=Path(temp)/'chain.hdf'
            with h5py.File(filepath,'w') as handle:
                handle.create_dataset('chain',dtype='f8',shape=(2,0,3),
                                      chunks=(2,1,3),maxshape=(2,None,3))
            ns['run_mcmc'](ChainStub(filepath),nsteps=3,nburnsteps=4,
                           nwalkers=2,ncpu=2)
            with h5py.File(filepath,'r') as handle:
                self.assertEqual(handle['chain'].shape,(2,3,3))
                self.assertEqual(handle['burnin'].shape,(2,4,3))

    def test_legacy_prediction_mean_and_covariance(self):
        path=Path(__file__).with_name('emulator.py')
        tree=ast.parse(path.read_text())
        covariance=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='_Covariance')
        emulator=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Emulator')
        methods=[node for node in emulator.body if isinstance(node,ast.FunctionDef)
                 and node.name in ('predict','_inverse_transform')]
        ns={'__name__':'__main__','np':np,'do_transform_design':True,
            'transform_design':lambda x:x*2}
        exec(compile(ast.Module(body=[covariance]+methods,type_ignores=[]),str(path),'exec'),ns)

        class GPStub:
            def predict(self,x,return_cov=False):
                mean=x[:,0]
                return (mean,np.eye(len(x))*.5) if return_cov else mean

        model=type('LegacyEmulator',(),{name:ns[name] for name in
                                      ('predict','_inverse_transform')})()
        model.gps=[GPStub()]
        model._trans_matrix=np.array([[2.,3.]])
        model.scaler=type('Scaler',(),{'mean_':np.array([1.,2.])})()
        model._slices={'obs':slice(0,2)}
        model._var_trans=np.array([[4.,6.,6.,9.]])
        model._cov_trunc=np.eye(2)*.1
        model.nobs=2
        mean,cov=model.predict(np.array([[1.]]),return_cov=True,
                               extra_std=np.array(0.))
        np.testing.assert_allclose(mean['obs'],[[5.,8.]])
        np.testing.assert_allclose(cov['obs','obs'],[[[2.1,3.],[3.,4.6]]])


    def test_existing_global_metadata_and_resume_guard(self):
        path=Path(__file__).with_name('bayes_mcmc_cs_model.py')
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Chain')
        selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_covariance_metadata','_validate_covariance_metadata']]
        ns={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
        chain=type('MetadataOnly',(),ns)();chain.cs_rho=1.
        self.assertEqual(chain._covariance_metadata(),{'experimental_covariance_model':'compound_symmetry','experimental_cs_scope':'system','experimental_cs_rho':1.})
        with tempfile.TemporaryDirectory() as temp:
            with h5py.File(Path(temp)/'chain.hdf','w') as f:
                chain._validate_covariance_metadata(f)
                f.create_dataset('chain',shape=(2,2,1),dtype='f8')
                chain._validate_covariance_metadata(f)
                f.attrs['experimental_cs_scope']='observable'
                with self.assertRaises(ValueError):chain._validate_covariance_metadata(f)


if __name__=='__main__':unittest.main()
