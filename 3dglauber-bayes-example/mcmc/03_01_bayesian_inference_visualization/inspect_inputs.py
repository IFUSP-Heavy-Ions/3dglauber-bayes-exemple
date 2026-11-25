from pathlib import Path
import sys,os,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'python-packages'),str(ROOT/'mcmc/scripts'),str(ROOT)]
os.chdir(ROOT);os.environ['MPLCONFIGDIR']=str(HERE/'logs/matplotlib')
import h5py,numpy as np,dill
for name in ['01_01_baseline_reproduction','01_04_mcmc_convergence','01_00_reference_posterior']:
 p=next((ROOT/'mcmc'/name).glob('*.hdf'))
 with h5py.File(p,'r') as f:
  print('CHAIN',p,dict(f.attrs),[(k,getattr(v,'shape',None),dict(v.attrs)) for k,v in f.items()],flush=True)
from diagnose_phase2_numerical import load_emulator_module
m=load_emulator_module()
with (ROOT/'emulator/emulator-Au-Au-200-idf-0-npc-10.dill').open('rb') as f:e=dill.load(f)
e.__class__=m.Emulator
print('EMULATOR',e.nobs,e.npc,len(e.gps),'INPUT',e.gps[0].X_train_.shape,flush=True)
print('OBSERVABLES',list(e._slices.items()),flush=True)
from build_sector_studies import validate_reference
print('VALIDATED',validate_reference(e,'Au-Au-200'),flush=True)
with h5py.File(ROOT/'mcmc/01_01_baseline_reproduction/chain-idf-0.hdf','r') as f:x=f['chain'][0,-1,:][None,:]
mean,cov=e.predict(x,return_cov=True,extra_std=np.array(0.))
print('PREDICT',[(k,v.shape) for k,v in mean.items()],cov.array.shape,'finite',np.isfinite(cov.array).all(),flush=True)
