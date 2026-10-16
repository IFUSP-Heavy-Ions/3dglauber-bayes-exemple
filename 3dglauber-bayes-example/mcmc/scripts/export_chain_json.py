#!/usr/bin/env python3
"""Lossless HDF5 chain export to <chain directory>/json/<chain stem>.json."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
import h5py
import numpy as np


def export(path):
    path=path.expanduser().resolve()
    project=next(p for p in Path(__file__).resolve().parents if (p/'production_designs').is_dir())
    design=project/'production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production/design_ranges_main_AuAu-200.dat'
    with design.open() as f: names=[r['param'] for r in csv.DictReader(f)]
    out=path.parent/'json'/f'{path.stem}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    partial=out.with_suffix('.json.partial')
    with h5py.File(path,'r') as f,tempfile.TemporaryDirectory(prefix='chain-json-') as temp:
        source=f['chain']
        if source.ndim!=3 or source.shape[2]!=len(names):raise ValueError('Chain schema does not match parameter design.')
        shape=source.shape
        snapshot=np.memmap(Path(temp)/'samples.dat',mode='w+',dtype='float64',shape=shape)
        print(f'{path.name}: copying {shape} without thinning or discard',flush=True)
        for start in range(0,shape[1],512):
            block=source[:,start:start+512,:]
            if not np.isfinite(block).all():raise ValueError('Nonfinite samples cannot be represented in strict JSON.')
            snapshot[:,start:start+block.shape[1],:]=block
        snapshot.flush()
        digest=hashlib.sha256();size=0;count=0
        with partial.open('wb') as target:
            def write(text):
                nonlocal size
                encoded=text.encode('utf-8');target.write(encoded);digest.update(encoded);size+=len(encoded)
            meta=dict(source=str(path),dataset='chain',shape=list(shape),
                axes=['walker','step','parameter'],parameters=names,burnin_discarded=0,thin=1)
            write(json.dumps(meta,ensure_ascii=False,allow_nan=False)[:-1]+',"chain":[')
            for w in range(shape[0]):
                if w:write(',')
                write('[')
                for start in range(0,shape[1],1024):
                    block=np.asarray(snapshot[w,start:start+1024,:])
                    encoded=json.dumps(block.tolist(),separators=(',',':'),allow_nan=False)
                    # Validate every serialized block, preserving every float64 value.
                    if not np.array_equal(np.asarray(json.loads(encoded)),block):
                        raise ValueError('Float round-trip validation failed.')
                    if start:write(',')
                    write(encoded[1:-1]);count+=block.size
                write(']')
                if (w+1)%25==0:print(f'{path.name}: {w+1}/{shape[0]} walkers exported',flush=True)
            write(']}\n')
        assert count==int(np.prod(shape))
        partial.replace(out)
        manifest=dict(file=str(out),shape=list(shape),float_values=count,
            bytes=size,sha256=digest.hexdigest(),validation='Every numeric block JSON round-trip checked exactly against source float64 values.')
        out.with_name(out.stem+'_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        del snapshot
        print(f'Saved: {out} ({size/1e9:.2f} GB)',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain',type=Path,nargs='+',required=True)
    for chain in parser.parse_args().chain:export(chain)
