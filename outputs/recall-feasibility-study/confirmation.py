"""Gate-locked confirmation with input-only exact-duplicate exclusion.

Added after pilot vector-overlap audit, before ANY confirmation outcomes.
No hypothesis/model threshold changes. The original frozen training algorithm
is imported unchanged. Input filtering never uses nearest-neighbor labels.
"""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import argparse,json
import h5py,numpy as np
from threadpoolctl import threadpool_limits
import study
from prepare import ROOT,WORK,load,rows,save,digest,assert_frozen
def checks():
    assert_frozen();gate=json.loads((ROOT/'gate.json').read_text())
    assert gate['confirmation_allowed'] and gate['H1_pass']
    lock=json.loads((ROOT/'CONFIRMATION_LOCK.json').read_text())
    assert lock['frozen_inputs_sha256']==digest(ROOT/'FROZEN.json')
    assert lock['confirmation_driver_sha256']==digest(Path(__file__))
    return lock
from pathlib import Path
def normalize(v):return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
def prepare():
    lock=checks();target=ROOT/'CONFIRMATION_SPLITS.json';assert not target.exists()
    ledger=json.loads((ROOT/'splits.json').read_text());record={}
    for d in ['glove','coco_clip']:
        spec=ledger['datasets'][d];path=WORK/('adaptive-retrieval/glove-25-angular.hdf5' if d=='glove' else 'diagnostic-data/coco-t2i-512-angular.hdf5')
        with h5py.File(path) as f:
            ds=f[spec['source']];prior=np.concatenate([spec['roles'][r] for r in ['design','calibration','pilot']])
            seen=set(x.tobytes() for x in normalize(rows(ds,prior)))
            original=spec['roles']['confirmation'];values=normalize(rows(ds,original));selected=[];removed=[]
            for ident,v in zip(original,values):
                key=v.tobytes()
                if key in seen:removed.append(ident)
                else:selected.append(ident);seen.add(key)
            spare=np.random.default_rng(99001).permutation(len(ds));forbidden=set(prior)|set(original)
            if d=='glove':
                forbidden.update(np.random.default_rng(73001).permutation(len(ds))[:100000].tolist())
                for p in (ROOT.parent/'github/budgeted-tail-risk-ann/results').glob('glove_*_provenance.json'):
                    forbidden.update(json.loads(p.read_text()).get('calibration_training_ids',[]))
            added=[];scanned=0
            for ident in spare:
                if len(selected)==2000:break
                if int(ident) in forbidden:continue
                v=normalize(rows(ds,[int(ident)]))[0];scanned+=1;key=v.tobytes()
                if key in seen:continue
                selected.append(int(ident));added.append(int(ident));seen.add(key)
            assert len(selected)==2000 and len(set(selected))==2000
        record[d]=dict(ids=selected,removed_duplicate_ids=removed,replacement_ids=added,
            original_vectors_inspected=2000,extra_vectors_inspected=scanned,confirmation_labels_evaluated=0)
    save(target,dict(datasets=record,driver_sha256=digest(Path(__file__)),
        original_frozen_sha256=digest(ROOT/'FROZEN.json'),rule='Exclude exact normalized vector repeats against design/calibration/pilot and within confirmation; fixed replacement order; no labels used.'))
    print('Input-only confirmation roles frozen:',{d:dict(duplicates=len(v['removed_duplicate_ids']),replacements=len(v['replacement_ids'])) for d,v in record.items()})
def confirmed_load(d,s,stage):
    base,q=load(d,s,stage);ledger=json.loads((ROOT/'splits.json').read_text());new=json.loads((ROOT/'CONFIRMATION_SPLITS.json').read_text())
    assert new['driver_sha256']==digest(Path(__file__))
    path=WORK/('adaptive-retrieval/glove-25-angular.hdf5' if d=='glove' else 'diagnostic-data/coco-t2i-512-angular.hdf5')
    with h5py.File(path) as f:q['confirmation']=normalize(rows(f[ledger['datasets'][d]['source']],new['datasets'][d]['ids']))
    return base,q
def run():
    checks();assert (ROOT/'CONFIRMATION_SPLITS.json').exists();study.load=confirmed_load
    with threadpool_limits(limits=1):
        for d in ['glove','coco_clip']:
            for s in range(5):
                if (ROOT/'results/confirmation'/f'{d}_{s}.json').exists():continue
                study.run(d,s,'confirmation')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else run()
