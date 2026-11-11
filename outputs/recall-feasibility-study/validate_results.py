"""Post-pilot independent trace checks; no model/gate tuning or final access."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from prepare import ROOT,load,assert_frozen,save,digest
from engine import IVF,distances
from control import groups,TARGET,components
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def main(stage='pilot'):
    assert_frozen();counts=Counter();duplicates=[]
    if stage=='confirmation':
        from confirmation import confirmed_load
        loader=confirmed_load;role='confirmation';expected=2000
    else:loader=load;role='pilot';expected=500
    for d in ['glove','coco_clip']:
        for s in range(5):
            base,q=loader(d,s,stage);idx=IVF(base,s)
            dc=distances(q['design'],idx.centers);ds=np.sort(dc,axis=1)
            a=np.sqrt(ds[:,0]/np.maximum(ds[:,1],1e-15));edges=np.quantile(a,[.8,.9,.95])
            pg=groups(distances(q[role],idx.centers),edges)
            prefix=ROOT/'results'/stage/f'{d}_{s}'
            meta=json.loads(prefix.with_suffix('.json').read_text())
            assert meta['frozen_sha256']==digest(ROOT/'FROZEN.json') and meta['confirmation_evaluated']==(stage=='confirmation')
            with np.load(str(prefix)+'_raw.npz',allow_pickle=False) as z:
                for r in read(Path(str(prefix)+'_policies.csv')):
                    f=float(r['cap_fraction']);n=int(r['ncal']);m=r['method'];key=f'{f}__{n}__{m}'
                    v=z[key+'__loss'];w=z[key+'__work'];B=64+int(f*10000)
                    assert len(v)==len(w)==expected and abs(v.mean()-float(r['mean_fnr']))<1e-12
                    assert abs(w.mean()-float(r['mean_executed_distances']))<1e-9
                    assert np.all(w<=B) and int(r['cap_violations'])==0
                    rv=pg.T@components(v)/np.maximum(pg.sum(0)[:,None],1)
                    assert int(((rv>TARGET+1e-12)&(pg.sum(0)[:,None]>0)).sum())==int(r['empirical_risk_violations'])
                    counts['policy_cases']+=1
                for r in read(Path(str(prefix)+'_groups.csv')):
                    f=float(r['cap_fraction']);n=int(r['ncal']);m=r['method'];g=int(r['group'])
                    v=z[f'{f}__{n}__{m}__loss'][pg[:,g]]
                    assert len(v)==int(r['n'])
                    if len(v):np.testing.assert_allclose(components(v).mean(0),[float(r[k]) for k in ['mean_fnr','tail20','tail50']],atol=1e-12)
                    counts['group_rows']+=1
                for model in ['geometry','occupancy']:
                    assert abs(z[model+'_per_query_brier'].mean()-meta['diagnostic_models'][model]['brier'])<1e-12
                ex=z['crc_reference_executed'];ret=z['crc_reference_returned']
                assert np.all(ex>=ret) and np.all(ex<=2564)
                assert abs((ex.mean()-ret.mean())/ex.mean()-meta['lookahead']['undercount_fraction'])<1e-12
            if s==0:
                # Audit exact input duplicates, without touching confirmation rows.
                sets={role:set(row.tobytes() for row in q[role]) for role in q}
                item=dict(dataset=d,design_calibration_exact_vector_overlap=len(sets['design']&sets['calibration']),
                    design_evaluation_exact_vector_overlap=len(sets['design']&sets[role]),
                    calibration_evaluation_exact_vector_overlap=len(sets['calibration']&sets[role]))
                if stage=='confirmation':
                    _,old=load(d,s,'pilot');prior=set(row.tobytes() for row in old['pilot'])
                    item['pilot_confirmation_overlap']=len(prior&sets[role])
                    assert item['design_evaluation_exact_vector_overlap']==item['calibration_evaluation_exact_vector_overlap']==item['pilot_confirmation_overlap']==0
                    assert len(sets[role])==2000
                duplicates.append(item)
            counts['runs']+=1
    assert counts==dict(runs=10,policy_cases=600,group_rows=2400)
    save(ROOT/('VALIDATION_CONFIRMATION.json' if stage=='confirmation' else 'VALIDATION.json'),dict(passed=True,counts=dict(counts),exact_vector_overlap=duplicates,
        frozen_sha256=digest(ROOT/'FROZEN.json'),stage=stage,
        scope='Saved raw/summary agreement, independently rebuilt groups, hard caps, Brier means, returned/executed work; only the requested stage is loaded.'))
    print('Independent validation passed:',dict(counts),duplicates)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','confirmation'],default='pilot');args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.stage)
