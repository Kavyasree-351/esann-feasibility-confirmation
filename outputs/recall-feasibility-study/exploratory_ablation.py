"""Post-confirmation mechanism checks on PILOT only; not new confirmation.

All choices are recorded here before this ablation runs. Original learning code,
models, gates, and confirmation results are immutable. Geometry's selected tree
settings are reused for all subset/null controls; no retuning on pilot outcomes.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from prepare import ROOT,load,assert_frozen
from engine import IVF
from control import components,floor
from study import CAPS,write
def main():
    assert_frozen();records=[]
    for d in ['glove','coco_clip']:
        for s in range(5):
            base,q=load(d,s,'pilot');idx=IVF(base,s);dz=idx.traces(q['design']);pz=idx.traces(q['pilot'])
            scale=float(np.median(np.sqrt(dz['coarse'].min(1))))
            features=lambda z:np.stack([idx.features(z,64+int(f*10000),scale) for f in CAPS],axis=1)
            targets=lambda z:np.stack([components(floor(z,64+int(f*10000))[1]) for f in CAPS],axis=1)
            x=features(dz);xp=features(pz);y=targets(dz);yp=targets(pz)
            original=json.loads((ROOT/'results/pilot'/f'{d}_{s}.json').read_text());params=original['diagnostic_models']['geometry']
            variants={'full_geometry':list(range(9)),'ambiguity_only':list(range(5)),
                'distance_only':[0,1,2,3,5,6],'radii_only':[0,1,2,3,7,8],'shuffled_geometry':list(range(9))}
            for name,cols in variants.items():
                xx=x[:,:,cols].copy();pp=xp[:,:,cols].copy()
                if name=='shuffled_geometry':
                    rng=np.random.default_rng(8132+s)
                    xx[:,:,4:]=xx[rng.permutation(500),:,4:];pp[:,:,4:]=pp[rng.permutation(500),:,4:]
                model=ExtraTreesRegressor(n_estimators=64,max_depth=params['depth'],min_samples_leaf=params['leaf'],random_state=900,n_jobs=1)
                model.fit(xx.reshape(-1,len(cols)),y.reshape(-1,3))
                t=time.perf_counter();prediction=model.predict(pp.reshape(-1,len(cols))).reshape(500,len(CAPS),3)
                inference=time.perf_counter()-t;brier=float(((prediction[:,:,1:]-yp[:,:,1:])**2).mean())
                if name=='full_geometry':assert abs(brier-params['brier'])<1e-12,'Ablation setup does not reproduce original model'
                records.append(dict(dataset=d,seed=s,variant=name,brier=brier,depth=params['depth'],leaf=params['leaf'],
                    batch_prediction_seconds=inference,query_cap_rows=2500,
                    scope='Post-confirmation exploratory ablation on old pilot only; no new confirmation or gate'))
            print('Exploratory ablation',d,s,flush=True)
    write(ROOT/'exploratory_ablation.csv',records)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
