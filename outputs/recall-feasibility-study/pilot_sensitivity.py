"""Descriptive duplicate-excluded pilot score check; never retunes models."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from prepare import ROOT,load
from study import write
rows=[]
for d in ['glove','coco_clip']:
    _,q=load(d,0,'pilot');prior=set(v.tobytes() for role in ['design','calibration'] for v in q[role])
    mask=np.array([v.tobytes() not in prior for v in q['pilot']])
    for s in range(5):
        with np.load(ROOT/'results/pilot'/f'{d}_{s}_raw.npz',allow_pickle=False) as z:
            a=z['occupancy_per_query_brier'][mask];b=z['geometry_per_query_brier'][mask]
            rows.append(dict(dataset=d,seed=s,retained_queries=int(mask.sum()),removed_queries=int((~mask).sum()),
                occupancy_brier=float(a.mean()),geometry_brier=float(b.mean()),relative_improvement=float(1-b.mean()/a.mean()),
                scope='Descriptive Brier sensitivity only; original gate unchanged; no retuning'))
write(ROOT/'pilot_duplicate_sensitivity.csv',rows)
print('Duplicate-excluded pilot score sensitivity saved; models and gates unchanged.')
