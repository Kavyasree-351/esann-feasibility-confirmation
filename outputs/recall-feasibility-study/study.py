"""Prospective small-memory diagnostic study. Confirmation is gate-locked."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,csv,json,time
from pathlib import Path
import numpy as np,psutil
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from prepare import ROOT,load,save,assert_frozen,digest
from engine import IVF,distances,M,K
from control import *
CAPS=[.05,.1,.25,.5,1.];NCAL=[100,300,1000,2000]
def write(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        path.write_text('dataset,seed,method,repeat,queries,seconds,ms_per_query\n');return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def bootstrap(v):
    rng=np.random.default_rng(123);means=[]
    for _ in range(2000):means.append(v[rng.integers(0,len(v),len(v))].mean())
    return np.quantile(means,[.025,.975]).tolist()
def models(idx,dz,pz,scale):
    features=[];target=[];pilot_features=[];pilot_targets=[]
    for cap in CAPS:
        B=M+int(cap*len(idx.x));features.append(idx.features(dz,B,scale));_,v=floor(dz,B);target.append(components(v))
        pilot_features.append(idx.features(pz,B,scale));_,v=floor(pz,B);pilot_targets.append(components(v))
    # query-major arrangement ensures query identity never crosses tuning roles.
    x=np.stack(features,axis=1);y=np.stack(target,axis=1);xp=np.stack(pilot_features,axis=1);yp=np.stack(pilot_targets,axis=1)
    results={};predictions={};tuning=[]
    for name,width in [('occupancy',4),('geometry',x.shape[-1])]:
        choices=[]
        for depth in [3,6]:
            for leaf in [10,30]:
                model=ExtraTreesRegressor(n_estimators=64,max_depth=depth,min_samples_leaf=leaf,random_state=900,n_jobs=1)
                model.fit(x[:300,:,:width].reshape(-1,width),y[:300].reshape(-1,3))
                pred=model.predict(x[300:,:,:width].reshape(-1,width)).reshape(200,len(CAPS),3)
                score=float(np.mean((pred[:,:,1:]-y[300:,:,1:])**2))
                tuning.append(dict(model=name,depth=depth,leaf=leaf,validation_brier=score))
                choices.append((score,depth,leaf))
        _,depth,leaf=min(choices)
        model=ExtraTreesRegressor(n_estimators=64,max_depth=depth,min_samples_leaf=leaf,random_state=900,n_jobs=1)
        model.fit(x[:,:,:width].reshape(-1,width),y.reshape(-1,3))
        pred=model.predict(xp[:,:,:width].reshape(-1,width)).reshape(len(xp),len(CAPS),3)
        per_query=((pred[:,:,1:]-yp[:,:,1:])**2).mean((1,2))
        predictions[name]=pred
        results[name]=dict(brier=float(per_query.mean()),per_query=per_query,depth=depth,leaf=leaf)
    delta=results['geometry']['per_query']-results['occupancy']['per_query']
    return results,predictions,yp,tuning,bootstrap(delta)
def run(dataset,seed,stage):
    started=time.perf_counter();base,q=load(dataset,seed,stage);idx=IVF(base,seed)
    role='pilot' if stage=='pilot' else 'confirmation'
    dz=idx.traces(q['design']);cz=idx.traces(q['calibration']);pz=idx.traces(q[role])
    da=np.sqrt(np.sort(dz['coarse'],axis=1)[:,0]/np.maximum(np.sort(dz['coarse'],axis=1)[:,1],1e-15))
    edges=np.quantile(da,np.arange(.1,1,.1));group_edges=np.quantile(da,[.8,.9,.95])
    scale=float(np.median(np.sqrt(np.min(dz['coarse'],axis=1))))
    cells=lambda z:np.searchsorted(edges,np.sqrt(np.sort(z['coarse'],axis=1)[:,0]/np.maximum(np.sort(z['coarse'],axis=1)[:,1],1e-15)))
    dc,cc,pc=cells(dz),cells(cz),cells(pz);cg=groups(cz['coarse'],group_edges);pg=groups(pz['coarse'],group_edges)
    t=time.perf_counter();reg,reg_tuning=tune_score(dz);path=paths(dz,dc,reg)
    design_seconds=time.perf_counter()-t
    t=time.perf_counter();mr,pred,observed,tuning,brier_ci=models(idx,dz,pz,scale)
    model_seconds=time.perf_counter()-t
    admissions=[]
    for name in ['occupancy','geometry']:
        for ci,f in enumerate(CAPS):
            for g in range(4):
                take=pg[:,g];n=int(take.sum())
                predicted=pred[name][take,ci].mean(0) if n else np.full(3,np.nan)
                measured=observed[take,ci].mean(0) if n else np.full(3,np.nan)
                admitted=bool(n and np.all(predicted<=TARGET))
                bad=bool(n and np.any(measured>TARGET+1e-12))
                admissions.append(dict(dataset=dataset,seed=seed,model=name,cap_fraction=f,group=g,n=n,
                    admitted=admitted,finite_pilot_infeasible=bad,false_admission=admitted and bad,
                    predicted_mean=float(predicted[0]),observed_mean=float(measured[0])))
    policy_rows=[];group_rows=[];floor_rows=[];timings=[];raw={};selected={};cal_start=time.perf_counter()
    for f in CAPS:
        cap=M+int(f*len(base));_,floor_loss=floor(pz,cap)
        for g in range(4):
            mask=pg[:,g]
            floor_rows.append(dict(dataset=dataset,seed=seed,cap_fraction=f,group=g,n=int(mask.sum()),
                **dict(zip(['mean','tail20','tail50'],risks(floor_loss,pg)[g].tolist()))))
        for ncal in NCAL:
            cs=sub(cz,slice(0,ncal));gs=cg[:ncal];cell=cc[:ncal]
            for kind in ['uniform','cell','score']:
                selection=time.perf_counter();param,status=select(path[kind],kind,cs,gs,cell,cap,reg)
                selection_seconds=time.perf_counter()-selection
                effective=param if param is not None else (64 if kind=='uniform' else np.full(10,64) if kind=='cell' else np.inf)
                v,w,p,visited=evaluate(pz,kind,effective,cap,pc,reg)
                risk_values=risks(v,pg);observed_viols=int(((risk_values>TARGET+1e-12)&(pg.sum(0)[:,None]>0)).sum())
                key=f'{f}__{ncal}__{kind}'
                raw[key+'__loss']=v;raw[key+'__work']=w
                policy_rows.append(dict(dataset=dataset,seed=seed,cap_fraction=f,ncal=ncal,method=kind,status=status,
                    diagnostic_only=status!='issued',mean_fnr=float(v.mean()),mean_executed_distances=float(w.mean()),
                    mean_returned_distances=float(pz['work'][np.arange(len(p)),p].mean()),max_distances=int(w.max()),
                    cap_violations=int((w>cap).sum()),empirical_risk_violations=observed_viols,
                    min_calibration_group_count=int(gs.sum(0).min()),selection_seconds=selection_seconds))
                for gi in range(4):
                    group_rows.append(dict(dataset=dataset,seed=seed,cap_fraction=f,ncal=ncal,method=kind,status=status,
                        group=gi,n=int(pg[:,gi].sum()),mean_fnr=float(risk_values[gi,0]),tail20=float(risk_values[gi,1]),tail50=float(risk_values[gi,2])))
                if f==.25 and ncal==2000 and status=='issued':selected[kind]=param
    calibration_seconds=time.perf_counter()-cal_start
    # Weaker marginal-CRC reference, explicitly not the unmodified author engine.
    theta=score_crc(cz,reg);cap=M+int(.25*len(base));v,w,p,visited=evaluate(pz,'score',theta,cap,pc,reg)
    returned=pz['work'][np.arange(len(p)),p];under=float((w.mean()-returned.mean())/w.mean())
    # A query's return loss can remain unchanged despite an extra scanned list.
    raw['crc_reference_loss']=v;raw['crc_reference_executed']=w;raw['crc_reference_returned']=returned
    raw['geometry_per_query_brier']=mr['geometry'].pop('per_query');raw['occupancy_per_query_brier']=mr['occupancy'].pop('per_query')
    # Timed actual scans, balanced order, labels consulted only AFTER timing.
    for repeat in range(3):
        kinds=list(selected);kinds=kinds[repeat%max(len(kinds),1):]+kinds[:repeat%max(len(kinds),1)]
        for kind in kinds:
            before=time.perf_counter();outputs=[idx.serve(query,kind,selected[kind],cap,edges,reg) for query in q[role][:24]]
            elapsed=time.perf_counter()-before
            expected_v,expected_w,*_=evaluate(sub(pz,slice(0,24)),kind,selected[kind],cap,pc[:24],reg)
            for i,(ids,work,prefix) in enumerate(outputs):
                actual=1-len(set(ids)&set(pz['truth'][i]))/K
                assert abs(actual-expected_v[i])<1e-12,(kind,i,'live/trace recall mismatch')
                assert work==expected_w[i],(kind,i,'live/trace executed-work mismatch')
            timings.append(dict(dataset=dataset,seed=seed,method=kind,repeat=repeat,queries=24,seconds=elapsed,ms_per_query=elapsed/24*1000))
    result=dict(dataset=dataset,seed=seed,stage=stage,diagnostic_models=mr,diagnostic_tuning=tuning,
        regularization=reg,regularization_tuning=reg_tuning,paired_brier_difference_ci=brier_ci,
        lookahead=dict(reference='author-score discrete CRC adaptation',executed_mean=float(w.mean()),returned_mean=float(returned.mean()),
            undercount_fraction=under,affected_queries=int((w>returned).sum()),query_count=len(v),full_engine_verified=False),
        timings=dict(index_build_seconds=idx.build_seconds,design_truth_seconds=dz['seconds'],calibration_truth_seconds=cz['seconds'],
            evaluation_truth_seconds=pz['seconds'],path_design_seconds=design_seconds,diagnostic_learning_seconds=model_seconds,
            calibration_grid_selection_seconds=calibration_seconds,total_seconds=time.perf_counter()-started,
            peak_process_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)),
        frozen_sha256=digest(ROOT/'FROZEN.json'),confirmation_evaluated=stage=='confirmation')
    folder=ROOT/'results'/stage;folder.mkdir(parents=True,exist_ok=True);prefix=folder/f'{dataset}_{seed}'
    save(prefix.with_suffix('.json'),result)
    for name,records in [('policies',policy_rows),('groups',group_rows),('floors',floor_rows),('admissions',admissions),('latency',timings)]:write(Path(str(prefix)+'_'+name+'.csv'),records)
    np.savez_compressed(str(prefix)+'_raw.npz',**raw)
    print(dataset,seed,'Brier relative gain',round(1-mr['geometry']['brier']/mr['occupancy']['brier'],3),'lookahead undercount',round(under,3),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','confirmation']);p.add_argument('--datasets',nargs='+',default=['glove','coco_clip'],choices=['glove','coco_clip']);p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert_frozen();assert json.loads((ROOT/'SMOKE.json').read_text())['passed']
    if a.stage=='confirmation':
        gate=json.loads((ROOT/'gate.json').read_text())
        assert gate['confirmation_allowed'] and gate['frozen_sha256']==digest(ROOT/'FROZEN.json'),'Confirmation gate closed'
    with threadpool_limits(limits=1):
        for d in a.datasets:
            for s in a.seeds:
                if a.resume and (ROOT/'results'/a.stage/f'{d}_{s}.json').exists():continue
                run(d,s,a.stage)
if __name__=='__main__':main()
