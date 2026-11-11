"""Evaluate the unchanged H1 criteria on locked, fresh confirmation outcomes."""
import csv,json
from pathlib import Path
import numpy as np
from prepare import ROOT,save,digest,assert_frozen
from study import write
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
    assert_frozen();lock=json.loads((ROOT/'CONFIRMATION_LOCK.json').read_text())
    assert lock['confirmation_driver_sha256']==digest(ROOT/'confirmation.py')
    for name,sha in lock['pilot_files'].items():assert digest(ROOT/name)==sha,'Pilot evidence changed'
    summaries=[];decisions=[];allpol=[];allgroup=[];floors=[];latency=[];calibration=[];gate={}
    for d in ['glove','coco_clip']:
        passed=[]
        for s in range(5):
            prefix=ROOT/'results/confirmation'/f'{d}_{s}';r=json.loads(prefix.with_suffix('.json').read_text());m=r['diagnostic_models']
            frozen=lock['chosen_settings'][f'{d}_{s}']
            assert frozen['regularization']==r['regularization']
            assert frozen['models']=={k:dict(depth=v['depth'],leaf=v['leaf']) for k,v in m.items()},'Model selection changed'
            assert r['confirmation_evaluated'] and r['frozen_sha256']==lock['frozen_inputs_sha256']
            ar=read(Path(str(prefix)+'_admissions.csv'))
            by_model={}
            for model in ['occupancy','geometry']:
                records=[a for a in ar if a['model']==model]
                admitted=[a for a in records if a['admitted']=='True'];fp=sum(a['false_admission']=='True' for a in admitted)
                fn=sum(a['admitted']!='True' and a['finite_pilot_infeasible']!='True' for a in records)
                correct=sum((a['admitted']=='True')==(a['finite_pilot_infeasible']!='True') for a in records)
                by_model[model]=(len(admitted),fp)
                decisions.append(dict(dataset=d,seed=s,model=model,regimes=len(records),admitted=len(admitted),false_admissions=fp,false_refusals=fn,correct=correct,
                    meaning='finite confirmation workload; overlapping regimes are dependent'))
            admitted,fp=by_model['geometry'];rate=fp/admitted if admitted else None
            gain=1-m['geometry']['brier']/max(m['occupancy']['brier'],1e-15)
            success=gain>=.1 and admitted>0 and rate<=.1
            if success:passed.append(s)
            summaries.append(dict(dataset=d,seed=s,occupancy_brier=m['occupancy']['brier'],geometry_brier=m['geometry']['brier'],relative_brier_improvement=gain,
                paired_ci_low=r['paired_brier_difference_ci'][0],paired_ci_high=r['paired_brier_difference_ci'][1],
                admissions=admitted,false_admissions=fp,false_admission_fraction=rate,H1_pass=success,
                lookahead_undercount=r['lookahead']['undercount_fraction'],runtime_seconds=r['timings']['total_seconds'],peak_rss_bytes=r['timings']['peak_process_rss_bytes']))
            allpol+=read(Path(str(prefix)+'_policies.csv'));allgroup+=read(Path(str(prefix)+'_groups.csv'));floors+=read(Path(str(prefix)+'_floors.csv'));latency+=read(Path(str(prefix)+'_latency.csv'))
            with np.load(str(prefix)+'_raw.npz',allow_pickle=False) as z:
                assert len(z['geometry_per_query_brier'])==len(z['occupancy_per_query_brier'])==2000
                assert abs(z['geometry_per_query_brier'].mean()-m['geometry']['brier'])<1e-12
                assert abs(z['occupancy_per_query_brier'].mean()-m['occupancy']['brier'])<1e-12
        gate[d]=dict(passing_seeds=passed,passed=len(passed)>=4)
    assert len(allpol)==600 and len(allgroup)==2400 and len(floors)==200
    for name,data in [('confirmation_seed_summary',summaries),('confirmation_decisions',decisions),('confirmation_policies',allpol),('confirmation_groups',allgroup),('confirmation_floors',floors),('confirmation_latency',latency)]:write(ROOT/(name+'.csv'),data)
    # A fifth seed/second dataset is not an independent external replication:
    # fixed query pools are shared across the five index realizations.
    aggregate=[]
    for d in ['glove','coco_clip']:
        rr=[r for r in summaries if r['dataset']==d]
        record=dict(dataset=d,seeds=5)
        for field in ['occupancy_brier','geometry_brier','relative_brier_improvement','lookahead_undercount']:
            a=np.array([r[field] for r in rr]);record[field+'_mean']=float(a.mean());record[field+'_sd']=float(a.std(ddof=1))
        record['H1_passes']=sum(r['H1_pass'] for r in rr);record['admissions']=sum(r['admissions'] for r in rr);record['false_admissions']=sum(r['false_admissions'] for r in rr)
        aggregate.append(record)
    write(ROOT/'confirmation_aggregate.csv',aggregate)
    original=json.loads((ROOT/'splits.json').read_text());confirm=json.loads((ROOT/'CONFIRMATION_SPLITS.json').read_text())
    save(ROOT/'CONFIRMATION_RESULT.json',dict(H1=gate,replicated_H1=all(v['passed'] for v in gate.values()),
        frozen_inputs_sha256=digest(ROOT/'FROZEN.json'),confirmation_lock_sha256=digest(ROOT/'CONFIRMATION_LOCK.json'),
        full_author_engine_verified=False,novelty_established=False,confidence_scope='Exploratory seed/query summaries; empirical admissions do not give a distribution-free safety guarantee.'))
    save(ROOT/'DATA_ACCESS_LEDGER.json',dict(pilot_unique_queries_per_dataset=500,confirmation_unique_queries_per_dataset=2000,
        confirmation_outcomes_evaluated=True,old_reserved_official_final_queries_evaluated=0,
        coco_unused_official_query_rows=10000-len(set(sum([v for v in original['datasets']['coco_clip']['roles'].values()],[]))|set(confirm['datasets']['coco_clip']['ids'])),
        confirmation_duplicates_removed={d:len(v['removed_duplicate_ids']) for d,v in confirm['datasets'].items()},
        note='The confirmation split file records its frozen PRE-label state; this ledger records the completed evaluation state.'))
    lines=['# Fresh confirmation: unchanged diagnostic criterion','',
        'All pilot-selected tree settings and regularization parameters matched the lock. Original 600 policy cases and 2400 group rows were repeated on 2000 fresh, duplicate-excluded queries per dataset, across all five index seeds. Query roles never crossed; no previous official final TEST queries were opened.','',
        '| Dataset | Occupancy Brier mean ± SD | Geometry Brier mean ± SD | Relative gain mean ± SD | H1 seeds | False admissions / admissions |',
        '|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"|{r['dataset']}|{r['occupancy_brier_mean']:.4f} ± {r['occupancy_brier_sd']:.4f}|{r['geometry_brier_mean']:.4f} ± {r['geometry_brier_sd']:.4f}|{r['relative_brier_improvement_mean']:.1%} ± {r['relative_brier_improvement_sd']:.1%}|{r['H1_passes']}/5|{r['false_admissions']}/{r['admissions']}|")
    lines+=['',f"Replicated H1 criterion: **{all(v['passed'] for v in gate.values())}**.",
        'Counts of admitted groups/caps across seeds are dependent and must not be interpreted as independent Bernoulli trials. A zero observed false-admission count does not prove a 10% population false-admission bound.',
        '', '## Publication assessment',
        'This confirmation strengthens only the scoped empirical diagnostic finding if the criterion passes. ExtraTrees, geometric features, CRC and LTT are established tools; none is claimed as a new statistical algorithm. Operational utility relative to the occupancy model is reported separately in confirmation_decisions.csv. The full ConANN engine remains unverified, and no conclusion about its published latency or native ndis measurements is supported. Publication readiness still requires a clear comparison with prior feasibility-diagnostic work and substantive engineering value beyond average prediction-score gains.']
    (ROOT/'CONFIRMATION_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gate=gate,aggregate=aggregate),indent=2))
if __name__=='__main__':main()
