"""All-seed reports and a hard lock against ungated confirmation."""
import csv,json
from pathlib import Path
import numpy as np
from prepare import ROOT,save,digest,assert_frozen
from study import write
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
    assert_frozen();folder=ROOT/'results/pilot';runs=[];admit=[];policies=[];groups=[];floors=[];latency=[]
    for d in ['glove','coco_clip']:
        for s in range(5):
            p=folder/f'{d}_{s}';r=json.loads(p.with_suffix('.json').read_text());runs.append(r)
            for suffix,dest in [('admissions',admit),('policies',policies),('groups',groups),('floors',floors),('latency',latency)]:dest+=read(Path(str(p)+'_'+suffix+'.csv'))
    assert len(policies)==600 and len(groups)==2400 and len(floors)==200 and len(admit)==400
    rows=[];h1={};h2={}
    for d in ['glove','coco_clip']:
        passed=[];workpass=[]
        for s in range(5):
            r=next(r for r in runs if (r['dataset'],r['seed'])==(d,s));m=r['diagnostic_models']
            gain=1-m['geometry']['brier']/max(m['occupancy']['brier'],1e-15)
            ar=[a for a in admit if (a['dataset'],int(a['seed']),a['model'])==(d,s,'geometry') and a['admitted']=='True']
            false=sum(a['false_admission']=='True' for a in ar);rate=false/len(ar) if ar else None
            success=gain>=.1 and bool(ar) and rate<=.1
            if success:passed.append(s)
            if r['lookahead']['undercount_fraction']>=.05:workpass.append(s)
            rows.append(dict(dataset=d,seed=s,occupancy_brier=m['occupancy']['brier'],geometry_brier=m['geometry']['brier'],
                relative_brier_improvement=gain,paired_difference_ci_low=r['paired_brier_difference_ci'][0],
                paired_difference_ci_high=r['paired_brier_difference_ci'][1],admissions=len(ar),false_admissions=false,
                false_admission_fraction=rate,H1_pass=success,lookahead_undercount=r['lookahead']['undercount_fraction'],
                peak_process_rss_bytes=r['timings']['peak_process_rss_bytes'],runtime_seconds=r['timings']['total_seconds']))
        h1[d]=dict(passing_seeds=passed,passed=len(passed)>=4)
        h2[d]=dict(passing_seeds=workpass,passed=len(workpass)>=4)
    native=json.loads((ROOT/'NATIVE_AUDIT.json').read_text())
    H1=all(x['passed'] for x in h1.values());H2=all(x['passed'] for x in h2.values()) and native['random_differential_cases']==10000
    allowed=H1 or (H2 and native['full_author_engine_built'])
    gate=dict(H1=h1,H2=h2,H1_pass=H1,H2_kernel_scope_pass=H2,
        full_author_engine_verified=native['full_author_engine_built'],confirmation_allowed=allowed,
        frozen_sha256=digest(ROOT/'FROZEN.json'),confirmation_queries_evaluated=0,
        old_reserved_final_queries_evaluated=0,novelty_established=False)
    for name,data in [('seed_summary',rows),('all_policy_metrics',policies),('all_group_risks',groups),('all_floor_risks',floors),('all_admissions',admit),('all_latency',latency)]:write(ROOT/(name+'.csv'),data)
    save(ROOT/'gate.json',gate)
    lines=['# Prospective diagnostic extension: findings','',
        'All ten dataset/index-seed pilots retained: GloVe-25 and COCO CLIP text-to-image retrieval. 600 cap/calibration-size/method conditions, 2400 group rows, 200 floor rows. Query pools are shared across seeds; these counts are not independent external replications.','',
        '| Dataset | Seed | Occupancy Brier | Geometry Brier | Relative gain | False admissions | H1 pass | Work undercount |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"|{r['dataset']}|{r['seed']}|{r['occupancy_brier']:.4f}|{r['geometry_brier']:.4f}|{r['relative_brier_improvement']:.1%}|{r['false_admissions']}/{r['admissions']}|{r['H1_pass']}|{r['lookahead_undercount']:.1%}|")
    lines+=['',f'Geometry diagnostic gate: **{H1}**. Native replay/lookahead pilot gate: **{H2}**. Full author engine verified: **False**.',
        f'Fresh confirmation permitted: **{allowed}**. No confirmation rows or previously reserved final queries have been evaluated.','',
        '## Interpretation',
        '- The geometry model uses design labels and is a predictor, not a label-free or distribution-free infeasibility certificate.',
        '- Counting returned prefixes can omit lookahead work in this source-level replay. The unmodified full ConANN/Faiss engine could not be built locally because BLAS/MKL was missing. Its published performance claims have not been reproduced or refuted.',
        '- Author-score adapters differ from the original GSL root solver and full engine. Marginal CRC references do not certify simultaneous tail/group risk; LTT adapters share the stronger objective.',
        '- The actual author scanner can truncate lists with max_codes; whole-list prefix results do not cover arbitrary partial-list operation.',
        '- Do not promote this pilot into independent confirmation, a universal calibration impossibility, or a proven new method.','',
        '## Next action',
        'If the gates are closed, keep confirmation sealed and report the negative extension. Do not change thresholds or keep searching the same pilot for a success. If H1 passes, freeze the current hashes and use the gated confirmation command. If only the kernel-scope H2 passes, validate the full author engine on a supported Linux/BLAS environment first.']
    (ROOT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
