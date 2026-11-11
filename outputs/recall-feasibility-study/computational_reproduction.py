"""Rerun known query roles in a NEW sibling folder; never overwrite evidence.

This is computational reproduction, not a fresh scientific confirmation.
Input datasets must already exist in the shared workspace work/ folder.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=ROOT.parent/'recall-feasibility-reproduction');a=p.parse_args();dst=a.destination.resolve()
    assert not dst.exists(),'Destination must be new; refusing to overwrite any experiment'
    assert dst.parent==ROOT.parent,'Use a sibling under outputs/ so the pinned data-cache layout remains valid'
    dst.mkdir()
    for f in ROOT.iterdir():
        if f.is_file() and f.suffix in {'.py','.md','.json','.txt','.cpp'}:shutil.copyfile(f,dst/f.name)
    original=json.loads((ROOT/'CONFIRMATION_LOCK.json').read_text())
    def run(*args):subprocess.run([sys.executable,'-B',*args],cwd=dst,check=True)
    run('smoke.py');run('study.py','pilot');run('report.py');run('validate_results.py','--stage','pilot')
    assert json.loads((dst/'gate.json').read_text())['confirmation_allowed'],'Recomputed pilot gate failed; retain and investigate'
    files={};settings={}
    for f in sorted((dst/'results/pilot').glob('*')):
        files[f.relative_to(dst).as_posix()]=digest(f)
        if f.suffix=='.json':
            r=json.loads(f.read_text());settings[f.stem]=dict(regularization=r['regularization'],models={k:dict(depth=v['depth'],leaf=v['leaf']) for k,v in r['diagnostic_models'].items()})
    assert settings==original['chosen_settings'],'Recomputed design selection differs; no forced agreement'
    new=dict(original,pilot_files=files,chosen_settings=settings,gate_sha256=digest(dst/'gate.json'),
        computational_reproduction=True,fresh_scientific_confirmation=False)
    (dst/'CONFIRMATION_LOCK.json').write_text(json.dumps(new,indent=2)+'\n')
    run('confirmation.py','run');run('confirmation_report.py');run('validate_results.py','--stage','confirmation')
    # Preserve all timing variation. Compare scientific predictions, not clocks.
    comparisons=[]
    for stage in ['pilot','confirmation']:
        for f in sorted((ROOT/'results'/stage).glob('*.json')):
            a0=json.loads(f.read_text());b=json.loads((dst/'results'/stage/f.name).read_text())
            for model in ['geometry','occupancy']:
                error=abs(a0['diagnostic_models'][model]['brier']-b['diagnostic_models'][model]['brier'])
                comparisons.append(dict(stage=stage,run=f.stem,model=model,brier_absolute_difference=error))
                assert error<1e-10,'Scientific result mismatch; inspect retained rerun'
    (dst/'COMPUTATIONAL_REPRODUCTION.json').write_text(json.dumps(dict(
        fresh_scientific_confirmation=False,known_query_roles_reused=True,
        source_evidence_modified=False,comparisons=comparisons),indent=2)+'\n')
    print('Computational reproduction completed in new folder; original evidence preserved.')
if __name__=='__main__':main()
