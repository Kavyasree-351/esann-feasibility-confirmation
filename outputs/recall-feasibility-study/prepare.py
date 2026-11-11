"""Pin data and query roles before inspecting any new query outcomes."""
import argparse,hashlib,json,shutil,urllib.request
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent/'work'
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def prepare():
    target=ROOT/'splits.json'
    if target.exists():raise SystemExit('Split ledger already exists; refusing to change roles.')
    glove=WORK/'adaptive-retrieval/glove-25-angular.hdf5'
    coco=WORK/'diagnostic-data/coco-t2i-512-angular.hdf5'
    with h5py.File(glove) as f:gn=len(f['train'])
    with h5py.File(coco) as f:cn=len(f['train']);qn=len(f['test']);dim=f['train'].shape[1]
    rng=np.random.default_rng(73001);gperm=rng.permutation(gn)
    # Earlier cap-tail calibration inspected TRAIN queries. Exclude their union
    # before choosing any new query role; a fresh filename is not fresh data.
    previous=set()
    for p in (ROOT.parent/'github/budgeted-tail-risk-ann/results').glob('glove_*_provenance.json'):
        previous.update(json.loads(p.read_text()).get('calibration_training_ids',[]))
    eligible=gperm[100000:][~np.isin(gperm[100000:],list(previous))]
    query={'glove':eligible[:5000], 'coco_clip':rng.permutation(qn)[:5000]}
    banks={'glove':gperm[:100000], 'coco_clip':np.arange(cn)}
    records={}
    for d in query:
        ids=query[d];roles=dict(design=ids[:500].tolist(),calibration=ids[500:2500].tolist(),pilot=ids[2500:3000].tolist(),confirmation=ids[3000:5000].tolist())
        sets=[set(x) for x in roles.values()];assert all(not(a&b) for i,a in enumerate(sets) for b in sets[i+1:])
        corpora={str(s):np.sort(np.random.default_rng(81000+s).choice(banks[d],10000,replace=False)).tolist() for s in range(5)}
        for ids2 in corpora.values():assert all(not(set(ids2)&x) for x in sets) if d=='glove' else True
        records[d]=dict(roles=roles,corpora=corpora,source='train' if d=='glove' else 'test',
            dataset_sha256=digest(glove if d=='glove' else coco))
    save(target,dict(datasets=records,protocol_sha256=digest(ROOT/'PROTOCOL.md'),
        confirmation_queries_evaluated=0,old_official_final_queries_evaluated=0,
        previously_inspected_glove_train_query_ids_excluded=len(previous),
        coco_source=dict(url='https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-t2i-512-angular.hdf5',shape=[cn,qn,dim],encoder='CLIP ViT-B/16')))
    print('Frozen role IDs and dataset hashes; no query vector values read.')
def freeze():
    target=ROOT/'FROZEN.json'
    if target.exists():raise SystemExit('Already frozen; explicit amendment required to alter protocol/code.')
    files={p.name:digest(p) for p in sorted(ROOT.glob('*.py'))}
    files.update({n:digest(ROOT/n) for n in ['PROTOCOL.md','splits.json','SMOKE.json','NATIVE_AUDIT.json','author_stop_excerpt.cpp']})
    save(target,dict(files=files,confirmation_opened=False,pilot_started=False))
    print('Frozen',len(files),'source/protocol/split hashes.')
def assert_frozen():
    ledger=json.loads((ROOT/'FROZEN.json').read_text())
    for name,sha in ledger['files'].items():assert digest(ROOT/name)==sha,'Frozen input changed: '+name
def rows(ds,ids):
    ids=np.array(ids);order=np.argsort(ids);values=np.asarray(ds[ids[order]],dtype='f');return values[np.argsort(order)]
def load(d,seed,stage):
    ledger=json.loads((ROOT/'splits.json').read_text());spec=ledger['datasets'][d]
    path=WORK/('adaptive-retrieval/glove-25-angular.hdf5' if d=='glove' else 'diagnostic-data/coco-t2i-512-angular.hdf5')
    assert digest(path)==spec['dataset_sha256']
    roles=['design','calibration','pilot'] if stage=='pilot' else ['design','calibration','confirmation']
    with h5py.File(path) as f:
        base=rows(f['train'],spec['corpora'][str(seed)])
        q={role:rows(f[spec['source']],spec['roles'][role]) for role in roles}
    for x in [base,*q.values()]:x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return base,q
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['splits','freeze']);a=p.parse_args()
    prepare() if a.action=='splits' else freeze()
