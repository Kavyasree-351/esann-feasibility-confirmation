"""Matched LTT selection; separate weaker marginal CRC reference."""
import numpy as np
from engine import scores
TARGET=np.array([.1,.1,.05])
def components(v):return np.stack([v,v>.2+1e-12,v>.5+1e-12],axis=-1).astype(float)
def groups(coarse,edges):
    sd=np.sort(coarse,axis=1);a=np.sqrt(sd[:,0]/np.maximum(sd[:,1],1e-15))
    return np.column_stack([np.ones(len(a),bool)]+[a>=v for v in edges])
def risks(v,g):return g.T@components(v)/np.maximum(g.sum(0)[:,None],1)
def pvalues(v,g,reverse=False):
    z=risks(v,g);ng=g.sum(0);clip=np.clip(z,1e-15,1-1e-15)
    kl=clip*np.log(clip/TARGET)+(1-clip)*np.log((1-clip)/(1-TARGET))
    valid=z>TARGET if reverse else z<TARGET
    pv=np.where(valid,np.exp(-ng[:,None]*kl),1.);pv[ng==0]=1
    return pv
def floor(z,cap):
    p=(z['work']<=cap).sum(1)-1;assert np.all(p>=0)
    return p,z['loss'][np.arange(len(p)),p]
def evaluate(z,kind,param,cap,cells,reg):
    pmax,_=floor(z,cap);n=len(pmax);i=np.arange(n)
    if kind=='uniform':p=np.full(n,int(param))
    elif kind=='cell':p=np.asarray(param)[cells]
    else:p=(scores(z,reg)<=float(param)).sum(1)
    p=np.minimum(p,pmax);visited=np.minimum(p+1,pmax) if kind=='score' else p
    return z['loss'][i,p],z['work'][i,visited],p,visited
def crc_probe(loss):
    ok=np.flatnonzero((loss.sum(0)+1)/(len(loss)+1)<=.1)
    return int(ok[0]) if len(ok) else loss.shape[1]-1
def score_crc(z,reg):
    sc=scores(z,reg);grid=np.unique(sc)
    lo=-1;hi=len(grid)
    cells=np.zeros(len(sc),int);cap=int(z['work'][0,-1])
    # Smallest threshold passing conservative discrete CRC; no GSL approximation.
    while hi-lo>1:
        mid=(lo+hi)//2;v,*_=evaluate(z,'score',grid[mid],cap,cells,reg)
        if (v.sum()+1)/(len(v)+1)<=.1:hi=mid
        else:lo=mid
    return float(grid[hi]) if hi<len(grid) else float('inf')
def sub(z,take):return {k:v[take] for k,v in z.items() if isinstance(v,np.ndarray)}
def tune_score(design):
    records=[];chosen=0.;best=float('inf')
    for reg in [0.,.001,.01,.1]:
        theta=score_crc(sub(design,slice(0,300)),reg)
        v,w,*_=evaluate(sub(design,slice(300,None)),'score',theta,int(design['work'][0,-1]),np.zeros(200,int),reg)
        record=dict(regularization=reg,theta=theta,validation_fnr=float(v.mean()),validation_work=float(w.mean()))
        records.append(record)
        if v.mean()<=.1 and w.mean()<best:best=float(w.mean());chosen=reg
    return chosen,records
def paths(design,cells,reg):
    m=design['loss'].shape[1]-1
    template=np.array([crc_probe(design['loss'][cells==j]) for j in range(10)])
    cell=np.unique(np.stack([np.clip(template+i,0,m) for i in range(-m,m+1)]),axis=0)
    # np.unique sorts lexicographically; common-offset vectors remain nested.
    assert np.all(np.diff(cell,axis=0)>=0)
    return dict(uniform=list(range(m+1)),cell=list(cell),
        score=list(np.unique(np.r_[-np.inf,np.quantile(scores(design,reg),np.linspace(0,1,65)),np.inf])))
def select(path,kind,z,g,cells,cap,reg):
    _,v=floor(z,cap);witness=bool(pvalues(v,g,True).min()<=.025/(g.shape[1]*3))
    if witness:return None,'infeasible'
    last=None
    for param in reversed(path):
        v,*_=evaluate(z,kind,param,cap,cells,reg)
        exact=cap>=z['work'][0,-1] and ((kind=='score' and np.isposinf(param)) or (kind!='score' and np.all(np.asarray(param)==z['loss'].shape[1]-1)))
        joint=0. if exact else pvalues(v,g).max()
        if joint>.025:break
        last=param
    return last,'issued' if last is not None else 'unknown'
