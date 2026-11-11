"""Small-memory exact-within-list IVF and honest executed-work accounting."""
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
M=64;K=10
def distances(q,x):
    q=np.asarray(q,dtype='d');x=np.asarray(x,dtype='d')
    return np.maximum((q*q).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*q@x.T,0)
def topk(v,k=K):
    # Explicit boundary ties match stable full sorting by corpus row ID.
    cut=np.partition(v,k-1)[k-1];ids=np.r_[np.flatnonzero(v<cut),np.flatnonzero(v==cut)[:k-int((v<cut).sum())]]
    return ids[np.argsort(v[ids],kind='stable')]
class IVF:
    def __init__(self,x,seed,m=M):
        self.x=x;self.m=m;t=time.perf_counter()
        model=MiniBatchKMeans(n_clusters=m,random_state=seed,n_init=3,batch_size=1024,max_iter=100)
        self.labels=model.fit_predict(x);self.centers=model.cluster_centers_
        self.ids=[np.flatnonzero(self.labels==i) for i in range(m)];self.sizes=np.array([len(x) for x in self.ids])
        self.radii=np.array([np.sqrt(distances(self.centers[i:i+1],x[ids])).max() if len(ids) else 0. for i,ids in enumerate(self.ids)])
        self.build_seconds=time.perf_counter()-t
    def traces(self,q):
        t=time.perf_counter();n=len(q);coarse=distances(q,self.centers);order=np.argsort(coarse,axis=1,kind='stable')
        ranks=np.argsort(order,axis=1);loss=np.ones((n,self.m+1));work=np.c_[np.zeros(n,int),np.cumsum(self.sizes[order],axis=1)]+self.m
        kth=np.full_like(loss,np.inf);truth=np.empty((n,K),int)
        for lo in range(0,n,16):
            ds=distances(q[lo:lo+16],self.x)
            for local,d in enumerate(ds):
                i=lo+local;truth[i]=topk(d);r=ranks[i,self.labels[truth[i]]]
                loss[i,1:]=1-(r[:,None]<=np.arange(self.m)).mean(0)
                best=np.full(K,np.inf)
                for j,c in enumerate(order[i],1):
                    best=np.partition(np.r_[best,d[self.ids[c]]],K-1)[:K]
                    kth[i,j]=best.max()
        assert np.all(np.diff(loss,axis=1)<=1e-12) and np.all(loss[:,-1]==0)
        assert np.all(np.diff(kth[:,1:],axis=1)<=1e-12)
        return dict(loss=loss,work=work,kth=kth,coarse=coarse,order=order,truth=truth,seconds=time.perf_counter()-t)
    def features(self,z,cap,scale):
        p=(z['work']<=cap).sum(1)-1;n=len(p);i=np.arange(n);order=z['order'];sd=np.sort(z['coarse'],axis=1)
        radius=self.radii[order];small=np.maximum(sd[:,1],1e-15)
        next_p=np.minimum(p,self.m-1)
        return np.column_stack([np.full(n,(cap-self.m)/len(self.x)),p/self.m,
            (z['work'][i,p]-self.m)/len(self.x),self.sizes[order[i,next_p]]/len(self.x),
            np.sqrt(sd[:,0]/small),np.sqrt(sd[:,0])/max(scale,1e-12),
            np.sqrt(sd[:,0])/np.maximum(np.sqrt(sd[:,-1]),1e-12),
            radius[:,0]/max(scale,1e-12),radius[i,next_p]/max(scale,1e-12)])
    def serve(self,q,kind,param,cap,edges,lamreg=0.):
        """Actual list scans. No truth, precomputed kth distances or loss curves."""
        coarse=distances(q[None,:],self.centers)[0];order=np.argsort(coarse,kind='stable')
        ambiguity=np.sqrt(coarse[order[0]]/max(coarse[order[1]],1e-15));cell=int(np.searchsorted(edges,ambiguity))
        budget=int(param) if kind=='uniform' else int(param[cell]) if kind=='cell' else self.m
        candidates=np.array([],int);all_d=np.array([],float);executed=self.m;returned=np.array([],int);count=0
        for j,c in enumerate(order,1):
            if executed+len(self.ids[c])>cap or (kind!='score' and j>budget):break
            prior=returned.copy();candidates=np.r_[candidates,self.ids[c]];executed+=len(self.ids[c])
            new=distances(q[None,:],self.x[self.ids[c]])[0]
            all_d=np.r_[all_d,new]
            # Stable by corpus ID for equal distances, irrespective of list order.
            ids_order=np.argsort(candidates,kind='stable');take=ids_order[np.argsort(all_d[ids_order],kind='stable')[:K]]
            returned=candidates[take];count=j
            if kind=='score':
                kth=float(np.max(all_d[take])) if len(take)==K else np.inf
                score=float(score_value(kth,lamreg,j,self.m))
                if score>float(param):returned=prior;count=j-1;break
        return returned,executed,count

def score_value(kth,reg,j,m=M):
    reg=np.float32(reg)
    return (np.float32(1)-np.minimum(np.float32(kth)/np.float32(4),np.float32(1))+
            reg*np.float32(np.maximum(0,np.asarray(j)-1)))/(np.float32(11)+reg*np.float32(m-1))
def scores(z,reg):return score_value(z['kth'][:,1:],reg,np.arange(1,z['loss'].shape[1]),z['loss'].shape[1]-1)
