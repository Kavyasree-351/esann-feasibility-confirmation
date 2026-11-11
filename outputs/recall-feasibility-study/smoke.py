"""Correctness controls before any new pilot outcomes."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from prepare import ROOT,save
from engine import IVF,distances,score_value
from control import floor,evaluate,groups,pvalues,select,paths
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(100);cls.base=rng.normal(size=(160,6)).astype('f');cls.q=rng.normal(size=(25,6)).astype('f')
        cls.idx=IVF(cls.base,1,8);cls.z=cls.idx.traces(cls.q)
        d=np.sort(cls.z['coarse'],axis=1);a=np.sqrt(d[:,0]/d[:,1]);cls.edges=np.quantile(a,np.arange(.1,1,.1));cls.cells=np.searchsorted(cls.edges,a)
    def test_truth_and_prefix_nesting(self):
        truth=np.argsort(distances(self.q,self.base),axis=1,kind='stable')[:,:10]
        np.testing.assert_array_equal(truth,self.z['truth'])
        self.assertTrue(np.all(np.diff(self.z['loss'],axis=1)<=1e-12))
    def test_live_executed_counts_and_losses(self):
        for cap in [8,70,168]:
            for kind,param in [('uniform',0),('uniform',4),('cell',np.arange(10)%9),('score',.01),('score',.08),('score',np.inf)]:
                v,w,*_=evaluate(self.z,kind,param,cap,self.cells,.01)
                for i,q in enumerate(self.q):
                    ids,work,prefix=self.idx.serve(q,kind,param,cap,self.edges,.01)
                    actual=1-len(set(ids)&set(self.z['truth'][i]))/10
                    self.assertAlmostEqual(v[i],actual,places=12)
                    self.assertEqual(work,w[i]);self.assertLessEqual(work,cap)
    def test_floor_dominance(self):
        for cap in [8,70,168]:
            p,v=floor(self.z,cap)
            self.assertTrue(np.all(np.where(self.z['work']<=cap,self.z['loss']>=v[:,None]-1e-12,True)))
    def test_empty_group_is_unknown(self):
        pv=pvalues(np.zeros(200),np.c_[np.ones(200,bool),np.zeros(200,bool)])
        self.assertTrue(np.all(pv[1]==1))
    def test_tail_count_threshold(self):
        self.assertGreater(.95**71,.025);self.assertLessEqual(.95**72,.025)
    def test_lookahead_is_not_free(self):
        score=score_value(np.array([3.,2.,1.]),0,np.arange(1,4),3)
        theta=(score[0]+score[1])/2
        self.assertEqual(int((score<=theta).sum()),1)
        self.assertEqual(min(int((score<=theta).sum())+1,3),2)
if __name__=='__main__':
    with threadpool_limits(limits=1):result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    save(ROOT/'SMOKE.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,final_queries_opened=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
