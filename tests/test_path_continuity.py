from pathlib import Path
import sys, unittest, importlib.util, pickle, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine/runtime'))
from sequence_policy import select
from path_state import remember,valid

def rails(end,z=0):
    u=np.arange(0.,end+.01,.5)
    return np.stack((np.column_stack((u*0-.78,-u,u*0+z)),np.column_stack((u*0+.78,-u,u*0+z))),1)

class PolicyTests(unittest.TestCase):
    def choose(self,new,old,**kw):
        return select(new,old,rails(8),contact_status='RIGHT',fresh_near=True,**kw)
    def test_short_success_retains_observed_tail(self):
        r=self.choose(rails(40),rails(100));self.assertTrue(r['fallback']);self.assertEqual(-r['pair'][-1,:,1].mean(),100)
    def test_prediction_length_is_not_observed_length(self):
        r=self.choose(rails(105),rails(100),primary_observed_end=85)
        self.assertTrue(r['fallback'])
    def test_new_long_path_replaces_old(self):
        self.assertFalse(self.choose(rails(110),rails(100))['fallback'])
    def test_near_contradiction_vetoes_cache(self):
        self.assertFalse(self.choose(rails(40),rails(100,z=.2))['fallback'])
    def test_far_contradiction_requests_new_measurement(self):
        p=rails(40);p[:,:,2]+=np.clip((-p[:,:,1]-10)/10,0,1)*.3
        r=self.choose(p,rails(100));self.assertEqual(r['scope'],'NEAR_ONLY_CONFLICTING_PATHS');self.assertEqual(len(r['pair']),17)
    def test_no_old_path_no_invention(self):
        self.assertFalse(self.choose(rails(40),None)['fallback'])
    def test_remember_does_not_refresh_source_age(self):
        base=dict(meta=dict(source_frame=1,generation=0,deadline=0.))
        meta=dict(source_frame=40,generation=0,deadline=4.)
        s=remember(rails(100),np.eye(4),meta,base)
        self.assertEqual(s['meta']['deadline'],0.)
        self.assertTrue(valid(s,meta,4.));self.assertFalse(valid(s,meta,5.01))
        other=dict(meta,generation=1);self.assertFalse(valid(s,other,4.))

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'engine/baseline/app'))
        from MVP.realtime_final.native_api import load
        cls.native=load()
    def run_cloud(self,gap=False):
        angle=np.linspace(0,np.pi,25)
        t=np.column_stack((.06*np.cos(angle),.04*np.sin(angle)))
        u=np.arange(.2,8,.4)
        if gap:u=np.r_[u,np.arange(17.,42.,.4)]
        cloud=np.array([[v,*q] for v in u for q in t])
        ids=np.flatnonzero(cloud[:,0]<8).astype(np.int64)
        m=self.native.Marcher(1,12,0);m.add_frame(0,cloud)
        result=m.extend_track(np.zeros(3),np.eye(3),ids,t,np.eye(4),np.eye(3),0,[0],1.)
        return m,result,cloud,ids,t
    def test_gap_reacquisition_uses_real_points(self):
        _,r,cloud,_,_=self.run_cloud(True)
        keys=np.concatenate([b['keys'] for b in r['blocks']]).astype(np.int64)
        self.assertEqual(len(keys),len(np.unique(keys)))
        self.assertTrue(np.all(keys<len(cloud)))
        confirmed=np.concatenate([b['keys'] for b in r['blocks'] if b['state']==2]).astype(np.int64)
        self.assertGreater(cloud[confirmed,0].max(),30.)
    def test_empty_tail_is_not_fabricated(self):
        _,r,_,_,_=self.run_cloud(False);self.assertEqual(len(r['blocks']),1)
    def test_future_cloud_is_rejected(self):
        m,_,cloud,ids,t=self.run_cloud(False);m.add_frame(1,cloud)
        with self.assertRaises(RuntimeError):m.extend_track(np.zeros(3),np.eye(3),ids,t,np.eye(4),np.eye(3),0,[0,1],1.)

if __name__=='__main__':unittest.main(verbosity=2)
