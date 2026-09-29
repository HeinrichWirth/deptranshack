import unittest
import numpy as np
from MVP.final_pipeline.spatial_scheduler import SpatialScheduler,transform_prediction
from MVP.performance_final.direction import direction


def records(x):
    rows=[]
    for i,v in enumerate(x):
        P=np.eye(4);P[0,3]=v
        rows.append(dict(lidar_pose_in_folder=P.tolist(),header_time_ns=i*100000000,pose_status='ok',pose_uses_future=False))
    return rows


class Fake:
    def __init__(self,positions,fail=()):
        self.records=records(positions);self.run='test';self.cache=None;self.calls=[];self.fail=fail
    def process_frame(self,i):
        self.calls.append(i);C=np.array([[0.,0.,0.],[8.,0.,0.]])
        p=dict(C=C,B=np.repeat(np.eye(3)[None],2,axis=0),center=C,s=np.array([0.,8.]),pair=np.stack((C+[0,-.8,0],C+[0,.8,0]),axis=1))
        return dict(prediction=None if i in self.fail else p,c4_smooth=dict(C=C,B=p['B'],s=p['s']),
            summary=dict(original_reason='failed' if i in self.fail else '',timing=dict(T_TOTAL=.01)),
            cloud=dict(xyz=np.zeros((1,3)),source_frame=np.array([i])))


class SchedulerTests(unittest.TestCase):
    def test_accumulated_threshold(self):
        e=Fake([0,.28,.57]);s=SpatialScheduler(e)
        a,b,c=[s.process_frame(i) for i in range(3)]
        self.assertEqual(e.calls,[0,2]);self.assertEqual(b['summary']['status'],'GEOMETRY_REUSED_LOW_MOTION')
        np.testing.assert_allclose(b['prediction']['center'],a['prediction']['center']-[.28,0,0])
        self.assertEqual(c['summary']['geometry_age_frames'],0)
    def test_exact_half_meter_updates(self):
        e=Fake([0,.5]);s=SpatialScheduler(e);s.process_frame(0);s.process_frame(1);self.assertEqual(e.calls,[0,1])
    def test_failed_refresh_explicit_stale(self):
        e=Fake([0,.6,.8],fail=(1,2));s=SpatialScheduler(e);s.process_frame(0)
        for i in (1,2):
            r=s.process_frame(i)['summary'];self.assertEqual(r['status'],'UPDATE_FAILED_USING_STALE_GEOMETRY');self.assertEqual(r['source_geometry_frame'],0)
    def test_gap_resets(self):
        e=Fake([0,.2],fail=(1,));e.records[1]['header_time_ns']=1000000000;s=SpatialScheduler(e);s.process_frame(0)
        r=s.process_frame(1);self.assertIsNone(r['prediction']);self.assertFalse(r['summary']['valid_geometry_state'])
    def test_missing_initial_state_not_reuse(self):
        e=Fake([0,.1],fail=(0,1));s=SpatialScheduler(e)
        self.assertEqual(s.process_frame(0)['summary']['status'],'NO_GEOMETRY_STATE');s.process_frame(1);self.assertEqual(e.calls,[0,1])
    def test_past_direction(self):
        B,reason,info=direction(records([0,.28,.57,.8]),2)
        self.assertEqual(info['mode'],'CAUSAL_ACCUMULATED_POSES');np.testing.assert_array_equal(B[:,0],[1,0,0])
    def test_pose_direction_gap(self):
        r=records([0,.28,.57,.8]);r[1]['pose_status']='bad'
        B,_,_=direction(r,2);self.assertIsNone(B)
    def test_covariance_roundtrip(self):
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);p=dict(C=np.ones((2,3)),B=np.repeat(np.eye(3)[None],2,axis=0),
            center_cov=np.repeat(np.diag([1.,2.,3.])[None],2,axis=0))
        restored=transform_prediction(transform_prediction(p,R,np.array([1,2,3]),True),R,np.array([1,2,3]),False)
        for k in p:np.testing.assert_array_equal(p[k],restored[k])
    def test_consumer_snapshot_without_solve(self):
        e=Fake([0,.2]);s=SpatialScheduler(e);s.process_frame(0)
        snapshot=s.geometry_for_pose(e.records[1],1)
        self.assertEqual(e.calls,[0]);self.assertEqual(snapshot['source_frame'],0)
        self.assertEqual(snapshot['geometry_age_frames'],1)
        self.assertIsNone(s.geometry_for_pose(e.records[1],1,minimum_source_frame=1))


if __name__=='__main__':unittest.main()
