import unittest
import numpy as np
from MVP.performance_final.tests.test_scheduler import Fake,records
from MVP.performance_final2.async_worker import Policy
from MVP.performance_final2.native_api import startup
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial import cKDTree

class Contracts(unittest.TestCase):
    def test_latest_queue_and_snapshot_age(self):
        fake=Fake(np.arange(30)*.6);p=Policy(fake.records)
        p.arrival(0,0);p.arrival(1,.1);p.offer(0,1,.1);self.assertEqual(p.start(.1)['frame'],0)
        for i in range(2,11):p.arrival(i,i*.1);p.offer(i-1,i,i*.1)
        self.assertEqual(p.pending_latest_frame['frame'],9);self.assertEqual(p.max_pending,1)
        p.complete(fake.process_frame(0),1.05);self.assertEqual(p.start(1.05)['frame'],9)
        snap=p.snapshot(10);self.assertEqual(snap['geometry_source_frame'],0);self.assertAlmostEqual(snap['geometry_age_seconds'],1.)
        self.assertAlmostEqual(snap['remaining_horizon'],2.)
    def test_no_publish_across_gap(self):
        fake=Fake([0,.6,1.2]);p=Policy(fake.records);p.arrival(0,0);p.arrival(1,.1);p.offer(0,1,.1);p.start(.1)
        fake.records[2]['header_time_ns']=2000000000;p.arrival(2,2.)
        self.assertFalse(p.complete(fake.process_frame(0),2.1));self.assertIsNone(p.snapshot(2))
    def test_pose_latency(self):
        p=Policy(records([0,.7,1.4]));p.offer(1,1,.1);self.assertIsNone(p.start(.1));p.offer(1,2,.2);self.assertEqual(p.start(.2)['frame'],1)
    def test_low_motion_after_publish_discards_pending(self):
        fake=Fake([0,.1,.2]);p=Policy(fake.records);p.offer(0,1,0);p.start(0);p.offer(1,2,.1)
        p.complete(fake.process_frame(0),.2);self.assertIsNone(p.start(.2));self.assertFalse(p.worker_busy)
    def test_native_fd_bounds_zero_and_ownership(self):
        n=startup();rng=np.random.default_rng(7);points=rng.normal(size=(25,2));template=rng.normal(size=(19,2));tree=cKDTree(template)
        for a in (np.zeros(2),np.array([1.-1e-12,-1.+1e-12]),np.array([-.03,.7])):
            p=n.Problem(points,template,np.full(2,-1.),np.ones(2),False)
            f=p.fun(a);original=f.copy();J=p.jac(a);f[:]=99;p.fun(a+.000001)
            np.testing.assert_array_equal(J,approx_derivative(lambda x:tree.query(points-x)[0]/.015,a,method='2-point',rel_step=.001,bounds=(-np.ones(2),np.ones(2))))
            self.assertTrue(J.flags.f_contiguous);self.assertFalse(np.shares_memory(f,J));self.assertFalse(np.array_equal(original,f))
    def test_native_pool_order(self):
        pool=startup().Pool(4)
        try:self.assertEqual(pool.map(lambda x:x*x,list(range(33))),[x*x for x in range(33)])
        finally:pool.close()
    def test_jacobian_cache_independent_calls(self):
        n=startup();rng=np.random.default_rng(24);points=rng.normal(size=(20,2));template=rng.normal(size=(17,2));tree=cKDTree(template)
        p=n.Problem(points,template,np.full(2,-2.),np.full(2,2.));a=np.array([.2,-.3]);b=np.array([.7,.1]);p.fun(a)
        for x in (b,a,b):
            expected=approx_derivative(lambda y:tree.query(points-y)[0]/.015,x,method='2-point',rel_step=.001,bounds=(-np.ones(2)*2,np.ones(2)*2))
            np.testing.assert_array_equal(p.jac(x),expected)
    def test_grid_superset_exact_refinement(self):
        n=startup();rng=np.random.default_rng(19);points=rng.normal(size=(2000,3));grid=n.Grid(points,.3)
        for c in rng.normal(size=(20,3)):
            ids=grid.query(c,.5);exact=ids[np.asarray(sorted(cKDTree(points[ids]).query_ball_point(c,.5)),dtype=int)]
            np.testing.assert_array_equal(exact,np.asarray(sorted(cKDTree(points).query_ball_point(c,.5)),dtype=int))

if __name__=='__main__':unittest.main()
