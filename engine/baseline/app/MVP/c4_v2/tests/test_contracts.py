import unittest
import numpy as np
from MVP.c4_v2.native_api import startup
from MVP.c4_v2.persistent import project_sensor,world,sensor


class Contracts(unittest.TestCase):
    def setUp(self):
        self.native=startup();self.t=np.c_[np.linspace(-.06,.06,31),np.zeros(31)]
        self.p=np.r_[self.t+[.02,-.01],self.t+[.02,-.01]];self.job=(self.p,np.zeros(2),.08,False)

    def test_deterministic_threads(self):
        values=[self.native.Solver(n,12,0).fit_batch([self.job]*3,self.t,3) for n in (1,2,4,8,16)]
        for v in values[1:]:self.assertEqual(v,values[0])

    def test_sparse_abstention(self):
        out=self.native.Solver(1,12,0).fit_batch([(np.zeros((2,2)),np.zeros(2),.08,False)],self.t,3)
        self.assertEqual(out[0][0],[]);self.assertEqual(out[0][1]['candidate_n'],1)

    def test_unique_support_and_bounds(self):
        value=self.native.Solver(1,12,0).fit_batch([self.job],self.t,3)[0]
        self.assertEqual(value[1]['candidate_n'],len(self.t))
        for c in value[0]:self.assertTrue(np.all(abs(np.asarray(c['anchor']))<=.08));self.assertLessEqual(c['support_unique'],31)

    def test_no_scipy_callback(self):
        from unittest.mock import patch
        with patch('scipy.optimize.least_squares',side_effect=RuntimeError('forbidden callback')):
            self.assertTrue(self.native.Solver(4,12,0).fit_batch([self.job],self.t,3)[0][0])

    def test_transform_roundtrip(self):
        P=np.eye(4);a=.3;P[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];P[:3,3]=[100,-20,3]
        x=np.array([[1.,2.,3.],[4,5,6]]);np.testing.assert_allclose(sensor(world(x,P),P),x,atol=1e-12)

    def test_local_station_prevents_other_segment(self):
        p=np.array([[0.,0,0],[20,0,0],[20,1,0],[0,1,0]])
        station=project_sensor(p,np.array([3.,.7,0]),2,2,np.array([1.,0,0]))
        self.assertAlmostEqual(station,3.)

    def test_native_future_rejected(self):
        m=self.native.Marcher(1,12,0);m.add_frame(0,np.zeros((3,3)))
        with self.assertRaisesRegex(RuntimeError,'causal'):
            m.extend_track(np.zeros(3),np.eye(3),np.arange(3,dtype=np.int64),self.t,np.eye(4),np.eye(3),0,[1],1.)

if __name__=='__main__':unittest.main()
