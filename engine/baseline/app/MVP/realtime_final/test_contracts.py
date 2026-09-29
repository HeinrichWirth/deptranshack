"""Causality, immutable leases and late-result safety; no GT involved."""
import unittest
from concurrent.futures import Future
from types import MappingProxyType
import numpy as np
from .memory_engine import SharedRing,MemoryEngine
from .live import Scheduler
from MVP.final_pipeline.frame_data import FrameData

class Contracts(unittest.TestCase):
    def setUp(self):
        self.records=[dict(header_time_ns=i*100000000,pose_status='ok',pose_uses_future=False,lidar_pose_in_folder=np.eye(4).tolist(),file='unused') for i in range(30)]
    def raw(self,i):
        a=np.array([[0.,0.,0.],[1.,2.,3.]]);a.setflags(write=False);r=np.array([1,2]);r.setflags(write=False)
        return FrameData(a,r,r,MappingProxyType({}),'NO_FILE_IO',i,0)
    def test_snapshot_survives_eviction_without_copies(self):
        ring=SharedRing(self.records)
        for i in range(12):ring.arrive(self.raw(i))
        lease=ring.snapshot(11);raw=lease.get(0,11);self.assertIs(raw,ring.entries[0][0]);self.assertFalse(raw.world.flags.writeable)
        for i in range(12,26):ring.arrive(self.raw(i))
        self.assertNotIn(0,ring.entries);self.assertIs(raw,lease.get(0,11));self.assertEqual(len(ring.entries),12)
        with self.assertRaises(ValueError):lease.get(12,11)
        with self.assertRaises(ValueError):lease.get(-1,11)
    def make_scheduler(self,source=9,generation=0):
        s=Scheduler.__new__(Scheduler);s.active={};s.free=[];s.pending=None;s.generation=generation;s.latest_published=source;s.state={'sentinel':'keep'};s.solves=[];s.events=[];s.discarded_stale=0;return s
    def test_old_completion_cannot_overwrite_newer(self):
        s=self.make_scheduler();f=Future();f.set_result(({'prediction':{'should_not_be_read':1}},dict(frame=5)));s.active[f]=dict(worker=0,frame=5,generation=0,requested=0,started=0)
        s.poll();self.assertEqual(s.latest_published,9);self.assertEqual(s.state,{'sentinel':'keep'});self.assertEqual(s.discarded_stale,1)
    def test_gap_invalidates_active_completion(self):
        s=self.make_scheduler(generation=2);f=Future();f.set_result(({'prediction':{'should_not_be_read':1}},dict(frame=12)));s.active[f]=dict(worker=0,frame=12,generation=1,requested=0,started=0)
        s.poll();self.assertEqual(s.solves[-1]['publish_status'],'INVALIDATED_GAP');self.assertEqual(s.latest_published,9)
if __name__=='__main__':unittest.main()
