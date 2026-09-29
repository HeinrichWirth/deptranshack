"""Build immutable indexes concurrently; commit history strictly in source order."""
import time
from types import MappingProxyType
import numpy as np
from MVP.realtime_final.memory_engine import SharedRing

class ParallelRing(SharedRing):
    def arrive_many(self, raws, pool):
        if not raws:
            return
        keys = [r.frame for r in raws]
        if keys != sorted(set(keys)) or keys[0] <= self.latest or keys[-1]-keys[0] > 11:
            raise ValueError('Invalid causal history batch')
        def build(raw):
            k = raw.frame
            begin = time.perf_counter()
            P = np.asarray(self.records[k]['lidar_pose_in_folder'])
            xyz = (raw.world-P[:3,3])@P[:3,:3]
            meta = dict(sensor_distance_at_source=np.linalg.norm(xyz,axis=1).astype(np.float32),
                        azimuth=np.rint(np.arctan2(xyz[:,1],xyz[:,0])/1e-5).astype(np.int32))
            for a in (xyz,*meta.values()):
                a.setflags(write=False)
            pre = time.perf_counter()
            # Native store builds each index independently and locks insertion.
            # All new keys are within the same 12-frame window, so eviction
            # cannot remove another member of this batch.
            self.store.add_frame(k,raw.world)
            end = time.perf_counter()
            return (raw,xyz,MappingProxyType(meta)),dict(frame=k,points=len(xyz),
                metadata_ms=(pre-begin)*1000,index_ms=(end-pre)*1000,arrival_ms=(end-begin)*1000)
        for k,(entry,stats) in zip(keys,pool.map(build,raws)):
            self.entries[k] = entry
            self.latest = k
            self.arrivals.append(stats)
        self.entries = {k:e for k,e in self.entries.items() if k>=self.latest-11}
