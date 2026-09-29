"""GREEN: retain exact raw arrays and source-only diagnostics for 12 past frames."""
from collections import OrderedDict
import time
import numpy as np
from ..frame_data import FrameData


class HistoryCache:
    def __init__(self,run_dir,records,metadata=True):
        self.run_dir=run_dir;self.records=records;self.frames=OrderedDict();self.source_meta={}
        self.metadata=metadata;self.reads=0;self.bytes_read=0;self.latest=-1;self.read_seconds=0.

    def get(self,k,limit,allow_disk=True):
        if not 0<=k<=limit:raise ValueError('FUTURE_CLOUD_FORBIDDEN')
        if k not in self.frames:
            if not allow_disk:raise ValueError('PRELOADED_CACHE_MISS')
            started=time.perf_counter();p=FrameData.read(self.run_dir/self.records[k]['file'],k)
            self.frames[k]=p;self.reads+=1;self.bytes_read+=p.source_bytes
            self.read_seconds+=time.perf_counter()-started
        return self.frames[k]

    def advance(self,index):
        if index<=self.latest:raise ValueError('Cache must advance in causal order')
        self.latest=index
        for k in list(self.frames):
            if k<index-11:self.frames.pop(k);self.source_meta.pop(k,None)

    def memory_bytes(self):
        return sum(p.world.nbytes+p.ring.nbytes+p.point_index.nbytes+sum(a.nbytes for a in p.measurements.values()) for p in self.frames.values())+sum(sum(a.nbytes for a in m.values()) for m in self.source_meta.values())

    def read(self,k,index,allow_disk=True):
        before=time.perf_counter();p=self.get(k,index,allow_disk);read_ms=(time.perf_counter()-before)*1000
        start=time.perf_counter();row=self.records[k]
        P=np.asarray(self.records[index]['lidar_pose_in_folder']);K=np.asarray(row['lidar_pose_in_folder'])
        xyz=(p.world-P[:3,3])@P[:3,:3]
        if k in self.source_meta:
            meta=self.source_meta[k]
        else:
            source=(p.world-K[:3,3])@K[:3,:3]
            meta=dict(sensor_distance_at_source=np.linalg.norm(source,axis=1).astype(np.float32),
                      azimuth=np.rint(np.arctan2(source[:,1],source[:,0])/1e-5).astype(np.int32))
            if self.metadata:self.source_meta[k]=meta
        n=len(xyz);age=(int(self.records[index]['header_time_ns'])-int(row['header_time_ns']))/1e9
        part=dict(xyz=xyz,source_frame=np.full(n,k,dtype=np.int32),source_row=np.arange(n,dtype=np.uint32),source_point_index=p.point_index,
            age_frames=np.full(n,index-k,dtype=np.int32),age_seconds=np.full(n,age,dtype=np.float32),
            age_distance=np.full(n,np.linalg.norm(P[:3,3]-K[:3,3]),dtype=np.float32),
            sensor_distance_at_source=meta['sensor_distance_at_source'],sensor_distance_at_T=np.linalg.norm(xyz,axis=1).astype(np.float32),
            ring=p.ring,azimuth=meta['azimuth'])
        return part,dict(frame=k,points=n,read_ms=read_ms,transform_ms=(time.perf_counter()-start)*1000)
