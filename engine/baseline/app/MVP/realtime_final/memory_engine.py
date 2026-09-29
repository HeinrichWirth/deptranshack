"""Fresh C4 solves sharing only immutable raw frames and their exact spatial indexes."""
from .native_api import load
from MVP.c4_v2.native_engine import NativeEngine, infer_native
from MVP.performance_final.engine import clone
from MVP.final_pipeline.frame_data import FrameData
from MVP.final_pipeline.pipeline import RawRoi
from roi_source import FIELDS, join, point_keys
from fusion import CausalSource
from types import MethodType, MappingProxyType
import inspect,time,numpy as np

# Exact key lookup instead of scanning the full current frame for every block.
_code=inspect.getsource(infer_native).replace("mask=np.isin(cloud['keys'],np.asarray(b['keys'],dtype=np.uint64));ranges=np.linalg.norm(cloud['xyz'][mask],axis=1)","selected=order[np.searchsorted(cloud['keys'][order],np.asarray(b['keys'],dtype=np.uint64))];ranges=np.linalg.norm(cloud['xyz'][selected],axis=1)")
_scope=dict(infer_native.__globals__);exec(compile(_code,__file__,'exec'),_scope);exact_infer=_scope['infer_native']

class SharedRing:
    def __init__(self,records):
        self.records=records;self.store=load().FrameStore();self.entries={};self.latest=-1;self.arrivals=[]
    def arrive(self,raw):
        k=raw.frame
        if k<=self.latest:raise ValueError('Arrival order must increase')
        begin=time.perf_counter();P=np.asarray(self.records[k]['lidar_pose_in_folder'])
        xyz=(raw.world-P[:3,3])@P[:3,:3]
        meta=dict(sensor_distance_at_source=np.linalg.norm(xyz,axis=1).astype(np.float32),azimuth=np.rint(np.arctan2(xyz[:,1],xyz[:,0])/1e-5).astype(np.int32))
        for a in (xyz,*meta.values()):a.setflags(write=False)
        pre=time.perf_counter();self.store.add_frame(k,raw.world);end=time.perf_counter()
        self.entries[k]=(raw,xyz,MappingProxyType(meta));self.latest=k
        self.entries={i:e for i,e in self.entries.items() if i>=k-11}
        self.arrivals.append(dict(frame=k,points=len(xyz),metadata_ms=(pre-begin)*1000,index_ms=(end-pre)*1000,arrival_ms=(end-begin)*1000))
    def snapshot(self,index):
        history,_=CausalSource('',index,self.records[:index+1]).history(dict(frames=12))
        return Snapshot(self.records,index,{k:self.entries[k] for k in history})
    def prepare_from_files(self,directory,index):
        history,_=CausalSource('',index,self.records[:index+1]).history(dict(frames=12))
        io=0.
        for k in sorted(history):
            if k not in self.entries:
                t=time.perf_counter();raw=FrameData.read(directory/self.records[k]['file'],k);io+=time.perf_counter()-t;self.arrive(raw)
        return io

class Snapshot:
    def __init__(self,records,index,entries):
        self.records=records[:index+1];self.index=index;self.entries=MappingProxyType(entries);self.frames=MappingProxyType({k:v[0] for k,v in entries.items()})
        self.reads=0;self.bytes_read=0;self.read_seconds=0.;self.latest=-1
    def advance(self,index):
        if index!=self.index or self.latest!=-1:raise ValueError('Snapshot is single-use and causal')
        self.latest=index
    def get(self,k,limit,allow_disk=False):
        if limit!=self.index or not self.index-11<=k<=self.index:raise ValueError('FUTURE_OR_EXPIRED_CLOUD_FORBIDDEN')
        return self.entries[k][0]
    def memory_bytes(self):
        return sum(e[0].world.nbytes+e[0].ring.nbytes+e[0].point_index.nbytes+e[1].nbytes+sum(x.nbytes for x in e[2].values())+sum(x.nbytes for x in e[0].measurements.values()) for e in self.entries.values())
    def read(self,k,index,allow_disk=False,rows=None):
        begin=time.perf_counter();raw=self.get(k,index);P=np.asarray(self.records[index]['lidar_pose_in_folder']);K=np.asarray(self.records[k]['lidar_pose_in_folder'])
        ids=np.arange(len(raw.world),dtype=np.uint32) if rows is None else np.asarray(rows,dtype=np.uint32)
        xyz=self.entries[k][1] if k==index and rows is None else (raw.world[ids]-P[:3,3])@P[:3,:3]
        meta=self.entries[k][2];n=len(xyz);age=(int(self.records[index]['header_time_ns'])-int(self.records[k]['header_time_ns']))/1e9
        part=dict(xyz=xyz,source_frame=np.full(n,k,np.int32),source_row=ids,source_point_index=raw.point_index[ids],age_frames=np.full(n,index-k,np.int32),age_seconds=np.full(n,age,np.float32),age_distance=np.full(n,np.linalg.norm(P[:3,3]-K[:3,3]),np.float32),sensor_distance_at_source=meta['sensor_distance_at_source'][ids],sensor_distance_at_T=np.linalg.norm(xyz,axis=1).astype(np.float32),ring=raw.ring[ids],azimuth=meta['azimuth'][ids])
        for a in part.values():a.setflags(write=False)
        return part,dict(frame=k,points=n,read_ms=0.,transform_ms=(time.perf_counter()-begin)*1000)

class ExactRoi(RawRoi):
    def materialize(self):
        parts=[self.current]
        for k,rows in sorted(self.used.items(),reverse=True):
            part,t=self.source.cache.read(k,self.i,False,sorted(rows));parts.append(part);self.timers[k]=t;self.source.read_times.append(t);self.source.accessed.append(k);self.accessed.append(k)
        c=join(parts);c['keys']=point_keys(c)
        c['meta']=dict(source_frames=sorted(set(self.accessed),reverse=True),queried_history_frames=sorted(self.used,reverse=True),current_points=len(self.current['xyz']),historical_roi_points=sum(len(p['xyz']) for p in parts[1:]),points=len(c['xyz']),points_processed=self.processed,io_details=list(self.timers.values()),roi_audit=self.audit,causal_cloud_access=sorted(set(self.accessed)),whole_history_concatenated=False,coordinate_transform='(p_map-t_T) @ R_T',cache_bytes=self.source.cache.memory_bytes())
        return c

class MemoryEngine(NativeEngine):
    def __init__(self,threads=8,tight=True,selective=True,profile=False):
        super().__init__(threads,12,0);self.marcher=load().Marcher(threads,12,0);self.marcher.options(tight);self.marcher.diagnostics(False,profile)
        self.config['preloaded']=True
        self.process_frame=MethodType(clone(self.process_frame.__func__,RawRoi=ExactRoi if selective else RawRoi,infer=lambda *a:exact_infer(self,*a)),self)
    def prepare(self,ring,index):
        self.cache=ring.snapshot(index);self.native_frames=set(self.cache.frames);self.marcher.attach(ring.store,sorted(self.native_frames))
        # The worker sees no metadata beyond the allowed one-pose delay.
        self.records=ring.records[:index+2]
    def close(self):pass
