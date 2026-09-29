"""One active solve + one newest pending request; no backlog of old frames."""
from .engine import Engine
from MVP.final_pipeline.spatial_scheduler import transform_prediction
from bootstrap import pose_record
from fusion import valid_edge
import numpy as np
import threading,time

def remaining(prediction):
    c=prediction['center'];d=np.diff(c,axis=0);den=np.sum(d*d,axis=1)
    f=-np.sum(c[:-1]*d,axis=1)/np.maximum(den,1e-18)
    j=int(np.argmin(np.linalg.norm(c[:-1]+np.clip(f,0,1)[:,None]*d,axis=1)))
    s=float(prediction['s'][j]+f[j]*(prediction['s'][j+1]-prediction['s'][j]))
    return float(prediction['s'][-1]-max(0.,s))

class Policy:
    def __init__(self,records):
        self.records=records;self.current_geometry=None;self.current_geometry_source_frame=None
        self.worker_busy=False;self.pending_latest_frame=None;self.active=None;self.last_success_pose=None
        self.generation=0;self.minimum_frame=0;self.last_arrival=None;self.events=[];self.max_pending=0
    def arrival(self,index,clock):
        if self.last_arrival is not None and (index!=self.last_arrival+1 or not valid_edge(self.records[self.last_arrival],self.records[index])):
            self.generation+=1;self.minimum_frame=index;self.current_geometry=None;self.current_geometry_source_frame=None;self.last_success_pose=None
            if self.pending_latest_frame is not None:
                self.events.append(dict(event='GEOMETRY_REQUEST_INVALIDATED_GAP',frame=self.pending_latest_frame['frame'],clock=clock))
                self.pending_latest_frame=None
        self.last_arrival=index
    def eligible(self,index):
        if index<self.minimum_frame:return False
        p=pose_record(self.records[index])
        if not p['valid'] or not np.isfinite(p['matrix']).all():return False
        return self.last_success_pose is None or np.linalg.norm(p['matrix'][:3,3]-self.last_success_pose[:3,3])>=.5
    def offer(self,index,ready_pose,clock):
        if index<0 or index>ready_pose or (index+1<len(self.records) and index+1>ready_pose):return
        if not self.eligible(index):return
        if self.active and index<=self.active['frame']:return
        if self.pending_latest_frame is not None:
            if index<=self.pending_latest_frame['frame']:return
            self.events.append(dict(event='GEOMETRY_REQUEST_SUPERSEDED',frame=self.pending_latest_frame['frame'],replacement=index,clock=clock))
        self.pending_latest_frame=dict(frame=index,generation=self.generation,requested=clock,ready_pose=ready_pose)
        self.max_pending=max(self.max_pending,1)
    def start(self,clock):
        if self.worker_busy or self.pending_latest_frame is None:return None
        request=self.pending_latest_frame;self.pending_latest_frame=None
        if request['generation']!=self.generation or not self.eligible(request['frame']):
            self.events.append(dict(event='GEOMETRY_REQUEST_NO_LONGER_NEEDED',frame=request['frame'],clock=clock));return None
        request['started']=clock;self.active=request;self.worker_busy=True;return request
    def complete(self,result,clock):
        request=self.active;self.active=None;self.worker_busy=False;i=request['frame'];published=False
        if request['generation']!=self.generation:status='GEOMETRY_RESULT_INVALIDATED_GAP'
        elif result['prediction'] is None:status='UPDATE_FAILED_USING_STALE_GEOMETRY' if self.current_geometry is not None else 'NO_GEOMETRY_STATE'
        else:
            P=np.asarray(self.records[i]['lidar_pose_in_folder'])
            state=dict(prediction=transform_prediction(result['prediction'],P[:3,:3],P[:3,3],True),
                curve=transform_prediction(result['c4_smooth'],P[:3,:3],P[:3,3],True),source_frame=i,pose=P.copy(),
                source_time=int(self.records[i]['header_time_ns'])/1e9,published=clock,generation=self.generation)
            self.current_geometry=state;self.current_geometry_source_frame=i;self.last_success_pose=P.copy()
            status='FRESH_GEOMETRY_PUBLISHED';published=True
        self.events.append(dict(event=status,frame=i,clock=clock,solve_wall_seconds=clock-request['started'],published=published,
            reason=result['summary']['original_reason'],requested=request['requested']))
        return published
    def snapshot(self,index):
        state=self.current_geometry;record=self.records[index];p=pose_record(record)
        if state is None or state['generation']!=self.generation or state['source_frame']>index or not p['valid']:return None
        P=p['matrix'];pred=transform_prediction(state['prediction'],P[:3,:3],P[:3,3],False)
        return dict(prediction=pred,geometry_source_frame=state['source_frame'],geometry_age_seconds=p['time_ns']/1e9-state['source_time'],
            distance_since_geometry_source=float(np.linalg.norm(P[:3,3]-state['pose'][:3,3])),remaining_horizon=remaining(pred))

class GeometryWorker:
    """Actual background worker; caller/consumer receives snapshots every cloud.

    Engine is exclusively owned by its worker thread. A pose generation protects
    late completions across gaps. Original permitted history is read on demand.
    """
    def __init__(self,run_dir,backend='native_batch_spatial',workers=1):
        self.engine=Engine(backend,workers);self.engine.start_run(run_dir);self.policy=Policy(self.engine.records)
        self.cv=threading.Condition();self.stop=False;self.error=None;self.thread=threading.Thread(target=self._work,name='C4-GeometryWorker',daemon=True);self.thread.start()
    def on_frame(self,index):
        now=time.perf_counter()
        with self.cv:
            if self.error:raise RuntimeError(self.error)
            self.policy.arrival(index,now);self.policy.offer(index-1,index,now);snapshot=self.policy.snapshot(index);self.cv.notify()
        return snapshot
    def _work(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.stop or self.policy.pending_latest_frame is not None)
                if self.stop:return
                request=self.policy.start(time.perf_counter())
            if request is None:continue
            try:result=self.engine.process_frame(request['frame'])
            except Exception as e:
                with self.cv:self.error=repr(e);self.stop=True;self.cv.notify_all()
                return
            with self.cv:self.policy.complete(result,time.perf_counter());self.cv.notify_all()
    def close(self):
        with self.cv:self.stop=True;self.cv.notify_all()
        self.thread.join();self.engine.close()
        if self.error:raise RuntimeError(self.error)
