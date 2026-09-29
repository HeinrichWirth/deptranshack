"""Additive spatial scheduling; imports frozen geometry unchanged.

Contract: MVP/performance_final/ORIENTATION_CONTRACT.md.
"""
from . import ROOT
import long_common as lc
import numpy as np
import time
from fusion import valid_edge
from bootstrap import pose_record


def transform_prediction(p,R,t,to_map):
    out={k:np.array(v,copy=True) for k,v in p.items()}
    rot=R if to_map else R.T
    for k in ('C','pair','center'):
        if k in out:out[k]=out[k]@R.T+t if to_map else (out[k]-t)@R
    if 'B' in out:out['B']=np.einsum('ij,njk->nik',rot,out['B'])
    for k in ('rail_cov','center_cov','covariance'):
        if k in out:out[k]=np.einsum('ij,...jk,lk->...il',rot,out[k],rot)
    return out


class SpatialScheduler:
    def __init__(self,engine=None):
        if engine is None:
            from MVP.performance_final.engine import Engine
            engine=Engine()
        self.engine=engine;self.threshold=.5;self.reset()

    def reset(self):
        self.last_valid_geometry=None;self.last_geometry_pose=None;self.last_geometry_station=None
        self.last_update_time=None;self.last_update_frame=None;self.last_frame=None
        self.station=0.;self.rows=[]

    def start_run(self,run_dir):
        self.reset();return self.engine.start_run(run_dir)

    def process_frame(self,index):
        wall=time.perf_counter();cpu=time.process_time();records=self.engine.records
        if self.last_frame is not None and index!=self.last_frame+1:raise ValueError('Scheduler requires a complete consecutive sequence')
        if not 0<=index<len(records):raise IndexError(index)
        record=records[index];pose=pose_record(record);P=pose['matrix'];now=pose['time_ns']/1e9
        valid=pose['valid'] and np.isfinite(P).all()
        if self.last_frame is not None:
            edge=valid_edge(records[self.last_frame],record)
            if not edge:self.last_valid_geometry=None;self.last_geometry_pose=None;self.last_update_time=None;self.last_update_frame=None
            else:self.station+=float(np.linalg.norm(P[:3,3]-np.asarray(records[self.last_frame]['lidar_pose_in_folder'])[:3,3]))
        self.last_frame=index
        distance=None if self.last_geometry_pose is None else float(np.linalg.norm(P[:3,3]-self.last_geometry_pose[:3,3]))
        fresh_result=None;attempt=False;reason='';direction_info={};low_reuse=False
        if not valid:status='NO_GEOMETRY_STATE';reason='INVALID_CURRENT_POSE'
        elif self.last_valid_geometry is not None and distance<self.threshold:
            status='GEOMETRY_REUSED_LOW_MOTION';low_reuse=True
        else:
            attempt=True;fresh_result=self.engine.process_frame(index);direction_info=dict(getattr(self.engine,'direction_info',{}))
            reason=fresh_result['summary']['original_reason']
            if fresh_result['prediction'] is not None:
                self.last_valid_geometry=dict(prediction=transform_prediction(fresh_result['prediction'],P[:3,:3],P[:3,3],True),
                    curve=transform_prediction(fresh_result['c4_smooth'],P[:3,:3],P[:3,3],True),
                    source_pose=P.copy(),source_frame=index,source_time=now,source_station=self.station)
                self.last_geometry_pose=P.copy();self.last_geometry_station=self.station
                self.last_update_time=now;self.last_update_frame=index;status='FRESH_GEOMETRY_UPDATE'
            else:status='UPDATE_FAILED_USING_STALE_GEOMETRY' if self.last_valid_geometry is not None else 'NO_GEOMETRY_STATE'
        # Drop raw cache entries even when no solve occurs. New frames can be
        # read by the obstacle consumer separately; no obstacle algorithm here.
        raw_current=None;raw_interface_started=time.perf_counter()
        if not attempt and self.engine.cache is not None:
            self.engine.cache.advance(index)
            if valid:
                raw=self.engine.cache.get(index,index)
                raw_current=(raw.world-P[:3,3])@P[:3,:3]
        elif fresh_result is not None:
            raw_current=fresh_result['cloud']['xyz'][fresh_result['cloud']['source_frame']==index]
        raw_interface_seconds=time.perf_counter()-raw_interface_started
        transform_start=time.perf_counter();prediction=None;curve=None
        if status=='FRESH_GEOMETRY_UPDATE':prediction=fresh_result['prediction'];curve=fresh_result['c4_smooth']
        elif self.last_valid_geometry is not None and valid:
            prediction=transform_prediction(self.last_valid_geometry['prediction'],P[:3,:3],P[:3,3],False)
            curve=transform_prediction(self.last_valid_geometry['curve'],P[:3,:3],P[:3,3],False)
        transform_seconds=time.perf_counter()-transform_start
        remaining=None
        if prediction is not None:
            # Coverage only: projection on native centerline, no geometry rewrite.
            c=prediction['center'];d=np.diff(c,axis=0);den=np.sum(d*d,axis=1)
            f=-np.sum(c[:-1]*d,axis=1)/np.maximum(den,1e-18)
            clipped=np.clip(f,0,1);j=int(np.argmin(np.linalg.norm(c[:-1]+clipped[:,None]*d,axis=1)))
            station=float(prediction['s'][j]+f[j]*(prediction['s'][j+1]-prediction['s'][j]))
            remaining=float(prediction['s'][-1]-max(0.,station))
        row=dict(run=self.engine.run,frame=index,status=status,geometry_state='FRESH_GEOMETRY_UPDATE' if status=='FRESH_GEOMETRY_UPDATE' else
            'REUSED_GEOMETRY_STATE' if prediction is not None else 'NO_GEOMETRY_STATE',solve_attempted=attempt,
            fresh=status=='FRESH_GEOMETRY_UPDATE',low_motion_reuse=low_reuse,stale=status=='UPDATE_FAILED_USING_STALE_GEOMETRY',
            valid_geometry_state=prediction is not None,geometry_age_frames=None if self.last_update_frame is None else index-self.last_update_frame,
            geometry_age_seconds=None if self.last_update_time is None else now-self.last_update_time,
            distance_since_geometry_update=None if self.last_geometry_pose is None else float(np.linalg.norm(P[:3,3]-self.last_geometry_pose[:3,3])),
            trigger_distance_m=distance,source_geometry_frame=self.last_update_frame,station=self.station,raw_interface_seconds=raw_interface_seconds,
            remaining_horizon_m=remaining,reason=reason,direction=direction_info,geometry_transform_seconds=transform_seconds,
            solve_seconds=0 if not attempt else fresh_result['summary']['timing']['T_TOTAL'],
            wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
        self.rows.append(row)
        return dict(summary=row,prediction=prediction,c4_smooth=curve,fresh_result=fresh_result,
            obstacle_input=dict(current_frame=index,coordinate_frame='current sensor',geometry_source_frame=self.last_update_frame,
                geometry=prediction,current_xyz=raw_current,remaining_horizon_m=remaining,geometry_status=status))

    def geometry_for_pose(self,record,index,minimum_source_frame=0):
        """Cheap snapshot for a separate every-cloud consumer, without a solve.

        Caller passes the latest pose-gap frame as minimum_source_frame. The
        snapshot is committed in one reference assignment after a complete solve;
        a running solve never publishes half-built arrays. This method does not
        implement obstacle detection or an asynchronous task queue.
        """
        state=self.last_valid_geometry;pose=pose_record(record)
        if state is None or not pose['valid'] or not np.isfinite(pose['matrix']).all():return None
        if not minimum_source_frame<=state['source_frame']<=index:return None
        P=pose['matrix'];seconds=pose['time_ns']/1e9-state['source_time']
        if seconds<0:return None
        return dict(prediction=transform_prediction(state['prediction'],P[:3,:3],P[:3,3],False),
            c4_smooth=transform_prediction(state['curve'],P[:3,:3],P[:3,3],False),
            source_frame=state['source_frame'],geometry_age_frames=index-state['source_frame'],geometry_age_seconds=seconds,
            distance_since_geometry_update=float(np.linalg.norm(P[:3,3]-state['source_pose'][:3,3])),
            status='LAST_COMMITTED_GEOMETRY_SNAPSHOT')

    def finish_run(self):return dict(run=self.engine.run,rows=self.rows)
