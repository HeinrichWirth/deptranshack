"""Persistent map geometry, fresh causal evidence, immutable confirmed prefix.

The STEP6 seed remains the actual cold bootstrap seed in its original sensor
coordinate system. It is never relabelled as a current-frame measurement.
Frozen downstream functions run in that fixed system; publication transforms
their coordinates to the current lidar. Quality drift is measured explicitly.
"""
from dataclasses import dataclass
import copy,time
import numpy as np
from .native_engine import NativeEngine,infer_native
from .solver_backend import tracker_type
from .native_api import startup
from MVP.performance_final2.engine import Engine as ReferenceEngine
from MVP.performance_final.spatial_cache import CachedSpatialRoi
from MVP.final_pipeline.adapters.geometry_input import build_geometry
from MVP.final_pipeline.timing import geometry_call
from track_geometry import curve_model
from fusion import valid_edge
import long_tracker
from MVP.performance_final.engine import clone
from MVP.performance_final2.tracker_backend import factory


def world(x,P):return np.asarray(x)@np.linalg.inv(P[:3,:3])+P[:3,3]
def sensor(x,P):return (np.asarray(x)-P[:3,3])@P[:3,:3]
def stations(x):return np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))]


def project_sensor(points,position,previous_s,displacement,forward):
    s=stations(points);d=np.diff(points,axis=0);length=np.linalg.norm(d,axis=1)
    if not len(d):raise ValueError('STATE_SHORT')
    t=np.clip(np.einsum('ij,ij->i',position-points[:-1],d)/np.maximum(length**2,1e-12),0,1)
    candidates=points[:-1]+t[:,None]*d;station=s[:-1]+t*length
    valid=(station>=previous_s-2)&(station<=previous_s+displacement+8)&(d@forward>0)
    if not valid.any():raise ValueError('STATION_PROJECTION_FAILED')
    score=np.linalg.norm(candidates-position,axis=1);score[~valid]=np.inf;k=int(np.argmin(score))
    if score[k]>5:raise ValueError('STATION_PROJECTION_TOO_FAR')
    return float(station[k])


@dataclass
class C4PersistentState:
    origin_pose: np.ndarray
    origin_frame: int
    last_frame: int
    last_pose: np.ndarray
    last_success_frame: int
    anchors: np.ndarray
    initial_basis: np.ndarray
    side: int
    seed: dict
    c4_seed: dict
    points: np.ndarray
    keys: np.ndarray
    states: np.ndarray
    residual: np.ndarray
    prediction: dict
    smooth: dict
    sensor_station: float=0.
    useful_horizon: float=0.
    last_extension_station: float=0.


class PersistentEngine:
    def __init__(self,threads=4,iterations=12,grid=0,repair=8,overlap=8,control='combined'):
        self.control=control;self.repair=float(repair);self.overlap=float(overlap)
        self.engine=ReferenceEngine('native_batch_spatial') if control=='persistence_only' else NativeEngine(threads,iterations,grid)
        self.validation_solver=startup().Solver(threads,iterations,grid)
        self.native_tt=factory('batch',1) if control=='persistence_only' else tracker_type(self.validation_solver)
        self.state=None;self.records=None;self.last_index=-1;self.stats=[];self.seams=[]
        self.reference_trackers=[]
        self.reference_infer=clone(long_tracker.infer,TemplateTracker=factory('batch',1,self.reference_trackers)) if control=='persistence_only' else None

    def start_run(self,path):
        result=self.engine.start_run(path);self.records=self.engine.records;self.run=self.engine.run;return result

    def cold(self,index,status,reason,started,cpu):
        # A recovery may follow reading this frame for validation. Rewind only
        # the cache bookkeeping, never the source limit or stored observations.
        if self.engine.cache.latest==index:self.engine.cache.latest=index-1
        result=self.engine.process_frame(index)
        if result['prediction'] is not None:
            P=np.asarray(self.records[index]['lidar_pose_in_folder']);p=result['c4'];c=result['cloud'];ix=p['indices']
            anchors=np.asarray(p['curve'])
            if len(anchors)<2:anchors=np.asarray(result['geometry_input']['anchor_xyz'])
            self.state=C4PersistentState(P,index,index,P,index,world(anchors,P),P[:3,:3]@p['seed']['basis'],int(p['seed']['side']),copy.deepcopy(result['seed']),copy.deepcopy(p['seed']),world(c['xyz'][ix],P),c['keys'][ix].copy(),p['point_state'].copy(),p['template_residual'].copy(),copy.deepcopy(result['prediction']),copy.deepcopy(result['c4_smooth']))
            self.state.useful_horizon=float(stations(self.state.anchors)[-1])
        elif status!='COLD_FULL_BOOTSTRAP':self.state=None
        self.last_index=index
        timing=dict(result['summary']['timing']);timing['T_TOTAL']=time.perf_counter()-started;timing['CPU_TOTAL']=time.process_time()-cpu
        result['summary'].update(timing=timing,c4_update_status=status,c4_update_reason=reason,bootstrap_executed=True)
        self.record(result,index,status,reason,started,cpu,0,0,0)
        return result

    def record(self,result,index,status,reason,started,cpu,repair,extension,reused):
        s=self.state;wall=time.perf_counter()-started;process=time.process_time()-cpu
        row=dict(run=self.run,frame=index,status=status,reason=reason,total_ms=wall*1000,cpu_ms=process*1000,average_active_cores=process/max(wall,1e-12),
            repair_m=repair,extension_m=extension,reused_m=reused,recomputed_m=repair+extension,available=result['prediction'] is not None,
            fallback=status in ('FALLBACK_FULL','STATE_RESET_POSE_GAP'),origin_frame=None if s is None else s.origin_frame,
            frames_since_cold=None if s is None else index-s.origin_frame,meters_since_cold=None if s is None else s.sensor_station,
            horizon_m=0 if s is None else max(0,stations(s.anchors)[-1]-s.sensor_station),
            last_success_frame=None if s is None else s.last_success_frame,
            native_stats=self.validation_solver.stats() if self.control=='persistence_only' else self.engine.marcher.stats())
        self.stats.append(row);result['persistent_stats']=row

    def publish(self,index,status,reason,started,cpu,repair=0,extension=0,reused=0):
        s=self.state;P=np.asarray(self.records[index]['lidar_pose_in_folder']);O=s.origin_pose;rotation=O[:3,:3].T@P[:3,:3]
        def transform(p):
            out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in p.items()}
            n=len(p['s']);centers=sensor(world(p['C'],O),P);forward=np.einsum('nij,jk->nik',p['B'].transpose(0,2,1),rotation).transpose(0,2,1)
            keep=(p['s']>=max(0,s.sensor_station-2))&(p['s']<=s.sensor_station+128)
            for k,v in out.items():
                if isinstance(v,np.ndarray) and v.ndim and len(v)==n:out[k]=v[keep]
            out['C']=centers[keep];out['B']=forward[keep]
            for key in ('pair','center'):
                if key in p:out[key]=sensor(world(p[key],O),P)[keep]
            for key in ('rail_cov','center_cov','covariance'):
                if key in p:out[key]=np.einsum('ij,...jk,kl->...il',rotation.T,p[key],rotation)[keep]
            return out
        pred=transform(s.prediction);smooth=transform(s.smooth);near=1 if s.side==1 else 0
        summary=dict(run=self.run,frame=index,status='FINAL_GEOMETRY_AVAILABLE',original_reason=reason,final_geometry_available=True,
            pipeline_component='C4_V2_EXPERIMENTAL',c4_update_status=status,c4_update_reason=reason,bootstrap_executed=False,
            timing=dict(T_TOTAL=time.perf_counter()-started,CPU_TOTAL=time.process_time()-cpu),persistent_origin_frame=s.origin_frame,
            fresh_evidence_history=12,stored_map_evidence_is_current=False)
        result=dict(summary=summary,prediction=pred,c4_smooth=smooth,seed=s.seed,geometry_input=None,c4=None,cloud=None,provenance=None,
            left_rail=pred['pair'][:,near],right_rail=pred['pair'][:,1-near],centerline=pred['pair'].mean(1))
        s.last_frame=index;s.last_pose=P;self.last_index=index;self.engine.last_index=index
        self.record(result,index,status,reason,started,cpu,repair,extension,reused);return result

    def process_frame(self,index):
        if index<=self.last_index:raise ValueError('chronological frames required')
        started=time.perf_counter();cpu=time.process_time();s=self.state
        if s is None:return self.cold(index,'COLD_FULL_BOOTSTRAP','NO_STATE',started,cpu)
        if not all(valid_edge(self.records[k-1],self.records[k]) for k in range(s.last_frame+1,index+1)):
            self.state=None;return self.cold(index,'STATE_RESET_POSE_GAP','INVALID_EDGE',started,cpu)
        P=np.asarray(self.records[index]['lidar_pose_in_folder']);displacement=float(np.linalg.norm(P[:3,3]-s.last_pose[:3,3]))
        model=curve_model(s.anchors,s.initial_basis)
        try:s.sensor_station=project_sensor(s.anchors,P[:3,3],s.sensor_station,displacement,model['B'][int(np.argmin(abs(model['s']-s.sensor_station))),:,0])
        except ValueError as error:return self.cold(index,'FALLBACK_FULL',str(error),started,cpu)
        old_tail=float(model['s'][-1]);remaining=old_tail-s.sensor_station
        if remaining<8:return self.cold(index,'FALLBACK_FULL','OBSERVATION_SHORT_HORIZON',started,cpu)
        self.engine.cache.advance(index)
        source=CachedSpatialRoi(self.run,index,self.records,self.engine.cache);xyz=source.current['xyz'];raw_world=self.engine.cache.get(index,index).world
        tt=self.native_tt(self.engine.contact.template,s.side,dict(min_support=3,support_tolerance=.02))
        # Validate a mapped overlap close to the sensor using CURRENT points.
        station=min(max(s.sensor_station+2,2),old_tail-self.overlap)
        k=int(np.argmin(abs(model['s']-station)));C=sensor(model['C'][k],P);B=P[:3,:3].T@model['B'][k]
        q=(xyz-C)@B;sel=(q[:,0]>=0)&(q[:,0]<=self.overlap)
        fits,_=tt.fit(q[sel,1:],[0,0],.12)
        if not fits or fits[0]['support_unique']<3:
            return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS','NO_FRESH_OVERLAP',started,cpu,reused=remaining)
        shift=float(np.linalg.norm(fits[0]['anchor']))
        if shift>.08:return self.cold(index,'FALLBACK_FULL','HARD_DISAGREEMENT',started,cpu)
        soft=shift>.03;repair=min(12,self.repair*1.5) if soft else self.repair
        # Work is triggered by lost useful horizon or an observed contradiction,
        # not a frame-count reset. No extension when still within one 4m step.
        needs_extension=s.sensor_station-s.last_extension_station>=4
        if not soft and not needs_extension:
            s.last_success_frame=index
            return self.publish(index,'INCREMENTAL_REUSE','PASS',started,cpu,reused=remaining)
        start_station=max(s.sensor_station,old_tail-repair-8)
        k=int(np.argmin(abs(model['s']-start_station)));C=sensor(model['C'][k],P);B=P[:3,:3].T@model['B'][k]
        q=(xyz-C)@B;sel=(q[:,0]>=0)&(q[:,0]<=8);fits,_=tt.fit(q[sel,1:],[0,0],.08)
        if not fits or fits[0]['support_unique']<3:
            return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS','NO_CURRENT_TAIL_GEOMETRY',started,cpu,reused=remaining)
        anchor=np.asarray(fits[0]['anchor']);d=tt.distances(q[:,1:],anchor);ids=np.flatnonzero(sel&(d<=.02)&np.all((q[:,1:]-anchor>=tt.lo)&(q[:,1:]-anchor<=tt.hi),1))
        if len(ids)<3:return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS','SPARSE_CURRENT_TAIL',started,cpu,reused=remaining)
        # Existing mapped frame is the prior; fresh actual support is the seed.
        seed=dict(status='AVAILABLE',reason='',anchor=C,basis=B,indices=ids,side=s.side,confidence=float(np.clip(fits[0]['score']/1.35,0,1)),seed_end=8.)
        if self.control=='persistence_only':
            p,cloud=self.reference_infer(source,seed,self.engine.contact.template,self.engine.c4)
            for tracker in self.reference_trackers:tracker.close()
            self.reference_trackers.clear()
        else:p,cloud=infer_native(self.engine,source,seed,self.engine.contact.template,self.engine.c4)
        # Do not shorten the confirmed map on a failed or merely tentative tail.
        new_world=world(np.asarray(p['curve']),P)
        if len(new_world)<2:return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS',p['reason'],started,cpu,reused=remaining)
        # Projection onto the known prefix is local, with a tangent extrapolation
        # only for station bookkeeping, never for synthetic geometry points.
        base_s=float(model['s'][k]);new_s=base_s+np.einsum('ij,j->i',new_world-model['C'][k],model['B'][k,:,0])
        boundary=max(old_tail-repair,base_s+8);keep_new=new_s>=boundary
        if not keep_new.any() or new_s[keep_new][-1]<old_tail-1:
            return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS','UNCONFIRMED_REPLACEMENT_TAIL',started,cpu,reused=remaining)
        old_s=stations(s.anchors);prefix=s.anchors[old_s<boundary];candidate=np.concatenate((prefix,new_world[keep_new]))
        if len(candidate)<2 or np.any(np.linalg.norm(np.diff(candidate,axis=0),axis=1)<.05):
            return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS','INVALID_JOIN',started,cpu,reused=remaining)
        snapshot=copy.deepcopy(s)
        s.anchors=candidate;new_tail=float(stations(candidate)[-1]);extension=max(0,new_tail-old_tail)
        # Frozen geometric input uses persistent accepted observations; ages are
        # never overwritten or fed into the new C4 candidate fit.
        ix=p['indices'];new_points=world(cloud['xyz'][ix],P);new_keys=cloud['keys'][ix]
        old_pos=np.einsum('ij,j->i',s.points-model['C'][k],model['B'][k,:,0])+base_s
        new_pos=np.einsum('ij,j->i',new_points-model['C'][k],model['B'][k,:,0])+base_s
        keep_old=old_pos<boundary;take_new=new_pos>=boundary
        keys=np.r_[s.keys[keep_old],new_keys[take_new]];_,unique=np.unique(keys,return_index=True);unique.sort()
        s.points=np.concatenate((s.points[keep_old],new_points[take_new]))[unique];s.keys=keys[unique]
        s.states=np.r_[s.states[keep_old],p['point_state'][take_new]][unique];s.residual=np.r_[s.residual[keep_old],p['template_residual'][take_new]][unique]
        old_pred=s.prediction;old_smooth=s.smooth
        O=s.origin_pose;numeric=dict(seed=s.c4_seed,curve=sensor(s.anchors,O),indices=np.arange(len(s.points)),point_state=s.states,template_residual=s.residual,steps=[])
        _,geometry=build_geometry(numeric,dict(xyz=sensor(s.points,O)),O)
        if geometry is None:self.state=snapshot;return self.publish(index,'UPDATE_FAILED_REUSE_PREVIOUS','INVALID_DOWNSTREAM_GEOMETRY',started,cpu,reused=remaining)
        (pred,smooth),times=geometry_call(s.seed,geometry,s.side,(s.keys>>np.uint64(32)).astype(int),index)
        # Publish the old confirmed prefix exactly. Frozen functions recompute
        # only the replacement values that are eligible for publication.
        n=min(len(old_pred['s']),len(pred['s']));fixed=np.flatnonzero((old_pred['s'][:n]<boundary)&(abs(old_pred['s'][:n]-pred['s'][:n])<1e-9))
        for newer,older in ((pred,old_pred),(smooth,old_smooth)):
            for key,v in newer.items():
                if isinstance(v,np.ndarray) and v.ndim and len(v)==len(newer['s']) and key in older and isinstance(older[key],np.ndarray) and older[key].ndim and len(older[key])==len(older['s']):v[fixed]=older[key][fixed]
        seam=int(fixed[-1]+1) if len(fixed) else 0
        if 0<seam<len(pred['C']):
            d=pred['C'][seam]-pred['C'][seam-1];t=pred['B'][seam-1,:,0];lateral=np.linalg.norm(d-d@t*t);angle=np.degrees(np.arccos(np.clip(pred['B'][seam,:,0]@t,-1,1)))
            self.seams.append(dict(run=self.run,frame=index,station=boundary,position_jump_m=float(lateral),tangent_jump_deg=float(angle),prefix_exact=bool(np.array_equal(pred['C'][fixed],old_pred['C'][fixed]))))
            if lateral>.08 or angle>5:
                self.state=snapshot;return self.cold(index,'FALLBACK_FULL','SEAM_DISAGREEMENT',started,cpu)
        s.prediction=pred;s.smooth=smooth;s.last_extension_station=s.sensor_station;s.last_success_frame=index;s.useful_horizon=max(s.useful_horizon,new_tail-s.sensor_station)
        status='INCREMENTAL_REPAIR_EXTEND' if extension>.1 else 'INCREMENTAL_REPAIR'
        return self.publish(index,status,'SOFT_DISAGREEMENT' if soft else 'PASS',started,cpu,repair=max(0,old_tail-boundary),extension=extension,reused=max(0,boundary-s.sensor_station))

    def close(self):self.engine.close()

