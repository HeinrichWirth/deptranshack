"""Causal source capability; raw XYZ are already in each run's registered map."""
from fusion_common import *
from bootstrap import pose_record
from functools import lru_cache

def sanitized(row):
    return {k:row[k] for k in ('file','header_time_ns','lidar_pose_in_folder','pose_status','pose_uses_future') if k in row}

def valid_edge(a,b):
    dt=(int(b['header_time_ns'])-int(a['header_time_ns']))/1e9
    A=np.asarray(a['lidar_pose_in_folder']);B=np.asarray(b['lidar_pose_in_folder']);ds=np.linalg.norm(B[:3,3]-A[:3,3])
    return bool(a.get('pose_status') in ('ok','origin') and b.get('pose_status') in ('ok','origin') and not a.get('pose_uses_future',False) and not b.get('pose_uses_future',False) and np.isfinite(A).all() and np.isfinite(B).all() and 0<dt<=.3 and ds<=5 and ds/dt<=50)

class CausalSource:
    def __init__(self,run,i,records=None):
        self.run=run;self.i=i
        rows=frames(run) if records is None else records
        # JSON container contains an entire recording. Only this prefix and a
        # separate sanitized T+1 pose leave this boundary. No future trajectory.
        self.past=[sanitized(r) for r in rows[:i+1]]
        self.direction_next=pose_record(rows[i+1]) if i+1<len(rows) else None
        self.accessed=[]
    def history(self,config):
        ids=[self.i];distance=config.get('distance');count=config.get('frames',1);reason='FRAME_LIMIT' if distance is None else 'DISTANCE_LIMIT'
        T=np.asarray(self.past[-1]['lidar_pose_in_folder'])[:3,3]
        for k in range(self.i-1,-1,-1):
            if distance is None and len(ids)>=count:break
            if not valid_edge(self.past[k],self.past[k+1]):reason='TIME_OR_POSE_GAP';break
            d=float(np.linalg.norm(np.asarray(self.past[k]['lidar_pose_in_folder'])[:3,3]-T))
            if distance is not None and d>distance+1e-10:break
            ids.append(k)
        else:reason='RUN_START'
        ids=[k for k in ids if self.i-k not in config.get('exclude_ages',[])]
        assert ids[0]==self.i
        return ids,reason
    def read(self,k):
        if not 0<=k<=self.i:raise ValueError('FUTURE_CLOUD_FORBIDDEN')
        self.accessed.append(k);row=self.past[k];path=dataset()/self.run/row['file'];t=time.perf_counter()
        with laspy.open(path) as f:h=f.header
        raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
        p=np.column_stack([np.asarray(raw[a],dtype=np.float64)*h.scales[j]+h.offsets[j] for j,a in enumerate(('X','Y','Z'))])
        ring=np.asarray(raw['ring']).copy() if 'ring' in raw.dtype.names else np.full(len(p),-1,dtype=np.int16)
        point=np.asarray(raw['point_index']).copy() if 'point_index' in raw.dtype.names else np.arange(len(p),dtype=np.uint32)
        del raw;read_ms=(time.perf_counter()-t)*1000;t=time.perf_counter()
        P=np.asarray(self.past[-1]['lidar_pose_in_folder']);K=np.asarray(row['lidar_pose_in_folder'])
        xyz=(p-P[:3,3])@P[:3,:3]
        source=(p-K[:3,3])@K[:3,:3] # Only source-distance/azimuth diagnostics, NOT re-registration.
        n=len(p);age=(int(self.past[-1]['header_time_ns'])-int(row['header_time_ns']))/1e9
        result=dict(xyz=xyz,source_frame=np.full(n,k,dtype=np.int32),source_row=np.arange(n,dtype=np.uint32),source_point_index=point,
            age_frames=np.full(n,self.i-k,dtype=np.int32),age_seconds=np.full(n,age,dtype=np.float32),age_distance=np.full(n,np.linalg.norm(P[:3,3]-K[:3,3]),dtype=np.float32),
            sensor_distance_at_source=np.linalg.norm(source,axis=1).astype(np.float32),sensor_distance_at_T=np.linalg.norm(xyz,axis=1).astype(np.float32),ring=ring,
            azimuth=np.rint(np.arctan2(source[:,1],source[:,0])/1e-5).astype(np.int32),source_sensor_xyz=source)
        return result,dict(frame=k,points=n,read_ms=read_ms,transform_ms=(time.perf_counter()-t)*1000)

def assemble(source,config,seed=None):
    start=time.perf_counter();ids,reason=source.history(config);parts=[];timers=[]
    for k in ids:
        p,t=source.read(k);parts.append(p);timers.append(t)
    alignments=[]
    if config.get('alignment_bound') and seed is not None and seed['status']=='AVAILABLE':
        from tracker import TemplateTracker
        from contact_rail_step2 import ContactRailDetector
        bound=config['alignment_bound'];B=np.asarray(seed['basis']);anchor=np.asarray(seed['anchor'])
        tt=TemplateTracker(ContactRailDetector().template,seed['side'],CFG)
        known=(parts[0]['xyz'][seed['indices']]-anchor)@B;current,_=tt.fit(known[(known[:,0]>=4)&(known[:,0]<=8),1:],np.zeros(2),bound)
        if current:
            a=current[0]['anchor']
            for k,part in zip(ids[1:],parts[1:]):
                q=(part['xyz']-anchor)@B;take=(q[:,0]>=4)&(q[:,0]<=8)
                candidate,_=tt.fit(q[take,1:],a,bound)
                delta=a-candidate[0]['anchor'] if candidate and candidate[0]['support_unique']>=3 else np.zeros(2)
                delta=np.clip(delta,-bound,bound);part['xyz']=part['xyz']+delta@B[:,1:].T
                part['sensor_distance_at_T']=np.linalg.norm(part['xyz'],axis=1).astype(np.float32)
                alignments.append(dict(frame=k,delta_vw=delta,delta_sensor_xyz=delta@B[:,1:].T,uses_only_current_confirmed_seed_overlap=True))
    c={name:np.concatenate([p[name] for p in parts]) for name in parts[0]};raw_n=len(c['xyz']);t=time.perf_counter()
    keep=np.arange(raw_n);representation=config.get('representation','RAW_CONCAT')
    if config.get('self_mask'):
        # Only use an externally documented physical envelope, never fit to labels.
        bounds=np.asarray(config['self_mask']);q=c['source_sensor_xyz'];drop=np.all((q>=bounds[0])&(q<=bounds[1]),axis=1)
        if seed is not None:drop[np.asarray(seed.get('indices',[]),dtype=int)]=False
        keep=keep[~drop]
    if representation=='VOXEL_UNIQUE':
        cell=config.get('voxel',.01);_,u=np.unique(np.floor(c['xyz'][keep]/cell).astype(np.int64),axis=0,return_index=True);chosen=keep[u]
        # Exact current-only seed is retained, even when a seed voxel has >1 row.
        if seed is not None:chosen=np.union1d(chosen,np.asarray(seed.get('indices',[]),dtype=int))
        keep=np.sort(chosen)
    if len(keep)!=raw_n:
        if seed is not None:
            s=copy.deepcopy(seed);s['indices']=np.searchsorted(keep,np.asarray(seed.get('indices',[]),dtype=int));seed=s
        c={name:value[keep] for name,value in c.items()}
    c.pop('source_sensor_xyz')
    c['meta']=dict(config=config,source_frames=ids,history_end_reason=reason,requested_frames=config.get('frames'),actual_frames=len(ids),raw_points=raw_n,points=len(c['xyz']),
        input_bytes=sum(a.nbytes for a in c.values()),max_age_seconds=max(float(p['age_seconds'][0]) if len(p['xyz']) else 0 for p in parts),
        read_ms=sum(x['read_ms'] for x in timers),read_history_ms=sum(x['read_ms'] for x in timers[1:]),transform_ms=sum(x['transform_ms'] for x in timers),fusion_ms=(time.perf_counter()-t)*1000,
        assembly_ms=(time.perf_counter()-start)*1000,source_details=timers,seed_preserved=True,coordinate_transform='(p_map - t_T) @ R_T; no second pose_k application',causal_cloud_access=source.accessed,alignments=alignments)
    return c,seed
