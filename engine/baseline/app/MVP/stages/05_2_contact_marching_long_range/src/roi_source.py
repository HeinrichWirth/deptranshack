"""Causal lazy historical ROI. Never concatenates complete historical clouds."""
from long_common import *
from fusion import CausalSource
from scipy.spatial import cKDTree

FIELDS=('xyz','source_frame','source_row','source_point_index','age_frames','age_seconds','age_distance','sensor_distance_at_source','sensor_distance_at_T','ring','azimuth')

def point_keys(p): return (p['source_frame'].astype(np.uint64)<<np.uint64(32)) | p['source_row'].astype(np.uint64)

def empty():
    return {k:np.empty((0,3) if k=='xyz' else 0,dtype=float if k in ('xyz','age_seconds','age_distance','sensor_distance_at_source','sensor_distance_at_T') else np.int64) for k in FIELDS}

def join(parts): return {k:np.concatenate([p[k] for p in parts]) for k in FIELDS} if parts else empty()

class RoiSource:
    def __init__(self,run,i,records=None):
        self.source=CausalSource(run,i,records); self.i=i;self.run=run
        self.parts={};self.trees={};self.timers={};self.used={};self.audit=[];self.processed=0
        self.current=self.read(i)

    def reset_audit(self):
        self.used={};self.audit=[];self.processed=0;self.accessed=[self.i]

    def read(self,k):
        if not 0<=k<=self.i:raise ValueError('FUTURE_CLOUD_FORBIDDEN')
        if k not in self.parts:
            p,t=self.source.read(k);p.pop('source_sensor_xyz',None)
            self.parts[k]=p;self.timers[k]=t
        if hasattr(self,'accessed') and k not in self.accessed:self.accessed.append(k)
        return self.parts[k]

    def history(self,cfg,count=None):
        mode=cfg.get('history','fixed'); n=count or cfg.get('frames',1)
        candidates,_=self.source.history(dict(frames=cfg.get('max_history',24) if mode in ('keyframe','uniform','coverage') else n))
        if mode not in ('keyframe','uniform'):return candidates
        T=np.array(self.source.past[self.i]['lidar_pose_in_folder'])[:3,3]
        d=np.array([np.linalg.norm(np.array(self.source.past[k]['lidar_pose_in_folder'])[:3,3]-T) for k in candidates])
        targets=np.array([0,.5,1,2,3,4,6,8]) if mode=='keyframe' else np.linspace(0,min(8.,d.max()),n)
        ids=[candidates[int(np.argmin(abs(d-x)))] for x in targets if x<=d.max()+1e-8]
        return sorted(set([self.i]+ids),reverse=True)

    def sphere(self,k,C,B,W):
        p=self.read(k)
        if k not in self.trees:self.trees[k]=cKDTree(p['xyz'])
        # Sphere contains the full 0..W slab and +/-0.6 m transverse ROI.
        center=C+B[:,0]*(W/2)
        ids=np.array(sorted(self.trees[k].query_ball_point(center,np.hypot(W/2,.9))),dtype=np.int64)
        self.processed+=len(ids)
        return p,ids,(p['xyz'][ids]-C)@B

    def query(self,C,B,W,previous_C,previous_B,previous_W,anchor,tt,cfg,frames_count,overlap_anchor=None):
        ids=self.history(cfg,frames_count);parts=[];audit=[];cover=[]
        gate=cfg.get('anchor_gate',.03);margin=max(.12,gate)+.02
        for k in ids:
            p,rows,q=self.sphere(k,C,B,W)
            old=(q[:,0]>=0)&(q[:,0]<=min(4.,W))
            roi=np.all((q[:,1:]-anchor>=tt.lo-margin)&(q[:,1:]-anchor<=tt.hi+margin),axis=1)
            correction=np.zeros(2);reg=None;accepted=True
            if k!=self.i and (cfg.get('reg_gate') or cfg.get('alignment')):
                known=q[old&roi,1:]
                cs,_=tt.fit(known,anchor,max(.03,cfg.get('alignment',0)))
                if cs and cs[0]['support_unique']>=3:
                    a=cs[0]['anchor'];d=tt.distances(known,a);good=d<=.04
                    # Registration score includes both fit residual and shift from confirmed track.
                    reg=float(np.hypot(np.quantile(d[good],.9),np.linalg.norm(a-anchor))) if good.any() else None
                    if cfg.get('alignment'): correction=np.clip(anchor-a,-cfg['alignment'],cfg['alignment'])
                if cfg.get('reg_gate'):accepted=reg is not None and reg<=cfg['reg_gate']
            qfit=q.copy();qfit[:,1:]+=correction
            new=((p['xyz'][rows]-previous_C)@previous_B[:,0])>previous_W+1e-8
            m=(qfit[:,0]>=0)&(qfit[:,0]<=W)&new&np.all((qfit[:,1:]-anchor>=tt.lo-margin)&(qfit[:,1:]-anchor<=tt.hi+margin),axis=1)
            rr=rows[m] if accepted else np.empty(0,dtype=np.int64)
            part={f:p[f][rr] for f in FIELDS};part['uvw']=qfit[m] if accepted else np.empty((0,3));part['keys']=point_keys(part)
            bins=set(map(tuple,np.floor(part['uvw'][:,1:]/.01).astype(int)))
            parts.append(part);cover.append(bins)
            audit.append(dict(frame=k,age=self.i-k,points=len(rr),registration_score=reg,accepted=accepted,correction_vw=correction))
        if cfg.get('history')=='coverage' and len(parts)>1:
            selected=[0];occupied=set(cover[0]);available=set(range(1,len(parts)))
            while available and len(selected)<cfg['frames']:
                j=max(available,key=lambda z:(len(cover[z]-occupied),-z));gain=len(cover[j]-occupied)
                if gain<2:break
                occupied |= cover[j];selected.append(j);available.remove(j)
            for j in available:audit[j]['coverage_selected']=False
            parts=[parts[j] for j in sorted(selected)]
        for part in parts:
            if len(part['source_frame']) and part['source_frame'][0]!=self.i:
                k=int(part['source_frame'][0]);self.used.setdefault(k,set()).update(part['source_row'].tolist())
        self.audit.append(dict(origin=C,window=W,requested_history=frames_count,sources=audit))
        fields=FIELDS+('uvw','keys')
        return {f:np.concatenate([p[f] for p in parts]) for f in fields}

    def materialize(self):
        parts=[self.current]
        for k,rows in sorted(self.used.items(),reverse=True):
            p=self.read(k);ix=np.array(sorted(rows),dtype=int);parts.append({f:p[f][ix] for f in FIELDS})
        c=join(parts);c['keys']=point_keys(c)
        c['meta']=dict(source_frames=sorted(self.accessed,reverse=True),queried_history_frames=sorted(self.used,reverse=True),
          current_points=len(self.current['xyz']),historical_roi_points=sum(len(p['xyz']) for p in parts[1:]),points=len(c['xyz']),
          points_processed=self.processed,io_details=[self.timers[k] for k in sorted(set(self.accessed),reverse=True)],roi_audit=self.audit,
          causal_cloud_access=sorted(set(self.accessed)),whole_history_concatenated=False,coordinate_transform='(p_map-t_T) @ R_T',
          cache_bytes=sum(sum(v.nbytes for v in p.values()) for p in self.parts.values()))
        return c
