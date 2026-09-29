"""Causal spatial association. No expected obstacle list, frame-specific rules or GT."""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CONFIG = dict(voxel_m=.05, cluster_radius_m=.24, min_points=3, min_voxels=2,
              match_radius_m=.35, match_fraction=.20, max_gap_s=3., confirm_frames=2,
              max_support_voxels=12000)

def voxelize(xyz, size):
    cells, inverse = np.unique(np.floor(xyz / size).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centers = np.column_stack([np.bincount(inverse, weights=xyz[:,k])/counts for k in range(3)])
    return centers, inverse

def clusters(xyz, ids, pair, cfg=CONFIG):
    if len(xyz)<cfg['min_points']: return []
    v, inv=voxelize(xyz,cfg['voxel_m'])
    edges=cKDTree(v).query_pairs(cfg['cluster_radius_m'],output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v))).tocsr()
    _, labels=connected_components(graph,directed=False)
    point_labels=labels[inv];out=[]
    centers=pair.mean(1)
    for lab in np.unique(labels):
        take=np.flatnonzero(point_labels==lab);vv=v[labels==lab]
        if len(take)<cfg['min_points'] or len(vv)<cfg['min_voxels']:continue
        p=xyz[take];center=np.median(p,axis=0);k=int(np.argmin(np.linalg.norm(centers-center,axis=1)));k=min(k,len(pair)-2)
        fw=centers[k+1]-centers[k];fw/=np.linalg.norm(fw)
        right=pair[k,1]-pair[k,0];right-=fw*(right@fw);right/=np.linalg.norm(right)
        up=np.cross(fw,right)
        if up[2]<0:up=-up
        q=(p-centers[k])@np.column_stack((right,fw,up))
        lo=q.min(0);hi=q.max(0);width,depth,height=hi-lo
        proj=q[:,[0,2]]
        try:area=float(ConvexHull(proj).volume) if len(p)>=3 else 0.
        except QhullError:area=0.
        out.append(dict(indices=ids[take],points=p,voxels=vv,center=center,
            nearest_range_m=float(np.linalg.norm(p,axis=1).min()),center_range_m=float(np.linalg.norm(center)),
            width_m=float(width),height_m=float(height),depth_m=float(depth),bbox_area_m2=float(width*height),
            hull_area_m2=area,occupied_area_m2=float(len(np.unique(np.floor(proj/.05).astype(int),axis=0))*.0025),
            min_height_m=float(lo[2]),max_height_m=float(hi[2]),lateral_center_m=float(np.median(q[:,0])),
            slice_distance_m=float(-center[1]),count=len(p),voxel_count=len(vv)))
    return out

class Tracker:
    def __init__(self,cfg=None):self.cfg=dict(CONFIG,**(cfg or {}));self.tracks=[];self.next_id=1
    def update(self,source,timestamp,pose,generation,components):
        # Invalid/unaccepted pose cannot create an identity or a second observation.
        if pose is None:return [None]*len(components)
        P=np.asarray(pose);current=[c['voxels']@P[:3,:3].T+P[:3,3] for c in components]
        active=[t for t in self.tracks if t['generation']==generation and 0<=timestamp-t['last_time']<=self.cfg['max_gap_s']]
        choices=[]
        for i,v in enumerate(current):
            tree=cKDTree(v)
            for tr in active:
                old=tr['support'];gap=np.maximum(np.maximum(old.min(0)-v.max(0),v.min(0)-old.max(0)),0)
                if np.linalg.norm(gap)>self.cfg['match_radius_m']:continue
                a=cKDTree(old).query(v,k=1)[0];b=tree.query(old,k=1)[0]
                fraction=max(float(np.mean(a<=self.cfg['match_radius_m'])),float(np.mean(b<=self.cfg['match_radius_m'])))
                if fraction<self.cfg['match_fraction']:continue
                choices.append((float(min(np.median(a),np.median(b)))+(1-fraction)*.1,i,tr['id']))
        # One object may be split into disconnected observed pieces. Pieces can
        # share an existing ID; distinct old IDs are NEVER merged retrospectively.
        selected={}
        for score,i,tid in sorted(choices):
            if i not in selected:selected[i]=tid
        assignments=[];by_id={t['id']:t for t in self.tracks}
        for i,(c,v) in enumerate(zip(components,current)):
            tid=selected.get(i)
            if tid is None:
                tid=self.next_id;self.next_id+=1
                tr=dict(id=tid,generation=generation,first_frame=source,first_time=timestamp,last_frame=source,last_time=timestamp,
                    confirmed_frame=None,observations=[],support=v,frames=set())
                self.tracks.append(tr);by_id[tid]=tr
            tr=by_id[tid]
            tr['frames'].add(source);tr['last_frame']=source;tr['last_time']=timestamp
            if tr['confirmed_frame'] is None and len(tr['frames'])>=self.cfg['confirm_frames']:tr['confirmed_frame']=source
            support,_=voxelize(np.concatenate((tr['support'],v)),self.cfg['voxel_m'])
            if len(support)>self.cfg['max_support_voxels']:
                # Deterministic support cap; exact source indices remain in observations.
                support=support[np.linspace(0,len(support)-1,self.cfg['max_support_voxels'],dtype=int)]
            tr['support']=support
            obs={k:val for k,val in c.items() if k not in ('indices','points','voxels','center')}
            obs.update(source_frame=source,time_s=timestamp,center_local=c['center'].tolist(),
                       source_indices=c['indices'].tolist(),world_center=np.median(v,axis=0).tolist())
            tr['observations'].append(obs);assignments.append(tid)
        return assignments

    def export(self):
        out=[]
        for tr in self.tracks:
            obs=tr['observations'];first=obs[0]
            best=max(obs,key=lambda r:(r['voxel_count'],r['count']))
            confirm=next((r for r in obs if r['source_frame']==tr['confirmed_frame']),None)
            out.append(dict(id=tr['id'],status='CONFIRMED' if confirm else 'SINGLE_FRAME',
                first_frame=tr['first_frame'],confirmed_frame=tr['confirmed_frame'],last_frame=tr['last_frame'],
                distinct_frames=len(tr['frames']),frames=sorted(tr['frames']),first_time_s=tr['first_time'],last_time_s=tr['last_time'],
                first_distance_m=first['nearest_range_m'],confirmation_distance_m=confirm['nearest_range_m'] if confirm else None,
                closest_distance_m=min(o['nearest_range_m'] for o in obs),first_size={k:first[k] for k in ('width_m','height_m','bbox_area_m2')},
                size_frame=best['source_frame'],width_m=best['width_m'],height_m=best['height_m'],depth_m=best['depth_m'],
                bbox_area_m2=best['bbox_area_m2'],hull_area_m2=best['hull_area_m2'],occupied_area_m2=best['occupied_area_m2'],
                max_height_m=max(o['max_height_m'] for o in obs),best_slice_distance_m=best['slice_distance_m'],
                observations=obs))
        return out
