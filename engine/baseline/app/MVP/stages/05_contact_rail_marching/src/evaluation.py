"""OFFLINE ONLY. This is the only module which reads classification/future clouds.

No object from this module is passed to the production predictor. Every entry
requires an on-disk completed prediction before any label is read.
"""
from common import *
from geometry import pca,transport
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import least_squares
from contact_rail_step2 import ContactRailDetector
from functools import lru_cache

THRESHOLDS=(.01,.02,.03,.05,.10,.20)
BINS=(0,10,20,30,40,50,60,75,100,125,150)

def require_prediction(group,run,i):
    folder=OUT/group/key(run,i)
    assert (folder/'PREDICTION_COMPLETE.json').exists(),'Evaluation before prediction is forbidden'
    return folder

def load_prediction(folder):
    result=load(folder/'prediction.json')
    with np.load(folder/'points.npz') as z:result.update({k:z[k] for k in z.files})
    return result

def voxel(p,cell):
    if not len(p):return p
    _,ix=np.unique(np.floor(p/cell).astype(np.int64),axis=0,return_index=True)
    return p[np.sort(ix)]

def prepare_run(run,group,starts):
    for i in starts:require_prediction(group,run,i)
    folder=OUT/'future_gt'/'raw005'/run;folder.mkdir(parents=True,exist_ok=True);mm=frames(run); audit=[]
    for i,row in enumerate(mm):
        target=folder/f'{i:06d}.npz'
        if target.exists():continue
        path=dataset()/run/row['file']
        with laspy.open(path) as f:h=f.header
        raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
        field='raw_classification' if h.point_format.id<6 else 'classification';mask=31 if h.point_format.id<6 else 255
        ids=np.flatnonzero((raw[field]&mask)==2)
        p=np.column_stack([raw[a][ids].astype(float)*h.scales[j]+h.offsets[j] for j,a in enumerate(('X','Y','Z'))])
        p=voxel(p,.005)
        np.savez_compressed(target,xyz=p)
        audit.append(dict(run=run,frame=i,source_points=len(raw),class2_rows=len(ids),reference_voxel_points=len(p),bytes=path.stat().st_size))
        del raw
    if audit:csv_write(folder/'source_audit.csv',audit)
    return run,len(mm)

@lru_cache(maxsize=1000)
def class2_map(run,i):
    with np.load(OUT/'future_gt'/'raw005'/run/f'{i:06d}.npz') as z:return z['xyz']

def valid_edge(a,b):
    dt=(int(b['header_time_ns'])-int(a['header_time_ns']))/1e9
    A=np.array(a['lidar_pose_in_folder']);B=np.array(b['lidar_pose_in_folder']);ds=np.linalg.norm(B[:3,3]-A[:3,3])
    return a.get('pose_status') in ('ok','origin') and b.get('pose_status') in ('ok','origin') and not a.get('pose_uses_future',False) and not b.get('pose_uses_future',False) and 0<dt<=.3 and ds<=5 and ds/dt<=50

def linked_component(points,seed):
    if not len(points):return points,dict(components=0,component_points=0)
    cells,inv=np.unique(np.floor(points/.15).astype(np.int64),axis=0,return_inverse=True)
    centers=(cells+.5)*.15;pairs=cKDTree(centers).query_pairs(.37,output_type='ndarray')
    if len(pairs):
        graph=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(cells),len(cells))).tocsr()
        nc,lab=connected_components(graph,directed=False)
    else:nc=len(cells);lab=np.arange(nc)
    d,j=cKDTree(centers).query(seed)
    if d>.50:return np.empty((0,3)),dict(components=nc,seed_gap_m=float(d),component_points=0)
    keep=lab[inv]==lab[j]
    return points[keep],dict(components=nc,seed_gap_m=float(d),component_points=int(keep.sum()),discarded_other_component_points=int((~keep).sum()))

def make_curve(p,poses,initial,side):
    """Evaluation-only progress and robust template anchors; no large-gap interpolation."""
    if len(p)<6 or len(poses)<2:return np.empty((0,3)),np.empty((0,3,3)),np.empty(0)
    seg=np.diff(poses,axis=0);lens=np.linalg.norm(seg,axis=1);s=np.r_[0,np.cumsum(lens)]
    _,j=cKDTree(poses).query(p); j=np.minimum(j,len(seg)-1)
    t=seg[j]/np.maximum(lens[j,None],1e-8);progress=s[j]+np.einsum('ij,ij->i',p-poses[j],t)
    bins=np.floor(progress/.5).astype(int); which=[b for b in np.unique(bins) if b>=0 and (bins==b).sum()>=3]
    med=np.array([np.median(p[bins==b],axis=0) for b in which])
    if len(med)<3:return np.empty((0,3)),np.empty((0,3,3)),np.empty(0)
    ss=np.array(which)*.5+.25;det=ContactRailDetector();tp=det.template*np.array([side,1]);tree=cKDTree(tp)
    B=np.array(initial);cur=[];Bs=[];kept=[]
    for k,center in enumerate(med):
        near=abs(ss-ss[k])<=3
        try:
            tangent,_=pca(med[near],B[:,0],True);BB=transport(B,tangent)
        except ValueError:BB=B
        q=(p[bins==which[k]]-center)@BB[:,1:]
        if len(q)<3:continue
        def resid(a):return tree.query(q-a)[0]/.015
        opt=least_squares(resid,[0.,0.],bounds=(-np.ones(2)*.12,np.ones(2)*.12),loss='cauchy',max_nfev=20,diff_step=1e-3)
        cur.append(center+opt.x@BB[:,1:].T);Bs.append(BB);kept.append(ss[k]);B=BB
    return np.asarray(cur),np.asarray(Bs),np.asarray(kept)

def reference(run,i,group='development'):
    folder=require_prediction(group,run,i);target=OUT/'future_gt'/(key(run,i)+'_reference_v2.npz');meta=target.with_suffix('.json')
    if target.exists():
        with np.load(target) as z:data={k:z[k] for k in z.files}
        return data,load(meta)
    pred=load_prediction(folder);seed=pred['seed'];mm=frames(run);P=np.array(mm[i]['lidar_pose_in_folder']);R=P[:3,:3];origin=P[:3,3]
    end=i;travel=0.;why='RUN_END'
    while end+1<len(mm):
        if not valid_edge(mm[end],mm[end+1]):why='TIME_OR_POSE_GAP';break
        travel+=np.linalg.norm(np.array(mm[end+1]['lidar_pose_in_folder'])[:3,3]-np.array(mm[end]['lidar_pose_in_folder'])[:3,3]);end+=1
        if travel>=150:why='150M_HORIZON';break
    current=(class2_map(run,i)-origin)@R
    pcs=[(class2_map(run,j)-origin)@R for j in range(i+2,end+1)]
    allfuture=voxel(np.concatenate(pcs),.005) if pcs else np.empty((0,3))
    if seed.get('basis') is None or seed.get('anchor') is None:
        data=dict(current=current,future=np.empty((0,3)),allfuture=allfuture,curve=np.empty((0,3)),curve_basis=np.empty((0,3,3)),curve_s=np.empty(0))
        info=dict(reason='START_UNAVAILABLE',future_end=end,future_frames=max(0,end-i-1),horizon_reason=why)
    else:
        B=np.array(seed['basis']);a=np.array(seed['anchor']);forward=allfuture@B[:,0]
        allfuture=allfuture[(forward>=0)&(np.linalg.norm(allfuture,axis=1)<=160)]
        linked,component=linked_component(allfuture,a+B[:,0]*4)
        poses=np.array([m['lidar_pose_in_folder'] for m in mm[i:end+1]])[:,:3,3];poses=(poses-origin)@R
        curve,BB,ss=make_curve(linked,poses,B,seed['side'])
        align=[]
        if len(current) and len(linked):
            d=cKDTree(linked).query(current)[0];common=d<.30;align=d[common]
        data=dict(current=current,future=linked,allfuture=allfuture,curve=curve,curve_basis=BB,curve_s=ss)
        info=dict(reason='',future_end=end,future_frames=max(0,end-i-1),horizon_reason=why,travel_m=travel,**component,
                  alignment=percentiles(align),alignment_common_threshold_m=.30,
                  max_GT_available_range=float(np.linalg.norm(linked,axis=1).max()) if len(linked) else None,
                  curve_interpolation_max_gap_m=.80,voxel_m=.005,component_voxel_m=.15,component_neighbor_m=.37)
    np.savez_compressed(target,**data);save(meta,info);return data,info

def curve_match(points,ref):
    """Nearest segment only within observed consecutive bins, never a GT gap."""
    curve=ref['curve'];ss=ref['curve_s'];N=len(points)
    dist=np.full(N,np.nan);near=np.full((N,3),np.nan);valid=np.zeros(N,dtype=bool)
    if len(curve)<2:return dist,near,valid
    pairs=np.flatnonzero((np.diff(ss)<=.80)&(np.linalg.norm(np.diff(curve,axis=0),axis=1)<=1.0))
    if not len(pairs):return dist,near,valid
    A=curve[pairs];D=curve[pairs+1]-A;den=(D*D).sum(axis=1)
    for ids in np.array_split(np.arange(N),max(1,int(np.ceil(N/1000)))):
        t=np.einsum('ijk,jk->ij',points[ids,None,:]-A,D)/np.maximum(den,1e-9)
        q=A+np.clip(t,0,1)[:,:,None]*D
        d=np.linalg.norm(points[ids,None,:]-q,axis=2);j=d.argmin(axis=1)
        dist[ids]=d[np.arange(len(ids)),j];near[ids]=q[np.arange(len(ids)),j]
        # Endpoint extrapolation > 0.30m along a reference segment is unscored.
        excess=np.maximum(0,np.maximum(-t[np.arange(len(ids)),j],t[np.arange(len(ids)),j]-1))*np.sqrt(den[j])
        valid[ids]=excess<=.30
    dist[~valid]=np.nan;return dist,near,valid

def evaluate_prediction(pred,xyz,ref,info):
    indices=pred['indices'];pts=xyz[indices];ranges=np.linalg.norm(pts,axis=1)
    row=dict(run=pred['run'],start_frame=pred['start_frame'],status=pred['status'],stop_reason=pred['reason'],seed_available=pred['seed']['status']=='AVAILABLE',
             predicted_points=len(pts),march_steps=sum(s['status']=='ACCEPTED' for s in pred['steps']),seed_ms=pred['seed']['seed_ms'],march_ms=pred['total_ms'],
             total_ms=pred['seed']['seed_ms']+pred['total_ms'],max_GT_available_range=info.get('max_GT_available_range'),future_frames=info['future_frames'],
             max_accepted_range=float(ranges.max()) if len(pts) else None)
    if not len(pts):return row,[],[],dict(distance=np.empty(0),evaluable=np.empty(0,dtype=bool))
    _,_,valid=curve_match(pts,ref)
    dist=cKDTree(ref['future']).query(pts)[0] if len(ref['future']) else np.full(len(pts),np.nan)
    dist[~valid]=np.nan
    # T+2 cannot observe the near part that the train has already passed. Validate
    # bootstrap against CURRENT labels, and ONLY the marched points against future.
    seed_mask=pred['point_step']==0
    if len(ref['current'])>=3:
        dist[seed_mask]=cKDTree(ref['current']).query(pts[seed_mask])[0]
        valid[seed_mask]=True
    row['near_reference']='GT_CURRENT_T (seed only); GT_FUTURE_FUSED for marching'
    row['evaluation_eligible']=bool(info['future_frames']>=2 and (info.get('max_GT_available_range') or 0)>=12 and len(ref['curve'])>=3)
    row['evaluated_points']=int(valid.sum());row['point_error_p95']=percentiles(dist).get('p95')
    for tol in THRESHOLDS:row[f'point_matched_{int(tol*100)}cm']=float(np.mean(dist[valid]<=tol)) if valid.any() else None
    continuous={tol:0. for tol in (.02,.05,.10)};alive={tol:True for tol in continuous};sr=[]
    for step in [dict(step_index=0,status='SEED',anchor_3d=pred['seed']['anchor'])]+pred['steps']:
        k=step['step_index'];pm=pred['point_step']==k;sv=valid&pm
        rr=dict(run=pred['run'],start_frame=pred['start_frame'],step_index=k,status=step['status'],failure_reason=step.get('failure_reason',''))
        rr.update({a:step.get(a) for a in ('window_start_s','window_end_s','delta_yaw','delta_pitch','delta_roll','confirmed_overlap_n','candidate_n','new_support_n','anchor_v','anchor_w','template_score','confidence','transverse_spread','range_near','range_far','total_ms','pca_ms','bishop_ms','refinement_ms','projection_ms','template_ms','update_ms')})
        for name in ('plane_origin','basis'):
            if step.get(name) is not None:
                a=np.asarray(step[name]);rr.update({name+'_'+str(j):v for j,v in enumerate(a.ravel())})
        if step.get('plane_origin') is not None and len(ref['curve']):
            location=np.asarray(step['plane_origin']);j=np.argmin(np.linalg.norm(ref['curve']-location,axis=1))
            if np.linalg.norm(ref['curve'][j]-location)<=2:
                B=np.asarray(step['basis']);G=ref['curve_basis'][j]
                rr['tangent_angle_error_deg']=float(np.rad2deg(np.arccos(np.clip(B[:,0]@G[:,0],-1,1))))
                dv=G.T@B[:,0];rr['yaw_angle_error_deg']=float(np.rad2deg(np.arctan2(dv[1],dv[0])))
                rr['pitch_angle_error_deg']=float(np.rad2deg(np.arctan2(dv[2],np.hypot(dv[0],dv[1]))))
                rr['gt_pitch_deg']=float(np.rad2deg(np.arcsin(G[2,0])))
                if 0<j<len(ref['curve'])-1:
                    rr['gt_curvature_per_m']=float(np.arccos(np.clip(ref['curve_basis'][j-1][:,0]@ref['curve_basis'][j+1][:,0],-1,1))/max(.01,ref['curve_s'][j+1]-ref['curve_s'][j-1]))
        if step.get('anchor_3d') is not None:
            anchor=np.array(step['anchor_3d'])[None,:];ad,closest,ok=curve_match(anchor,ref)
            rr['GT_anchor_error']=float(ad[0]) if ok[0] else None
            if ok[0]:
                B=np.array(step.get('basis',pred['seed']['basis']));d=(anchor-closest)@B;rr.update(lateral_error=float(d[0,1]),vertical_error=float(d[0,2]))
                j=np.argmin(np.linalg.norm(ref['curve']-anchor,axis=1));G=ref['curve_basis'][j]
                rr['tangent_angle_error_deg']=float(np.rad2deg(np.arccos(np.clip(B[:,0]@G[:,0],-1,1))))
                rr['gt_pitch_deg']=float(np.rad2deg(np.arcsin(G[2,0])))
        if pm.any():
            # Continuous reach stops at the FIRST bad or unscorable accepted block.
            rr['evaluated_fraction']=float(sv.sum()/pm.sum());rr['point_p95']=percentiles(dist[sv]).get('p95')
            for tol in continuous:
                if alive[tol] and sv.sum()>=3 and sv.sum()/pm.sum()>=.80 and np.mean(dist[sv]<=tol)>=.95:
                    continuous[tol]=float(ranges[pm].max())
                else:alive[tol]=False
        sr.append(rr)
    row.update({f'continuous_reach_{int(t*100)}cm':v if row['evaluation_eligible'] else None for t,v in continuous.items()})
    row['farthest_correct_isolated_range']=float(ranges[valid&(dist<=.05)].max()) if np.any(valid&(dist<=.05)) else None
    row['anchor_error_p95']=percentiles([r['GT_anchor_error'] for r in sr if r.get('GT_anchor_error') is not None and r['step_index']>0]).get('p95')
    terminal=pred['steps'][-1] if pred['steps'] else {}
    if terminal.get('plane_origin') is not None:row['max_attempted_range']=float(np.linalg.norm(np.array(terminal['plane_origin'])+np.array(terminal['basis'])[:,0]*pred['config']['window']))
    # Evaluation diagnoses are separate from the detector's primary STOP reason.
    accepted_bad=[r for r in sr if r['status']=='ACCEPTED' and r.get('point_p95') is not None and r['point_p95']>.20]
    row['wrong_structure_suspected']=bool(accepted_bad)
    row['terminal_evaluation_reason']='WRONG_STRUCTURE' if accepted_bad else 'GT_UNAVAILABLE_AHEAD' if (info.get('max_GT_available_range') or 0)<(row.get('max_accepted_range') or 0)+2 else 'UNRESOLVED'
    bins=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        keep=(ranges>=lo)&(ranges<hi)&valid
        gt=ref['future'];g=(np.linalg.norm(gt,axis=1)>=lo)&(np.linalg.norm(gt,axis=1)<hi) if len(gt) else np.zeros(0,dtype=bool)
        br=dict(run=pred['run'],start_frame=pred['start_frame'],lo=lo,hi=hi,predicted=int(((ranges>=lo)&(ranges<hi)).sum()),evaluated=int(keep.sum()),reference_points=int(g.sum()),point_error_p95=percentiles(dist[keep]).get('p95'))
        for t in THRESHOLDS:br[f'matched_{int(t*100)}cm']=float(np.mean(dist[keep]<=t)) if keep.any() else None
        if g.any() and len(pts):br['gt_cloud_coverage_5cm']=float(np.mean(cKDTree(pts).query(gt[g])[0]<=.05))
        bins.append(br)
    return row,sr,bins,dict(distance=dist,evaluable=valid,range=ranges)
