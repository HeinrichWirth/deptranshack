"""Strict two-pose adapter to the unchanged production STEP1 and FINAL STEP2.

Only two sanitized pose records and the current XYZ enter this API. No selector
is given a trajectory, and no classification field is accepted.
"""
import numpy as np
from time import perf_counter
from rail2d_detector import basis
from rail2d_head_v2 import detect as detect_rails

POSE_KEYS={'matrix','time_ns','valid'}

def pose_record(row):
    return dict(matrix=np.asarray(row['lidar_pose_in_folder'],dtype=float),time_ns=int(row['header_time_ns']),
                valid=row.get('pose_status') in ('ok','origin') and not row.get('pose_uses_future',False))

def initial_frame(a,b,minimum_motion=.50):
    if b is None:return None,'missing_T_plus_1'
    if set(a)!=POSE_KEYS or set(b)!=POSE_KEYS:raise ValueError('Only T and T+1 sanitized poses allowed')
    A=np.asarray(a['matrix']); B=np.asarray(b['matrix'])
    if not a['valid'] or not b['valid'] or not np.isfinite(A).all() or not np.isfinite(B).all():return None,'unreliable_pose'
    dt=(b['time_ns']-a['time_ns'])/1e9; d=B[:3,3]-A[:3,3]; ds=np.linalg.norm(d)
    if not 0<dt<=.3 or ds>5 or ds/dt>50:return None,'pose_time_gap'
    if ds<minimum_motion:return None,'T_plus_1_motion_below_50cm'
    try:return basis(A[:3,:3].T@d),''
    except ValueError as e:return None,str(e)

def production_seed(xyz,a,b,rail_cfg,contact):
    start=perf_counter(); B,reason=initial_frame(a,b)
    result=dict(status='START_UNAVAILABLE',reason=reason,seed_ms=0.,basis=B,indices=np.empty(0,dtype=np.int64))
    if B is not None:
        u=xyz@B[:,0]; ids=np.flatnonzero((u>=0)&(u<=8)&np.isfinite(xyz).all(axis=1))
        r=detect_rails(xyz[ids]@B[:,1:],rail_cfg,details=False)
        result.update(step1_status=r['status'],step1_reason=r['reason'])
        if r['status']=='ok':
            # Preserve the production STEP2 full-matrix multiplication order.
            uvw=xyz@B; ids=np.flatnonzero((uvw[:,0]>=0)&(uvw[:,0]<=8)&np.isfinite(uvw).all(axis=1))
            q=contact.detect({'uvw':uvw[ids]},r['pair'])
            result.update(step2_status=q['status'],confidence=q['confidence'],diagnostics=q['diagnostics'])
            if q['status'] in ('LEFT','RIGHT'):
                result.update(status='AVAILABLE',reason='',indices=ids[q['support_indices']],side=-1 if q['status']=='LEFT' else 1,
                              anchor=np.array([0.,q['anchor_v'],q['anchor_w']])@B.T,seed_end=8.)
            else:result['reason']='STEP2_'+q['status']+':'+q['diagnostics']['reason']
        else:result['reason']='STEP1_'+r['reason']
    result['seed_ms']=1000*(perf_counter()-start)
    return result
