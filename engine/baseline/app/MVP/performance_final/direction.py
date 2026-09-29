import numpy as np
from bootstrap import initial_frame,pose_record
from fusion import valid_edge
from rail2d_detector import basis


def direction(records,index):
    current=pose_record(records[index])
    next_pose=pose_record(records[index+1]) if index+1<len(records) else None
    B,reason=initial_frame(current,next_pose)
    if B is not None:return B,'',dict(mode='FROZEN_T_TPLUS1',from_frame=index,to_frame=index+1)
    if reason not in ('T_plus_1_motion_below_50cm','missing_T_plus_1'):
        return None,reason,dict(mode='UNAVAILABLE')
    P=current['matrix']
    for j in range(index-1,-1,-1):
        if not valid_edge(records[j],records[j+1]):break
        d=P[:3,3]-np.asarray(records[j]['lidar_pose_in_folder'])[:3,3]
        if np.linalg.norm(d)>=.5:
            try:return basis(P[:3,:3].T@d),'',dict(mode='CAUSAL_ACCUMULATED_POSES',from_frame=j,to_frame=index,baseline_m=float(np.linalg.norm(d)))
            except ValueError as e:return None,str(e),dict(mode='UNAVAILABLE')
    return None,'CAUSAL_POSE_BASELINE_BELOW_50CM',dict(mode='UNAVAILABLE',original_reason=reason)
