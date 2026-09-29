"""Frozen C4 numeric geometry contract, no raw rail/GT access.

Same geometry construction as old seed_input.build_input. Unused profile
fitting is omitted because the approved final candidate never consumes it.
"""
import numpy as np
from track_geometry import curve_model

def count_rounded_unique(points):
    rounded=np.round(points,3)
    if not len(rounded):return 0
    keys=np.rint(rounded*1000).astype(np.int64)
    if keys.min() < -1048576 or keys.max() >=1048576:
        return len(np.unique(rounded,axis=0))
    keys+=1048576
    packed=(keys[:,0]<<42)|(keys[:,1]<<21)|keys[:,2]
    return len(np.unique(packed))


def build_geometry(prediction, cloud, pose):
    p = prediction
    initial = np.asarray(p['seed']['basis'])
    anchors = np.asarray(p.get('curve', []))
    # Exact frozen adapter fallback: the accepted C4 bootstrap already covers
    # 0..8 m. This represents that seed window, never a far extrapolation.
    if len(anchors) < 2:
        anchors=np.array([p['seed']['anchor'],np.asarray(p['seed']['anchor'])+8*initial[:,0]])
    model = curve_model(anchors, initial)
    if model is None:
        return None, None
    points = cloud['xyz'][p['indices']]
    states = p['point_state']
    stepanchors = np.array([p['seed']['anchor']]+[s['anchor_3d'] for s in p['steps'] if s.get('anchor_3d') is not None])
    stepbases = np.array([initial]+[s['basis'] for s in p['steps'] if s.get('anchor_3d') is not None])
    counts, crstate, legacy, position, tangent = [], [], [], [], []
    C, B, s = model['C'], model['B'], model['s']
    for k, (center, basis) in enumerate(zip(C, B)):
        rel = (points-center) @ basis
        inside = (abs(rel[:,0])<=2) & (np.linalg.norm(rel[:,1:],axis=1)<.30)
        take = inside & (states==2)
        n = count_rounded_unique(points[take])
        counts.append(n); crstate.append(2 if n>=2 else (1 if inside.any() else 3))
        legacy.append(stepbases[np.argmin(np.linalg.norm(stepanchors-center,axis=1))])
        residual = p['template_residual'][take]; finite = residual[np.isfinite(residual)]
        res = float(np.median(finite)) if len(finite) else .02
        position.append(max(.01,res)+(.015 if n<5 else .005))
        low,high=max(0,k-2),min(len(C)-1,k+2)
        angle=np.arccos(np.clip(B[low,:,0]@B[high,:,0],-1,1))
        tangent.append(max(np.deg2rad(.1),float(angle)/4))
    return model, dict(s=s,C=C,B=B,legacy_B=np.array(legacy),cr_state=np.array(crstate,dtype=np.uint8),
        support=np.array(counts),position_sigma=np.array(position),tangent_sigma=np.array(tangent),
        initial_basis=initial,anchor_knots=model['knots'],anchor_xyz=model['anchors'],
        world_up_proxy=np.asarray(pose)[:3,:3].T@np.array([0.,0.,1.]))
