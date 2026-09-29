"""Anchor the old 3D shape to the endpoint of current near rails.

Input prior must already be in the current sensor coordinates with pose provenance.
No points, contact detector, annotations or future scans are read here.
"""
import numpy as np
from time import perf_counter

def basis(forward,across):
    f=forward/np.linalg.norm(forward);r=across-f*np.dot(across,f);r/=np.linalg.norm(r)
    up=np.cross(f,r)
    return np.column_stack((f,r,up))

def reanchor(prior,near,*,pose_available=True):
    start=perf_counter()
    def result(status,**kw):return dict(status=status,ms=(perf_counter()-start)*1000,**kw)
    if not pose_available:return result('NEEDS_POSE_OR_EXPLICIT_MOTION_ESTIMATE')
    old=np.asarray(prior,float);near=np.asarray(near,float)
    if any(p.ndim!=3 or p.shape[1:]!=(2,3) or len(p)<3 or not np.isfinite(p).all() for p in (old,near)):return result('INVALID_PAIR')
    oc=old.mean(1);nc=near.mean(1);u=-oc[:,1];end=-nc[-1,1]
    if np.any(np.diff(u)<=1e-6):return result('NONMONOTONIC_PRIOR')
    if u[0]>end-1 or u[-1]<end+1:return result('INSUFFICIENT_PRIOR_AHEAD')
    def at(v):return np.array([[np.interp(v,u,old[:,side,axis]) for axis in range(3)] for side in range(2)])
    anchor=at(end)
    # Match sides by current sensor X, not by contact-side ordering.
    order=np.argsort(anchor[:,0]);near_order=np.argsort(near[-1,:,0]);old=old[:,order];anchor=anchor[order];near=near[:,near_order]
    old_forward=at(end+1).mean(0)-at(end-1).mean(0)
    new_forward=nc[-1]-nc[-3]
    B_old=basis(old_forward,anchor[1]-anchor[0]);B_new=basis(new_forward,near[-1,1]-near[-1,0]);rotation=B_new@B_old.T
    tail=old[u>end+1e-6]
    corrected_tail=(tail-anchor)@rotation.T+near[-1]
    pair=np.concatenate((near,corrected_tail))
    centers=pair.mean(1);steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    if np.any(steps<1e-6) or np.any(steps>5):return result('DISCONTINUITY')
    # The original local observations are the prefix; only the old tail is moved.
    return result('REANCHORED',pair=pair,near_count=len(near),rotation=rotation,
        rotation_degrees=float(np.rad2deg(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),
        anchor_shift_m=float(np.linalg.norm(nc[-1]-anchor.mean(0))),
        length_m=float(steps.sum()),near_length_m=float(steps[:len(near)-1].sum()),
        old_anchor=anchor,new_anchor=near[-1],prior_side_order=order.tolist(),near_side_order=near_order.tolist(),
        distant_shape_observed=False,uses_future=False)
