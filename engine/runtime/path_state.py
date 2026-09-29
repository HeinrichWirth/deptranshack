"""Current-near prefix and causal persistence; no detector or cloud reads."""
import numpy as np


def attach_near(prior, near, transition=4.):
    """Exact current near prefix, C1 offset transition, unchanged far shape."""
    prior=np.asarray(prior,float);near=np.asarray(near,float)
    near=near[:,np.argsort(near[-1,:,0])]
    u=-prior.mean(1)[:,1];end=-near[-1].mean(0)[1]
    if len(prior)<3 or np.any(np.diff(u)<=1e-6) or u[-1]<=end+.1:
        return near.copy(),'NEAR_ONLY_SHORT_OR_NONMONOTONIC_PRIOR'
    order=np.argsort(prior[np.argmin(abs(u-end)),:,0]);prior=prior[:,order]
    def at(v):return np.array([[np.interp(v,u,prior[:,j,k]) for k in range(3)] for j in (0,1)])
    anchor=at(end);span=min(1.,(u[-1]-end)/2)
    old_slope=(at(end+span)-at(end))/span
    new_slope=(near[-1]-near[-3])/(-near[-1].mean(0)[1]+near[-3].mean(0)[1])
    tail=prior[u>end+1e-6].copy();tu=u[u>end+1e-6]
    L=min(transition,u[-1]-end);t=np.clip((tu-end)/L,0,1)
    shift=near[-1]-anchor;change=new_slope-old_slope
    tail+=(1-3*t*t+2*t*t*t)[:,None,None]*shift+L*(t-2*t*t+t*t*t)[:,None,None]*change
    pair=np.concatenate((near,tail));steps=np.linalg.norm(np.diff(pair.mean(1),axis=0),axis=1)
    if not np.isfinite(pair).all() or np.any(steps<=1e-6) or np.any(steps>5):return near.copy(),'NEAR_ONLY_INVALID_JOIN'
    return pair,'CURRENT_NEAR_WITH_BLENDED_TAIL'


def remember(pair, pose, meta, base):
    """Persist the corrected world path, without refreshing the far-source age."""
    P=np.asarray(pose);pair=np.asarray(pair)
    assert base['meta']['source_frame']<=meta['source_frame']
    return dict(base,pair_world=pair@P[:3,:3].T+P[:3,3],original_count=17,
                correction_meta=dict(meta),correction_source_frame=meta['source_frame'])


def valid(state, meta, now, max_base_ms=5000., max_correction_ms=1000.):
    return state is not None and state['meta']['generation']==meta['generation'] and state['correction_meta']['generation']==meta['generation'] and state['correction_source_frame']<=meta['source_frame'] and (now-state['meta']['deadline'])*1000<=max_base_ms and (now-state['correction_meta']['deadline'])*1000<=max_correction_ms
