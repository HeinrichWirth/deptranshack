"""Only complete stage results create a primary path; no standalone STEP2 gate."""
import numpy as np
from path_state import attach_near
from reanchor import reanchor


def agreement(prior, near, vertical_m=.03, lateral_m=.10):
    """Independent current near observations veto an incompatible cached path.

    Compare X/Z at each rail's own forward coordinate, not differently oriented
    cross-section endpoints. This is an integration check, never a GT gate.
    """
    p=np.asarray(prior,float);n=np.asarray(near,float)
    if p.ndim!=3 or len(p)<3 or not np.isfinite(p).all():
        return dict(accepted=False,reason='INVALID_PRIOR')
    p=p[:,np.argsort(p[np.argmin(abs(-p.mean(1)[:,1]-4)),:,0])]
    n=n[:,np.argsort(n[-1,:,0])]
    grid=np.arange(2.,8.01,.5);deltas=[]
    for side in (0,1):
        u=-p[:,side,1];v=-n[:,side,1]
        if np.any(np.diff(u)<=1e-6):return dict(accepted=False,reason='NONMONOTONIC_PRIOR')
        q=grid[(grid>=max(u.min(),v.min()))&(grid<=min(u.max(),v.max()))]
        if len(q)<6:return dict(accepted=False,reason='INSUFFICIENT_OVERLAP')
        deltas.append(np.column_stack([np.interp(q,u,p[:,side,k])-np.interp(q,v,n[:,side,k]) for k in (0,2)]))
    d=abs(np.concatenate(deltas));x,z=np.quantile(d,.9,axis=0)
    accepted=bool(x<=lateral_m and z<=vertical_m)
    return dict(accepted=accepted,reason='AGREES_WITH_CURRENT_NEAR' if accepted else 'CURRENT_NEAR_CONTRADICTION',
                lateral_p90_m=float(x),vertical_p90_m=float(z),lateral_limit_m=lateral_m,vertical_limit_m=vertical_m)


def select(primary, fallback, near, *, contact_status, fresh_near, vertical_m=.03, lateral_m=.10):
    """contact_status comes ONLY from a completed scheduled full pipeline run."""
    if contact_status in ('NOT_FOUND','AMBIGUOUS') and fresh_near and fallback is not None:
        r=reanchor(fallback,near)
        if r['status']=='REANCHORED':
            return dict(pair=r['pair'],scope='RAIL_FALLBACK_AFTER_FULL_STEP2_FAILURE',fallback=True,
                        agreement=None,retry=False,reanchor={k:v for k,v in r.items() if k not in ('pair','rotation','old_anchor','new_anchor')})
    if primary is None:
        return dict(pair=near.copy(),scope='NEAR_ONLY_WAITING_FULL_CHAIN',fallback=False,agreement=None,retry=True,reanchor=None)
    check=agreement(primary,near,vertical_m,lateral_m)
    if not check['accepted']:
        return dict(pair=near.copy(),scope='NEAR_ONLY_RECOMPUTE_CONTRADICTORY_PATH',fallback=False,agreement=check,retry=True,reanchor=None)
    pair,join=attach_near(primary,near)
    return dict(pair=pair,scope='HELD_COMPLETE_PIPELINE_PATH' if len(pair)>len(near) else 'NEAR_ONLY_SHORT_PRIMARY',
                fallback=False,agreement=check,retry=len(pair)<=len(near),reanchor=None,join_status=join)
