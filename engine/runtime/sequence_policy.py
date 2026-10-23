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


def select(primary, fallback, near, *, contact_status, fresh_near, vertical_m=.03, lateral_m=.10, primary_observed_end=None):
    """contact_status comes ONLY from a completed scheduled full pipeline run."""
    # A locally successful but truncated march must not erase a still-valid
    # observed tail. Caller applies the existing age/pose-generation checks.
    # No object detections or desired range enter this decision.
    if fresh_near and primary is not None and fallback is not None and fallback is not primary:
        old=np.asarray(fallback,float);new=np.asarray(primary,float)
        old_check=agreement(old,near,vertical_m,lateral_m)
        new_check=agreement(new,near,vertical_m,lateral_m)
        if old_check['accepted'] and new_check['accepted']:
            old_end=-old[-1,:,1].mean()
            new_end=-new[-1,:,1].mean() if primary_observed_end is None else float(primary_observed_end)
            if old_end>new_end+4.:
                # Compare the measured common prefix too. A conflicting new
                # path is not evidence that the old distant shape still holds.
                grid=np.arange(2.,min(old_end,new_end)+1e-6,.5)
                def project(p):
                    p=p[:,np.argsort(p[np.argmin(abs(-p.mean(1)[:,1]-4)),:,0])]
                    return np.stack([np.column_stack([np.interp(grid,-p[:,side,1],p[:,side,k]) for k in (0,2)]) for side in (0,1)])
                delta=np.quantile(abs(project(old)-project(new)),.9,axis=(0,1))
                if delta[0]<=lateral_m and delta[1]<=.08:
                    pair,join=attach_near(old,near)
                    return dict(pair=pair,scope='HELD_OBSERVED_TAIL_AFTER_SHORT_MARCH',fallback=True,
                                agreement=old_check,retry=True,reanchor=None,join_status=join,
                                prefix_lateral_p90_m=float(delta[0]),prefix_vertical_p90_m=float(delta[1]))
                # Request a fresh complete solve instead of accepting either
                # incompatible far estimate as a confirmed corridor.
                return dict(pair=near.copy(),scope='NEAR_ONLY_CONFLICTING_PATHS',fallback=False,
                            agreement=new_check,retry=True,reanchor=None)
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
