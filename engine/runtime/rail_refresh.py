"""Current raw-cloud rail anchoring, frozen STEP1, no CR/labels/future clouds."""
import json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'baseline/app'),str(ROOT/'baseline/app/MVP/stages/05_2_contact_marching_long_range/src')]
import long_common
from rail2d_head_v2 import detect
from frozen_classification_common import surface_support
CFG=json.loads((ROOT/'baseline/app/MVP/stages/01_rail2d/config/detector_v2.json').read_text())

def refresh(points,prior=None):
    started=time.perf_counter();timing={};contact_input={}
    def fail(reason,**kw):return dict(status=reason,timing_ms=dict(timing,total=(time.perf_counter()-started)*1000),**contact_input,**kw)
    xyz=np.asarray(points)[:,:3]
    # Current sensor forward only: a wrong/curved far prior must not steer STEP1.
    f=np.array([0.,-1.,0.])
    right=np.array([-f[1],f[0],0.]);basis=np.column_stack((f,right,[0.,0.,1.]))
    valid=np.isfinite(xyz).all(1)&np.any(xyz!=0,axis=1);u=xyz@f
    ids=np.flatnonzero(valid&(u>=0)&(u<=8));uvw=xyz[ids]@basis
    timing['roi']=(time.perf_counter()-started)*1000;before=time.perf_counter()
    found=detect(uvw[:,1:],CFG,details=False);timing['step1']=(time.perf_counter()-before)*1000
    if found['status']!='ok':return fail('STEP1_REJECTED',detector_reason=found['reason'])
    contact_input=dict(contact_uvw=uvw,head_pair=np.asarray(found['pair']))
    before=time.perf_counter();pair=np.asarray(found['pair']);mask=surface_support(uvw[:,1:],found,CFG)
    selected=uvw[mask];selected_ids=ids[mask];which=np.argmin(np.linalg.norm(selected[:,None,1:]-pair[None],axis=2),axis=1)
    def sections(min_points,min_top):
        obs=[];used=[];counts=[]
        for k in range(8):
            heads=[];members=[];support=[]
            for side in (0,1):
                m=(selected[:,0]>=k)&(selected[:,0]<k+1)&(which==side);q=selected[m]
                if len(q)<min_points:break
                top=q[q[:,2]>=np.quantile(q[:,2],.7)]
                if len(top)<min_top:break
                heads.append(np.median(top,axis=0));members.extend(selected_ids[m].tolist());support.append(len(q))
            if len(heads)==2:obs.append(heads);used.extend(members);counts.append(support)
        return obs,used,counts
    obs,used,counts=sections(8,3);sparse_completion=False
    if len(obs)<4:
        # The 2D head's fixed +/-2 cm height band clips a sloping 3D rail.
        # Three dense paired sections initialize a 3D band with the SAME width
        # and height. Recollect raw support along that band, not around one Z.
        if len(obs)<3:return fail('INSUFFICIENT_PAIRED_SECTIONS',sections=len(obs))
        initial=np.asarray(obs);seed_u=initial[:,:,0].mean(1)
        seed_X=np.column_stack((np.ones(len(seed_u)),seed_u-seed_u.mean()))
        all_X=np.column_stack((np.ones(len(uvw)),uvw[:,0]-seed_u.mean()))
        bands=[]
        for side in (0,1):
            coefficient=np.linalg.lstsq(seed_X,initial[:,side,1:],rcond=None)[0]
            if np.max(abs(coefficient[1]))>.03:return fail('UNSTABLE_SEED_SLOPE')
            delta=uvw[:,1:]-all_X@coefficient
            bands.append((abs(delta[:,0])<=CFG['surface_half_width_m'])&(abs(delta[:,1])<=CFG['surface_half_height_m']))
        mask=bands[0]|bands[1]
        selected=uvw[mask];selected_ids=ids[mask];which=np.where(bands[0][mask],0,1)
        obs,used,counts=sections(8,3);sparse_completion=True
    if len(obs)<4:return fail('INSUFFICIENT_PAIRED_SECTIONS',sections=len(obs))
    obs=np.asarray(obs);us=obs[:,:,0].mean(1)
    if np.ptp(us)<2.5:return fail('INSUFFICIENT_SPAN',sections=len(obs))
    # Robust shared slope and per-rail intercepts; two observed rails are mandatory.
    X=np.column_stack((np.ones(len(us)),us-us.mean()));fits=[];residuals=[]
    for side in (0,1):
        values=obs[:,side,1:];coef=np.linalg.lstsq(X,values,rcond=None)[0]
        for _ in range(3):
            res=np.linalg.norm(values-X@coef,axis=1);weights=np.minimum(1,.01/np.maximum(res,1e-9));coef=np.linalg.lstsq(X*weights[:,None],values*weights[:,None],rcond=None)[0]
        fits.append(coef);residuals.extend(np.linalg.norm(values-X@coef,axis=1))
    fits=np.asarray(fits);residual=float(np.quantile(residuals,.9));stability=0.
    gauge=float(np.median(obs[:,1,1]-obs[:,0,1]))
    if residual>.015 or not 1.48<=gauge<=1.70:return fail('INCONSISTENT_HEADS',residual_m=residual,gauge_m=gauge)
    if sparse_completion:
        # Held-out heads must agree with the other sections; dropping any one
        # section must not change the extrapolated 0..8 m heads by > 3 cm.
        endpoints=np.column_stack((np.ones(2),np.array([0.,8.])-us.mean()))
        for side in (0,1):
            values=obs[:,side,1:]
            for omitted in range(len(obs)):
                keep=np.arange(len(obs))!=omitted
                coef=np.linalg.lstsq(X[keep],values[keep],rcond=None)[0]
                held=float(np.linalg.norm(X[omitted]@coef-values[omitted]))
                change=float(np.max(np.linalg.norm(endpoints@(coef-fits[side]),axis=1)))
                stability=max(stability,change)
                if change>.03:return fail('UNSTABLE_SPARSE_HEADS',held_out_residual_m=held,endpoint_change_m=change)
    grid=np.linspace(0,8,17);line=np.column_stack((np.ones(len(grid)),grid-us.mean()))
    pairuvw=np.stack([np.column_stack((grid,line@fits[side])) for side in (0,1)],axis=1)
    near=pairuvw@basis.T
    correction=None;corrected=None;prior_status='NO_PRIOR'
    def accepted():
        timing['refit']=(time.perf_counter()-before)*1000;timing['total']=(time.perf_counter()-started)*1000
        return dict(status='ACCEPTED',near_pair=near,corrected_pair=corrected,observations=obs@basis.T,
            support_indices=np.unique(used),sections=len(obs),observed_u_range=[float(us.min()),float(us.max())],
            residual_m=residual,gauge_m=gauge,confidence=float(found.get('confidence',0)),correction=correction,
            prior_status=prior_status,timing_ms=timing,sparse_completion=sparse_completion,endpoint_stability_m=stability,
            section_support=counts,**contact_input,uses_contact=False,uses_future=False,uses_labels=False)
    if prior is not None:
        old=np.asarray(prior)@basis;ou=old.mean(1)[:,0]
        if np.any(np.diff(ou)<=0):
            prior_status='PRIOR_NOT_FORWARD';return accepted()
        # Preserve prior side ordering, independent of contact side.
        order=np.argsort(old[np.argmin(abs(ou-4)),:,1]);ordered=old[:,order,:]
        target=np.stack([np.column_stack([np.interp(us,ou,ordered[:,side,j]) for j in (1,2)]) for side in (0,1)],axis=1)
        delta=obs[:,:,1:]-target
        coefs=np.array([np.linalg.lstsq(X,delta[:,side],rcond=None)[0] for side in (0,1)])
        max_shift=float(np.max(np.linalg.norm(delta,axis=2)))
        if max_shift>.30 or np.max(abs(coefs[:,1,:]))>.03:
            prior_status='CORRECTION_TOO_LARGE';return accepted()
        # Do not extrapolate a local slope indefinitely. Outside supported span,
        # hold endpoint corrections and preserve old curve shape as an estimate.
        q=np.column_stack((np.ones(len(ou)),np.clip(ou,us.min(),us.max())-us.mean()))
        corrected=old.copy()
        for side in (0,1):corrected[:,order[side],1:]+=q@coefs[side]
        corrected=corrected@basis.T;correction=dict(max_shift_m=max_shift,coefficients=coefs.tolist());prior_status='CORRECTED'
    return accepted()
