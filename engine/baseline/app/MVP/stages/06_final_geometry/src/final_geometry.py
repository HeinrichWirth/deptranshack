"""HACKATHON_FINAL_CANDIDATE. Numeric API: causal C4 + supplied near seed."""
import json
import time
from pathlib import Path
import numpy as np
from smooth_core import plan_frame,grid,cluster,smooth_spline,curve_frames
from seed_core import seed_rebase
from rail_core import predict,INPUT_KEYS
from geometry_math import rail_covariance,cross_axes

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text(encoding='utf-8'))
ALLOWED=INPUT_KEYS|{'initial_basis','anchor_knots','anchor_xyz'}

def predict_geometry(seed,geometry,side,source_frames,current_frame,_timing=None):
    if set(geometry)!=ALLOWED:raise ValueError('Only frozen numeric C4 geometry is accepted; no labels or raw clouds')
    if not len(source_frames) or np.max(source_frames)>current_frame:raise ValueError('Future or missing source provenance')
    if seed['seed_length']!=8:raise ValueError('Frozen candidate requires the provided 0..8m seed')
    if side not in (-1,1):raise ValueError('Invalid contact-rail side')
    a=geometry;cfg=CONFIG['curve'];initial=a['initial_basis'];F=plan_frame(initial,a['world_up_proxy']);s=grid(float(a['anchor_knots'][-1]));ss,aa,_=cluster(a['anchor_knots'],a['anchor_xyz'],cfg['cluster']);cv=np.repeat(np.eye(3)[None]*.035**2,len(ss),axis=0);C,cov,optim=smooth_spline(ss,aa,cv,s,F,cfg);B=curve_frames(s,C,initial);rebased=seed_rebase(seed,s,C,B,side,initial)
    for j in range(len(s)):
        r=next(v for v in cfg['calibration'] if v['lo']<=np.linalg.norm(C[j])<v['hi']);T=B[j,:,1:];local=T.T@cov[j]@T;floor=np.array([r['sigma_v'],r['sigma_w']])**2;cov[j]+=T@np.diag(np.maximum(floor-np.diag(local),0))@T.T
    # Historical scalar position sigma was formed before anisotropic calibration.
    # It is overwritten below by the same final Jacobian propagation as STEP6.1.
    tangent=np.maximum(np.deg2rad(.1),np.interp(s,a['s'],a['tangent_sigma']));data=dict(s=s,C=C,B=B,legacy_B=B,cr_state=np.array([a['cr_state'][np.argmin(abs(a['s']-x))] for x in s]),support=np.interp(s,a['s'],a['support']),position_sigma=np.sqrt(np.trace(cov,axis1=1,axis2=2)/3),tangent_sigma=tangent,world_up_proxy=a['world_up_proxy'])
    if _timing is not None:_timing['middle']=time.perf_counter()
    pred=predict(rebased,data,[dict(beta=np.nan,sigma=.2,informative=False) for _ in s],side,CONFIG['rails']);scale=CONFIG['rails']['corridor_scale']
    for j in range(len(s)):
        Pc=cov[j]+np.eye(3)*(.004*pred['cr_gap_age'][j])**2;rc,J=rail_covariance(C[j],B[j],pred['state'][j],side,pred['state_cov'][j],Pc,tangent[j]);pred['rail_cov'][j]=rc*scale**2;Jc=(J[0]+J[1])/2;rr=pred['pair'][j].mean(axis=0)-C[j];bb,nn=cross_axes(B[j],pred['state'][j,0]);TT=np.column_stack((np.cross(bb,rr),np.cross(nn,rr)));pred['center_cov'][j]=(Pc+Jc@pred['state_cov'][j]@Jc.T+TT@TT.T*tangent[j]**2)*scale**2;pred['sigma_lateral'][j]=np.sqrt(np.maximum(np.einsum('i,rij,j->r',bb,pred['rail_cov'][j],bb),0));pred['sigma_vertical'][j]=np.sqrt(np.maximum(np.einsum('i,rij,j->r',nn,pred['rail_cov'][j],nn),0))
    return pred,dict(s=s,C=C,B=B,covariance=cov,tangent_sigma=tangent)
