"""Observation-only contact-rail marching. Geometry input, never LAS or labels.

The production entry point cannot accept an oracle. Oracle experiments call the
private engine from the separate evaluation module, after production is saved.
"""
import numpy as np
from time import perf_counter
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from geometry import binned,pca,transport,refine,scatter,angles

DEFAULT=dict(method='M2',window=8.,advance=4.,tail=8.,bin_size=.30,
             objective='trace',refine_bound=2.,roll_bound=2.,orientation_bound=5.,
             gate=.10,anchor_mode='free',plane='midpoint',transport='bishop',
             min_support=3,support_tolerance=.020,overlap_tolerance=.030,
             shape_limit=.025,ambiguity_margin=.025,max_steps=80,max_range=150.)

class TemplateTracker:
    def __init__(self,template,side,cfg):
        self.template=np.asarray(template)*np.array([side,1]);self.tree=cKDTree(self.template)
        self.lo=self.template.min(axis=0)-.015;self.hi=self.template.max(axis=0)+.015
        self.cfg=cfg

    def distances(self,p,anchor):return self.tree.query(p-anchor)[0]

    def fit(self,p,center,gate,fixed=False):
        """Small translation-only shape search; all candidates retained for audit."""
        center=np.asarray(center); p=np.asarray(p)
        region=np.all((p-center>=self.lo-gate)&(p-center<=self.hi+gate),axis=1)
        points=np.unique(p[region],axis=0)
        if len(points)<self.cfg['min_support']:return [],dict(candidate_n=len(points))
        # Limit optimization work using deterministic 2 mm cells, not density votes.
        _,ix=np.unique(np.floor(points/.002).astype(np.int32),axis=0,return_index=True); fitp=points[ix]
        candidates=[]
        seeds=[center] if fixed else [center,center+[gate*.7,0],center+[-gate*.7,0],center+[0,gate*.7],center+[0,-gate*.7]]
        for seed in seeds:
            if fixed:anchor=center;success=True
            else:
                def residual(a):return self.distances(fitp,a)/.015
                opt=least_squares(residual,np.clip(seed,center-gate+1e-9,center+gate-1e-9),bounds=(center-gate,center+gate),
                                  loss='cauchy',f_scale=1.,diff_step=1e-3,max_nfev=24)
                anchor=opt.x;success=bool(opt.success)
            d=self.distances(points,anchor); inside=np.all((points-anchor>=self.lo)&(points-anchor<=self.hi),axis=1)
            support=inside&(d<=self.cfg['support_tolerance']); n=int(support.sum())
            if n:
                err=float(np.mean(np.sort(d[inside])[:max(1,int(.7*inside.sum()))]));
                reverse=cKDTree(points).query(self.template[::2]+anchor)[0]
                coverage=float(np.mean(reverse<=.025)); score=float(np.exp(-err/.015)+.25*coverage+.1*(1-np.exp(-n/8)))
            else:err=.5;coverage=0.;score=-1.
            cand=dict(anchor=anchor,score=score,residual=err,support_unique=n,coverage=coverage,success=success)
            if not any(np.linalg.norm(anchor-c['anchor'])<.006 for c in candidates):candidates.append(cand)
        candidates.sort(key=lambda c:-c['score'])
        return candidates,dict(candidate_n=len(points))

def predict(xyz,seed,template,config=None):
    if not isinstance(xyz,np.ndarray) or xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('Only N x 3 current-frame geometry is allowed')
    return _engine(xyz,seed,template,dict(DEFAULT,**(config or {})))

def _engine(xyz,seed,template,cfg,oracle_orientation=None):
    begin=perf_counter(); n=len(xyz); steps=[]
    result=dict(status=seed['status'],reason=seed.get('reason',''),seed=seed,config=cfg,steps=steps)
    if seed['status']!='AVAILABLE':
        result.update(indices=np.empty(0,dtype=np.int64),point_step=np.empty(0,dtype=np.uint16),total_ms=(perf_counter()-begin)*1000)
        return result
    B=np.array(seed['basis']); initial=B.copy(); C=np.array(seed['anchor']); W=float(seed.get('seed_end',8.))
    chosen=np.asarray(seed['indices'],dtype=np.int64); seen=np.zeros(n,dtype=bool);seen[chosen]=True
    idx=[chosen.copy()]; point_step=[np.zeros(len(chosen),dtype=np.uint16)]
    seedlocal=(xyz[chosen]-C)@B; local=[seedlocal]; residuals=[np.full(len(chosen),np.nan)]
    confidences=[np.full(len(chosen),seed['confidence'])]; anchors=[np.zeros((len(chosen),2))]
    arc=[seedlocal[:,0].copy()]; arc_origin=0.; curve=[C.copy()]
    tt=TemplateTracker(template,seed['side'],cfg); finite=np.isfinite(xyz).all(axis=1)
    reason='RESEARCH_RANGE_LIMIT'
    for k in range(1,cfg['max_steps']+1):
        st=perf_counter(); previous_B=B.copy(); previous_C=C.copy(); previous_W=W
        support=xyz[seen]; su=(support-C)@B[:,0]; end=float(su.max())
        recent=support[su>=end-cfg['tail']]; clock=perf_counter()
        rec=dict(step_index=k,previous_basis=B.copy(),previous_origin=C.copy(),window_start_s=arc_origin+cfg['advance'],
                 window_end_s=arc_origin+cfg['advance']+cfg['window'],status='STOP',new_support_n=0,
                 pca_ms=0.,bishop_ms=0.,refinement_ms=0.,projection_ms=0.,template_ms=0.,update_ms=0.)
        steps.append(rec)
        try:
            fitpoints=recent if cfg['method']=='M1' else binned(recent,B[:,0],cfg['bin_size'])
            t,center=pca(fitpoints,B[:,0],robust=cfg['method']!='M1')
            rec['pca_ms']=(perf_counter()-clock)*1000;clock=perf_counter()
            if cfg['method']=='M0':newB=initial.copy()
            elif cfg['transport']=='reset_up':
                up=np.array([0.,0.,1.]); up-=np.dot(up,t)*t;up/=np.linalg.norm(up);newB=np.column_stack((t,np.cross(up,t),up))
            else:newB=transport(B,t)
            rec['bishop_ms']=(perf_counter()-clock)*1000
            # The origin is obtained from confirmed points and the previous frame.
            advance=cfg['advance']
            if cfg['plane']=='start':advance=max(.25,previous_W-cfg['tail'])
            elif cfg['plane']=='end':advance=previous_W*.85
            Cnew=center+B[:,0]*(advance-float((center-C)@B[:,0]))
            overlap_ids=np.flatnonzero(seen&(((xyz-Cnew)@newB[:,0])>=0)&(((xyz-Cnew)@newB[:,0])<=cfg['window']))
            if len(np.unique(xyz[overlap_ids],axis=0))<cfg['min_support']:raise ValueError('LOCAL_FRAME_DEGENERATE')
            overlap_fit=binned(xyz[overlap_ids],newB[:,0],cfg['bin_size'])
            clock=perf_counter();optinfo=None
            if cfg['method'] in ('M3','M4'):
                newB,optinfo=refine(overlap_fit,newB,cfg['refine_bound'],cfg['objective'])
                if not optinfo['success']:raise ValueError('ORIENTATION_OPT_FAILED')
            if oracle_orientation is not None:newB=oracle_orientation(Cnew,newB,cfg['tail'])
            rec['refinement_ms']=(perf_counter()-clock)*1000
            if cfg['method']=='M4':
                # Roll diagnostic uses only already confirmed overlap.
                choices=[]
                for deg in np.linspace(-cfg['roll_bound'],cfg['roll_bound'],7):
                    RB=newB@Rotation.from_rotvec([np.deg2rad(deg),0.,0.]).as_matrix()
                    op=(xyz[overlap_ids]-Cnew)@RB
                    cs,_=tt.fit(op[:,1:],np.zeros(2),.06)
                    if cs:choices.append((cs[0]['score'],float(deg),RB))
                if choices:newB=max(choices,key=lambda x:x[0])[2]
                rec['roll_candidates']=[list(x[:2]) for x in choices]
            rec.update(plane_origin=Cnew,basis=newB,orientation_optimization=optinfo,**angles(B,newB))
            rec['transverse_spread']=scatter(overlap_fit,newB,cfg['objective'])
            if np.linalg.norm([rec['delta_yaw'],rec['delta_pitch']])>cfg['orientation_bound']:
                reason='ORIENTATION_JUMP';break
            clock=perf_counter();uvw=(xyz-Cnew)@newB
            slab=finite&(uvw[:,0]>=0)&(uvw[:,0]<=cfg['window'])
            overlap_ids=np.flatnonzero(slab&seen)
            # Unknown means beyond the previous observed slab, not an old rejected point.
            unknown=slab&~seen&(((xyz-previous_C)@previous_B[:,0])>previous_W+1e-8)
            new_ids=np.flatnonzero(unknown)
            rec.update(confirmed_overlap_n=len(overlap_ids),unknown_points_n=len(new_ids),overlap_indices=overlap_ids,
                       projection_ms=(perf_counter()-clock)*1000)
            clock=perf_counter(); overlap=uvw[overlap_ids,1:]
            oc,od=tt.fit(overlap,np.zeros(2),.12)
            if not oc or oc[0]['support_unique']<cfg['min_support']:reason='OVERLAP_INCONSISTENT';break
            oa=oc[0]['anchor']; overlap_error=tt.distances(overlap,oa)
            rec.update(overlap_anchor=oa,overlap_residual_p90=float(np.quantile(overlap_error,.90)))
            if rec['overlap_residual_p90']>cfg['overlap_tolerance']:reason='OVERLAP_INCONSISTENT';break
            nc,nd=tt.fit(uvw[new_ids,1:],oa,cfg['gate'],fixed=cfg['anchor_mode']=='fixed')
            rec.update(candidate_n=nd['candidate_n'],template_candidates=nc)
            if not len(new_ids):reason='OUT_OF_CURRENT_CLOUD';break
            if not nc:reason='NO_NEW_POINTS' if not nd['candidate_n'] else 'TOO_FEW_SUPPORT';break
            best=nc[0];a=best['anchor'];rec.update(anchor_v=float(a[0]),anchor_w=float(a[1]),template_score=best['score'],template_residual=best['residual'])
            if best['support_unique']<cfg['min_support']:reason='TOO_FEW_SUPPORT';break
            if best['residual']>cfg['shape_limit']:reason='SHAPE_MISMATCH';break
            comp=next((c for c in nc[1:] if np.linalg.norm(c['anchor']-a)>.04 and c['support_unique']>=cfg['min_support']),None)
            rec['competing_candidate']=comp
            if comp is not None and best['score']-comp['score']<cfg['ambiguity_margin']:reason='AMBIGUOUS_CANDIDATE';break
            # A new anchor may explain the curve's tiny drift but must still explain old overlap.
            if np.quantile(tt.distances(overlap,a),.90)>cfg['overlap_tolerance']:reason='OVERLAP_INCONSISTENT';break
            d=tt.distances(uvw[new_ids,1:],a);inside=np.all((uvw[new_ids,1:]-a>=tt.lo)&(uvw[new_ids,1:]-a<=tt.hi),axis=1)
            take=inside&(d<=cfg['support_tolerance']); new_ids=new_ids[take];d=d[take]
            if not len(new_ids):reason='NO_NEW_POINTS';break
            confidence=float((1-np.exp(-best['support_unique']/8))*np.exp(-best['residual']/.015)*min(1.,(best['score']-comp['score'])/.15 if comp else 1.))
            rec.update(template_ms=(perf_counter()-clock)*1000,confidence=confidence,new_support_n=len(new_ids),support_indices=new_ids,
                       range_near=float(np.linalg.norm(xyz[new_ids],axis=1).min()),range_far=float(np.linalg.norm(xyz[new_ids],axis=1).max()))
            clock=perf_counter(); arc_origin+=float(np.linalg.norm(Cnew-previous_C)); seen[new_ids]=True
            idx.append(new_ids);point_step.append(np.full(len(new_ids),k,dtype=np.uint16));local.append(uvw[new_ids]);residuals.append(d)
            confidences.append(np.full(len(new_ids),confidence));anchors.append(np.repeat(a[None,:],len(new_ids),axis=0));arc.append(arc_origin+uvw[new_ids,0])
            # Each anchor is located at the center of actual NEW longitudinal support.
            anchor3=Cnew+np.array([np.median(uvw[new_ids,0]),*a])@newB.T;curve.append(anchor3)
            rec.update(anchor_3d=anchor3,update_ms=(perf_counter()-clock)*1000,status='ACCEPTED',failure_reason='',total_ms=(perf_counter()-st)*1000)
            C=Cnew;B=newB;W=cfg['window']
            if rec['range_far']>=cfg['max_range']:reason='RESEARCH_RANGE_LIMIT';break
        except (ValueError,np.linalg.LinAlgError) as e:
            reason=str(e) if str(e) in ('LOCAL_FRAME_DEGENERATE','ORIENTATION_OPT_FAILED','GT_UNAVAILABLE_AHEAD') else 'LOCAL_FRAME_DEGENERATE'
            rec['exception']=str(e);break
    if steps and steps[-1]['status']!='ACCEPTED':
        steps[-1].update(failure_reason=reason,total_ms=(perf_counter()-st)*1000)
    for rec in steps:
        rec['unattributed_ms']=max(0.,rec['total_ms']-sum(rec.get(k,0.) for k in ('pca_ms','bishop_ms','refinement_ms','projection_ms','template_ms','update_ms')))
    result.update(status='STOPPED',reason=reason,indices=np.concatenate(idx),point_step=np.concatenate(point_step),local_uvw=np.concatenate(local),
                  template_residual=np.concatenate(residuals),confidence=np.concatenate(confidences),point_anchor=np.concatenate(anchors),
                  s_from_seed=np.concatenate(arc),curve=np.asarray(curve),total_ms=(perf_counter()-begin)*1000)
    return result
