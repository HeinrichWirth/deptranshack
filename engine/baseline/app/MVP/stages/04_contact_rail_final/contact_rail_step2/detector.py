"""Minimal F4-NU production inference. Depends only on NumPy and SciPy.

No GT, recording identity, learned uncertainty, clustering or plotting dependency.
The frozen old confidence still decides acceptance. The pre-guard confidence is
computed AFTER that decision and cannot change status, side, anchor or support.
"""
from pathlib import Path
import json
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

GEOMETRY_KEYS={'uvw','point_id','source_row','intensity','ring','sensor_timestamp','frame_index','azimuth_ordinal'}

class ContactRailDetector:
    def __init__(self):
        folder=Path(__file__).resolve().parent
        self.cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        c=self.cfg
        with np.load(folder/'canonical_template.npz',allow_pickle=False) as a:
            origin=a['origin'];cell=float(a['cell'])
            mask=a['smooth']>=c['template_threshold_relative']*a['smooth'].max()
            ij=np.argwhere(mask);self.template=origin+ij*cell
            self.reverse_template=origin+ij[np.all(ij%2==0,axis=1)]*cell
        self.cell=cell;self.tree=cKDTree(self.template)
        self.low=self.template.min(axis=0)-c['template_padding'];self.high=self.template.max(axis=0)+c['template_padding']
        self.mean=np.array(c['prior_mean']);self.sigma=np.array(c['prior_sigma'])
        self.lo=np.array([c['roi_d'][0],c['roi_h'][0]]);self.hi=np.array([c['roi_d'][1],c['roi_h'][1]])

    def _roi(self,cloud,rail,side):
        sgn=(-1,1)[side];uvw=cloud['uvw']
        rel=np.column_stack((sgn*(uvw[:,1]-rail[0]),uvw[:,2]-rail[1]))
        ids=np.flatnonzero(np.all((rel>=self.lo)&(rel<=self.hi),axis=1)&(uvw[:,0]>=0)&(uvw[:,0]<=self.cfg['slab_m']))
        points,inv=np.unique(rel[ids],axis=0,return_inverse=True)
        return dict(p=points,ids=ids,inv=inv,rail=rail,side=side,sgn=sgn)

    def _scores(self,poses,points,point_tree,spacing=None):
        # Preserve the original vector operation order for coordinate/support parity.
        cfg=self.cfg;out=[];r=np.array([[np.cos(0.),-np.sin(0.)],[np.sin(0.),np.cos(0.)]])
        if spacing is None:spacing=np.median(point_tree.query(points,k=min(3,len(points)))[0][:,-1]) if len(points)>1 else .1
        tol=max(self.cell,1.5*spacing);shape_scale=cfg['shape_scale']
        for pos in np.array_split(poses,max(1,int(np.ceil(len(poses)/48)))):
            rel=(points[None,:,:]-pos[:,None,:])@r
            inside=np.all((rel>=self.low)&(rel<=self.high),axis=2)
            dist=self.tree.query(rel.reshape(-1,2))[0].reshape(inside.shape)
            count=np.sum(inside,axis=1)
            ordered=np.sort(np.where(inside,dist,np.inf),axis=1)
            n=np.maximum(1,np.floor(.6*count).astype(int));mask=np.arange(ordered.shape[1])[None,:]<n[:,None]
            robust=np.sum(np.where(mask,np.minimum(ordered,.5),0),axis=1)/n
            obs=np.exp(-robust/shape_scale)
            target=self.reverse_template@r.T+pos[:,None,:]
            reverse=point_tree.query(target.reshape(-1,2))[0].reshape(len(pos),-1)
            chamfer=np.mean(reverse,axis=1)
            coverage=np.exp(-chamfer/max(tol,self.cell))
            delta=(pos-self.mean)/self.sigma;geo=np.exp(-.5*np.sum(delta*delta,axis=1))
            ww=cfg['weights'];scores=ww['geometry']*geo+ww['observed']*obs+ww['coverage']*coverage
            scores[count<cfg['min_observations']]=-1e6
            out.append(np.column_stack((scores,geo,obs,coverage,count,robust,np.full(len(pos),tol))))
        return np.concatenate(out)

    def _candidate(self,pose,roi,cloud,score):
        r=np.array([[np.cos(0.),-np.sin(0.)],[np.sin(0.),np.cos(0.)]])
        rel=(roi['p']-pose)@r;distance=self.tree.query(rel)[0]
        inside=np.all((rel>=self.low)&(rel<=self.high),axis=1)
        tolerance=max(self.cfg['shape_scale'],float(score[6]))
        explained=inside&(distance<=tolerance)
        support=roi['ids'][explained[roi['inv']]];n=int(np.sum(explained))
        return dict(side=('LEFT','RIGHT')[roi['side']],anchor=roi['rail']+pose*np.array([roi['sgn'],1]),
            base_score=float(score[0]),score=float(score[0]),geometry=float(score[1]),shape=float(score[2]),coverage=float(score[3]),
            n=n,reason='insufficient_returns' if n<self.cfg['min_observations'] else '',support=support)

    def _fit(self,roi,cloud):
        p=roi['p'];margin=self.cfg['fixed_context_m']
        points=p[np.all((p-self.mean>=self.low-margin)&(p-self.mean<=self.high+margin),axis=1)]
        if len(points)<self.cfg['min_observations']:return []
        tree=cKDTree(p);out=[]
        spacing=np.median(tree.query(p,k=min(3,len(p)))[0][:,-1]) if len(p)>1 else .1
        for seed in (self.mean,np.median(points,axis=0)):
            def residual(pose):
                dist=self.tree.query(points-pose)[0]/self.cfg['shape_scale']
                prior=(pose-self.mean)/self.sigma*.1*np.sqrt(len(points))
                return np.r_[dist,prior]
            opt=least_squares(residual,np.clip(seed,self.lo+1e-8,self.hi-1e-8),bounds=(self.lo,self.hi),
                loss=self.cfg['robust_loss'],f_scale=1.,max_nfev=35,diff_step=1e-3)
            score=self._scores(opt.x[None,:],p,tree,spacing=spacing)[0]
            out.append(self._candidate(opt.x,roi,cloud,score))
        return out

    def _distinct(self,a,b):
        return a['side']!=b['side'] or np.linalg.norm(a['anchor']-b['anchor'])>self.cfg['distinct_distance_m']

    def _confidence(self,best,margin):
        # All scales and weights are the frozen F4 formula. The patch changes only
        # the margin argument, never this function or the acceptance decision.
        return float((1-np.exp(-best['n']/5))*(.3+.7*best['geometry'])*(.4+.6*best['shape'])*1.*1.*(1-np.exp(-max(0,margin)/.3)))

    def _decide(self,candidates,rails_available=True):
        pre=[c for c in candidates if not c['reason']]
        post=[]
        if pre:
            highest=max(c['base_score'] for c in pre)
            post=sorted([c for c in pre if c['base_score']>=highest-self.cfg['ranking_guard']],key=lambda c:-c['score'])
        result=dict(status='NOT_FOUND',side=None,anchor=None,confidence=0.,reason='no_supported_candidate' if rails_available else 'running_pair_unavailable',
            support=np.empty(0,dtype=np.int64),selected=None,candidates=candidates,all_valid_candidates_pre_guard=pre,ranked_candidates_post_guard=post,
            post_guard_competitor=None,post_guard_margin=None)
        if post:
            best=post[0];other=next((c for c in post[1:] if self._distinct(best,c)),None)
            margin=best['score']-other['score'] if other else self.cfg['no_competitor_margin']
            confidence=self._confidence(best,margin);status=best['side'];reason=''
            if other and other['side']!=best['side'] and margin<self.cfg['side_margin']:status='AMBIGUOUS';reason='competing_sides'
            elif confidence<self.cfg['decision_confidence_threshold']:status='NOT_FOUND';reason='low_confidence'
            result.update(status=status,side=best['side'] if status in ('LEFT','RIGHT') else None,anchor=best['anchor'],support=best['support'],
                selected=best,confidence=confidence,reason=reason,post_guard_competitor=other,post_guard_margin=float(margin))
        return result

    def _bookkeeping(self,result):
        start=perf_counter();best=result['selected'];old_conf=result['confidence']
        pre=result['all_valid_candidates_pre_guard'];post=result['ranked_candidates_post_guard']
        runner=max((c for c in pre if self._distinct(best,c)),key=lambda c:c['base_score'],default=None) if best is not None else None
        margin=float(best['base_score']-runner['base_score']) if runner is not None else None
        effective=margin if runner is not None else self.cfg['no_competitor_margin']
        new_conf=self._confidence(best,effective) if best is not None else 0.
        other=result['post_guard_competitor'];removed=runner is not None and not any(c is runner for c in post)
        case='NO_SELECTED_CANDIDATE' if best is None else 'C_NO_COMPETITOR' if runner is None else 'A_STRONG_REAL_COMPETITOR' if margin<=self.cfg['confidence_margin_scale'] else 'B_WEAK_REAL_COMPETITOR'
        result.update(confidence_old=old_conf,confidence_new=new_conf,confidence=new_conf,
            selected_score=None if best is None else best['base_score'],best_pre_guard_competitor_score=None if runner is None else runner['base_score'],
            best_post_guard_competitor_score=None if other is None else other['base_score'],pre_guard_margin=margin,
            confidence_margin=float(effective) if best is not None else None,runner_up_side=None if runner is None else runner['side'],
            runner_up_v=None if runner is None else float(runner['anchor'][0]),runner_up_w=None if runner is None else float(runner['anchor'][1]),
            runner_up_was_removed_by_guard=removed,competitor_case=case,confidence_synthetic_margin=best is not None and runner is None)
        result['bookkeeping_ms']=1000*(perf_counter()-start)
        return result

    def _predict(self,geometry,running_rails):
        if not isinstance(geometry,dict) or not set(geometry)<=GEOMETRY_KEYS:raise ValueError('Geometry-only input required; labels/GT/recording identity are forbidden')
        cloud=dict(geometry);cloud['uvw']=np.asarray(cloud['uvw'],dtype=np.float64)
        if cloud['uvw'].ndim!=2 or cloud['uvw'].shape[1]!=3 or not np.isfinite(cloud['uvw']).all():raise ValueError('uvw must be a finite N x 3 array in metres')
        rails=None if running_rails is None else np.asarray(running_rails,dtype=np.float64)
        if rails is not None and (rails.shape!=(2,2) or not np.isfinite(rails).all() or rails[0,0]>=rails[1,0]):raise ValueError('Expected ordered LEFT/RIGHT running rails, each [v,w]')
        candidates=[]
        if rails is not None:
            for side,rail in enumerate(rails):
                roi=self._roi(cloud,rail,side)
                if len(roi['p'])>=self.cfg['min_observations']:candidates.extend(self._fit(roi,cloud))
        result=self._decide(candidates,rails is not None)
        return self._bookkeeping(result)

    def detect(self,geometry,running_rails):
        """Status is authoritative: never reapply the old 0.04 gate to new confidence.

        Anchor is null for refusals. support_indices retain the frozen proposal's
        exact indices even for AMBIGUOUS; they are not a claim of accepted CR.
        Confidence is heuristic relative reliability, not a probability or guarantee.
        """
        r=self._predict(geometry,running_rails);accepted=r['status'] in ('LEFT','RIGHT')
        return dict(status=r['status'],side=r['side'],anchor_v=float(r['anchor'][0]) if accepted else None,
            anchor_w=float(r['anchor'][1]) if accepted else None,confidence=r['confidence'],support_indices=r['support'].copy(),
            diagnostics=dict(selected_score=r['selected_score'],runner_up_score=r['best_pre_guard_competitor_score'],score_margin=r['confidence_margin'],
                runner_up_side=r['runner_up_side'],runner_up_v=r['runner_up_v'],runner_up_w=r['runner_up_w'],runner_up_removed_by_guard=r['runner_up_was_removed_by_guard'],
                competitor_case=r['competitor_case'],synthetic_no_competitor_margin=r['confidence_synthetic_margin'],
                frozen_decision_confidence=r['confidence_old'],reason=r['reason'],confidence_semantics='heuristic relative reliability after pre-guard runner-up bookkeeping; not calibrated probability'))

@lru_cache(maxsize=1)
def _default_detector():return ContactRailDetector()

def detect_contact_rail(geometry,running_rails):
    return _default_detector().detect(geometry,running_rails)
