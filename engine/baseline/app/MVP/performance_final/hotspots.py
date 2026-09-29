"""Exact fit memoization and instrumentation around unmodified C4 functions."""
from . import ROOT
import numpy as np
import long_common as lc
import tracker
import long_tracker
import roi_source
import time
import copy
from collections import OrderedDict
from scipy.spatial import cKDTree
from .engine import clone
from MVP.final_pipeline.pipeline import RawRoi


class Profile:
    def __init__(self):self.events=[];self.fits=[];self.optimizers=[];self.captures=[]
    def timed(self,name,fn):
        def run(*args,**kwargs):
            t=time.perf_counter();value=fn(*args,**kwargs)
            self.events.append(dict(operation=name,seconds=time.perf_counter()-t,size=len(args[0]) if args and isinstance(args[0],np.ndarray) and args[0].ndim else None))
            return value
        return run
    def tree(self,points,*a,**kw):
        t=time.perf_counter();obj=cKDTree(points,*a,**kw)
        self.events.append(dict(operation='KDTree_build',seconds=time.perf_counter()-t,size=len(points),dimensions=points.shape[1]))
        profile=self
        class Tree:
            def query(self,*args,**kwargs):return profile.timed('KDTree_query',obj.query)(*args,**kwargs)
            def query_ball_point(self,*args,**kwargs):return profile.timed('KDTree_radius',obj.query_ball_point)(*args,**kwargs)
        return Tree()
    def least_squares(self,fun,x,*a,**kw):
        residual_seconds=0.;calls=0
        closure={name:cell.cell_contents for name,cell in zip(fun.__code__.co_freevars,fun.__closure__ or ())}
        candidate_points=len(closure['fitp']) if 'fitp' in closure else None
        def residual(z):
            nonlocal residual_seconds,calls
            t=time.perf_counter();v=fun(z);residual_seconds+=time.perf_counter()-t;calls+=1;return v
        t=time.perf_counter();opt=tracker.least_squares(residual,x,*a,**kw);total=time.perf_counter()-t
        self.optimizers.append(dict(seconds=total,residual_seconds=residual_seconds,non_residual_seconds=total-residual_seconds,
            residual_calls=calls,nfev=opt.nfev,njev=opt.njev,iterations=None,iterations_note='not exposed by scipy least_squares result',
            status=int(opt.status),parameters=len(x),points_per_candidate=candidate_points))
        return opt


def optimized_fit(self,p,center,gate,fixed=False):
    # Same formulas, arithmetic order, initialization, loss and candidate ordering.
    center=np.asarray(center);p=np.asarray(p)
    region=np.all((p-center>=self.lo-gate)&(p-center<=self.hi+gate),axis=1)
    points=np.unique(p[region],axis=0)
    if len(points)<self.cfg['min_support']:return [],dict(candidate_n=len(points))
    _,ix=np.unique(np.floor(points/.002).astype(np.int32),axis=0,return_index=True);fitp=points[ix]
    candidates=[];reverse_tree=None
    seeds=[center] if fixed else [center,center+[gate*.7,0],center+[-gate*.7,0],center+[0,gate*.7],center+[0,-gate*.7]]
    lo,hi=center-gate,center+gate
    for seed in seeds:
        if fixed:anchor=center;success=True
        else:
            def residual(a):return self.distances(fitp,a)/.015
            opt=tracker.least_squares(residual,np.clip(seed,center-gate+1e-9,center+gate-1e-9),bounds=(lo,hi),
                loss='cauchy',f_scale=1.,diff_step=1e-3,max_nfev=24)
            anchor=opt.x;success=bool(opt.success)
        d=self.distances(points,anchor);inside=np.all((points-anchor>=self.lo)&(points-anchor<=self.hi),axis=1)
        support=inside&(d<=self.cfg['support_tolerance']);n=int(support.sum())
        if n:
            err=float(np.mean(np.sort(d[inside])[:max(1,int(.7*inside.sum()))]))
            if reverse_tree is None:reverse_tree=cKDTree(points)
            reverse=reverse_tree.query(self.template[::2]+anchor)[0]
            coverage=float(np.mean(reverse<=.025));score=float(np.exp(-err/.015)+.25*coverage+.1*(1-np.exp(-n/8)))
        else:err=.5;coverage=0.;score=-1.
        cand=dict(anchor=anchor,score=score,residual=err,support_unique=n,coverage=coverage,success=success)
        if not any(np.linalg.norm(anchor-c['anchor'])<.006 for c in candidates):candidates.append(cand)
    candidates.sort(key=lambda c:-c['score'])
    return candidates,dict(candidate_n=len(points))


def make_tracker(backend,profile):
    basefit=clone(tracker.TemplateTracker.fit,least_squares=profile.least_squares,cKDTree=profile.tree) if profile else tracker.TemplateTracker.fit
    init=clone(tracker.TemplateTracker.__init__,cKDTree=profile.tree) if profile else tracker.TemplateTracker.__init__
    native=None
    if backend=='native':
        from .native_backend import load_native
        native=load_native()  # Explicit import error, never a silent fallback.
    class Tracker(tracker.TemplateTracker):
        def __init__(self,*a,**kw):init(self,*a,**kw);self.memo=OrderedDict();self.hits=0
        def distances(self,p,anchor):
            if native is not None:return native.distances(p,anchor,self.template)
            return super().distances(p,anchor)
        def fit(self,p,center,gate,fixed=False):
            t=time.perf_counter();key=None
            if backend in ('optimized','memo','native'):
                key=(np.asarray(p).shape,np.asarray(p).dtype.str,np.asarray(p).tobytes(),np.asarray(center,dtype=float).tobytes(),float(gate),fixed)
                if key in self.memo:
                    self.hits+=1;return copy.deepcopy(self.memo[key])
            if profile is not None and len(profile.captures)<128:
                profile.captures.append(dict(points=np.array(p,copy=True),center=np.array(center,copy=True),gate=gate,fixed=fixed,
                    template=self.template.copy(),cfg=dict(self.cfg)))
            result=(optimized_fit if backend in ('optimized','hoist','native') else basefit)(self,p,center,gate,fixed)
            if key is not None:
                self.memo[key]=copy.deepcopy(result)
                if len(self.memo)>256:self.memo.popitem(last=False)
            if profile is not None:profile.fits.append(dict(seconds=time.perf_counter()-t,input_points=len(p),candidate_points=result[1]['candidate_n'],candidates=len(result[0])))
            return result
    return Tracker


def make_infer(backend,profile=None):
    overrides=dict(TemplateTracker=make_tracker(backend,profile))
    if profile:
        for name in ('candidate_records','frame_for','clone_path','can_promote','template_part'):
            overrides[name]=profile.timed(name,getattr(long_tracker,name))
    return clone(long_tracker.infer,**overrides)


def make_roi(profile=None):
    if profile is None:return RawRoi
    sphere_impl=clone(roi_source.RoiSource.sphere,cKDTree=profile.tree)
    class Roi(RawRoi):
        sphere=profile.timed('sphere_full',sphere_impl)
        query=profile.timed('ROI_query_full',roi_source.RoiSource.query)
        materialize=profile.timed('materialize',roi_source.RoiSource.materialize)
    return Roi
