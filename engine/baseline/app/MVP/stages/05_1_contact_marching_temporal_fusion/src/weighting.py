"""Diagnostic frame-balanced statistics; geometry-only, original point support."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from tracker import TemplateTracker

def balanced_weights(frames,distance,tau=None):
    _,inv,count=np.unique(frames,return_inverse=True,return_counts=True)
    w=1/count[inv]
    if tau is not None:w=w*np.exp(-np.asarray(distance)/tau)
    return w

def weighted_pca(p,w,previous):
    if len(p)<3:raise ValueError('LOCAL_FRAME_DEGENERATE')
    w=w/w.sum();c=np.sum(p*w[:,None],axis=0);q=p-c
    values,vectors=np.linalg.eigh((q*w[:,None]).T@q);t=vectors[:,-1]
    if values[-1]<1e-8:raise ValueError('LOCAL_FRAME_DEGENERATE')
    if t@previous<0:t=-t
    return t,c

def quantile(x,w,f):
    k=np.argsort(x);return np.interp(f*np.sum(w),np.cumsum(w[k])-.5*w[k],x[k])

def weighted_binned(p,axis,cell,w):
    s=p@axis;b=np.floor((s-s.min())/cell).astype(int)
    return np.array([[quantile(p[b==i,j],w[b==i],.5) for j in range(3)] for i in np.unique(b)])

class WeightedTemplateTracker(TemplateTracker):
    def fit(self,p,center,gate,fixed=False,source_frames=None,age_distance=None,tau=None):
        center=np.asarray(center);p=np.asarray(p)
        region=np.all((p-center>=self.lo-gate)&(p-center<=self.hi+gate),axis=1)
        if source_frames is None:return super().fit(p,center,gate,fixed)
        points,inv=np.unique(p[region],axis=0,return_inverse=True)
        if len(points)<self.cfg['min_support']:return [],dict(candidate_n=len(points))
        original=balanced_weights(np.asarray(source_frames)[region],np.asarray(age_distance)[region],tau)
        w=np.bincount(inv,weights=original,minlength=len(points));w=w/w.mean()
        _,ix,cell_inverse=np.unique(np.floor(points/.002).astype(np.int32),axis=0,return_index=True,return_inverse=True);fitp=points[ix];fitw=np.bincount(cell_inverse,weights=w);fitw=fitw/fitw.mean()
        def loss(z):return np.array([fitw*np.log1p(z),fitw/(1+z),-fitw/(1+z)**2])
        seeds=[center] if fixed else [center,center+[gate*.7,0],center+[-gate*.7,0],center+[0,gate*.7],center+[0,-gate*.7]]
        candidates=[]
        for seed in seeds:
            if fixed:anchor=center;success=True
            else:
                opt=least_squares(lambda a:self.distances(fitp,a)/.015,np.clip(seed,center-gate+1e-9,center+gate-1e-9),bounds=(center-gate,center+gate),loss=loss,diff_step=1e-3,max_nfev=24)
                anchor=opt.x;success=bool(opt.success)
            d=self.distances(points,anchor);inside=np.all((points-anchor>=self.lo)&(points-anchor<=self.hi),axis=1);support=inside&(d<=self.cfg['support_tolerance']);n=int(support.sum())
            if n:
                threshold=quantile(d[inside],w[inside],.7);trim=inside&(d<=threshold);err=float(np.average(d[trim],weights=w[trim]))
                coverage=float(np.mean(cKDTree(points).query(self.template[::2]+anchor)[0]<=.025));score=float(np.exp(-err/.015)+.25*coverage+.1*(1-np.exp(-n/8)))
            else:err=.5;coverage=0.;score=-1.
            cand=dict(anchor=anchor,score=score,residual=err,support_unique=n,coverage=coverage,success=success)
            if not any(np.linalg.norm(anchor-c['anchor'])<.006 for c in candidates):candidates.append(cand)
        candidates.sort(key=lambda c:-c['score']);return candidates,dict(candidate_n=len(points))
