"""Observed-to-empirical-template latent translation. Pure array API; no GT.

Each source contributes a normalized likelihood, then sources are combined.
Visibility is marginalized; unobserved template surfaces have no absence cost.
Top modes remain distinct. Reported mean is the winning mode, never their mean.
"""
import time
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.ndimage import map_coordinates
from functools import lru_cache
from curve_geometry import rho_weights

MODES=('FULL','TOP_ONLY','BOTTOM_ONLY','VERTICAL_ONLY','CORNER','SPARSE_MULTI','ONE_FACE','UNKNOWN')

class TemplateLikelihood:
    def __init__(self,template,representation='pointset'):
        self.template=np.asarray(template,float);self.kind=representation;self.tree=cKDTree(self.template)
        t=self.template;lo=np.quantile(t,.25,axis=0);hi=np.quantile(t,.75,axis=0)
        self.subsets=[t,t[t[:,1]>=hi[1]],t[t[:,1]<=lo[1]],t[t[:,0]<=lo[0]],t[(t[:,0]<=lo[0])|(t[:,1]>=hi[1])],t,t[(t[:,0]>=hi[0])|(t[:,1]<=lo[1])],t]
        self.trees=[cKDTree(v) for v in self.subsets];self.normals=[]
        for point in t:
            _,ids=self.tree.query(point,k=min(9,len(t)));cov=np.cov(t[ids].T);self.normals.append(np.linalg.eigh(cov)[1][:,0])
        self.normals=np.array(self.normals)
        h=ConvexHull(t);self.hull=t[h.vertices]
    def distance(self,q):
        if self.kind=='occupancy':
            d,ids=self.tree.query(q,k=8);sigma=.008
            # Probability occupancy, normalized by local kernel count. Not an
            # inverse/template-coverage score; missing surfaces cost nothing.
            energy=-2*sigma*sigma*(logsumexp(-d*d/(2*sigma*sigma),axis=1)-np.log(8))
            return np.sqrt(np.maximum(energy,0)),ids[:,0]
        if self.kind=='envelope':
            a=self.hull;b=np.roll(a,-1,axis=0);v=b-a;delta=q[:,None,:]-a;t=np.clip(np.einsum('nki,ki->nk',delta,v)/np.sum(v*v,axis=1),0,1);d=np.linalg.norm(delta-t[:,:,None]*v,axis=2)
            return d.min(axis=1),self.tree.query(q)[1]
        return self.tree.query(q)
    def visible_likelihood(self,q,sigma):
        if self.kind=='subsets':d=np.column_stack([tree.query(q)[0] for tree in self.trees])
        else:
            d0,_=self.distance(q);d=np.column_stack([d0]+[tree.query(q)[0] for tree in self.trees[1:]])
        # Each mode evaluated only on observed points. Uniform mode prior.
        costs=np.mean(np.log1p((d/(2*sigma))**2),axis=0)
        p=np.exp(-costs-logsumexp(-costs));return float(-logsumexp(-costs)+np.log(len(costs))),p

@lru_cache(maxsize=16)
def cached_template(raw,representation):
    return TemplateLikelihood(np.frombuffer(raw,dtype=np.float64).reshape(-1,2),representation)

def fit_anchor(points,source,template,prior=None,range_m=None,ring=None,age_distance=None,state=None,config=None):
    begin=time.perf_counter();cfg=config or {};P=np.asarray(points,float);source=np.asarray(source);prior=np.zeros(2) if prior is None else np.asarray(prior,float);n=len(P)
    if not n:return dict(anchor=prior,covariance=np.eye(2)*.20**2,modes=[],visibility='UNKNOWN',visibility_probabilities=np.ones(8)/8,n=0,sources=0,coverage=0.,residual=None,state='LOW_SUPPORT',ms=0.,fusion_ms=0.)
    # All raw point IDs retained by caller. Here deterministic millimetre cells
    # prevent repeated identical records inflating a single source likelihood.
    mod=cached_template(np.asarray(template,dtype=np.float64).tobytes(),cfg.get('representation','pointset'));sources=[];fusion_start=time.perf_counter()
    for src in np.unique(source):
        ids=np.flatnonzero(source==src);_,ind=np.unique(np.round(P[ids],3),axis=0,return_index=True);ids=ids[np.sort(ind)]
        if len(ids)>160:ids=ids[np.linspace(0,len(ids)-1,160,dtype=int)]
        sigma=.008+(.0001*float(np.median(np.asarray(range_m)[ids])) if range_m is not None else 0)
        st=np.ones(len(ids)) if state is None else np.where(np.asarray(state)[ids]==2,1.,np.where(np.asarray(state)[ids]==1,.25,0.))
        if not st.any():continue
        div=len(np.unique(np.asarray(ring)[ids])) if ring is not None else 1
        quality=1.
        if cfg.get('source_weight','equal')=='quality':quality=min(1,np.sqrt(len(ids)/8))*min(1,np.sqrt(div/3))/(1+sigma/.03)
        sources.append(dict(ids=ids,sigma=sigma,weight=quality,point_weight=st/st.sum()))
    if not sources:return fit_anchor(P[:0],source[:0],template,prior,config=cfg)
    if cfg.get('fusion','balanced')=='concat':
        for src in sources:src['weight']=len(src['ids'])
    total=sum(s['weight'] for s in sources)
    for s in sources:s['weight']/=total
    fusion_ms=(time.perf_counter()-fusion_start)*1000
    # Smooth Cauchy likelihood supports noisy/partial empirical profiles; robust
    # loss alternatives use their IRLS-integrated energies consistently.
    loss=cfg.get('loss','cauchy')
    def energy(r):
        if loss=='huber':return np.where(r<=1.5,.5*r*r,1.5*(r-.75))
        if loss=='tukey':return (4.685**2/6)*(1-np.maximum(0,1-(r/4.685)**2)**3)
        if loss=='student':return 2.5*np.log1p(r*r/4)
        return np.log1p(r*r/4)
    # Batch the geometry lookup; per-source normalization remains explicit.
    ids_all=np.concatenate([src['ids'] for src in sources]);src_idx=np.repeat(np.arange(len(sources)),[len(src['ids']) for src in sources]);pw=np.concatenate([src['point_weight'] for src in sources]);sigma_all=np.concatenate([np.full(len(src['ids']),src['sigma']) for src in sources]);PP=P[ids_all]
    def objective(shift,details=False):
        qq=PP-shift;d,_=mod.distance(qq);costs=np.bincount(src_idx,weights=pw*energy(d/sigma_all),minlength=len(sources))
        # Marginalize all permitted modes without knowing any GT visibility.
        # Full empirical likelihood remains primary; no template->observed term.
        mm=np.stack([costs]+[np.bincount(src_idx,weights=pw*energy(tree.query(qq)[0]/sigma_all),minlength=len(sources)) for tree in mod.trees[1:]],axis=1);marginal=-logsumexp(-mm,axis=1)+np.log(8)
        costs=marginal if mod.kind=='subsets' else .85*costs+.15*marginal
        out=float(np.dot([src['weight'] for src in sources],costs));ps=[];vals=costs.tolist()
        if details:
            for src in sources:ps.append(mod.visible_likelihood(P[src['ids']]-shift,src['sigma'])[1])
        # Weak initialization preference ONLY breaks unidentifiable modes; local
        # Hessian below excludes it, so it cannot manufacture precision.
        out+=cfg.get('prior_weight',.005)*np.sum(((shift-prior)/.15)**2)
        return (out,ps,vals) if details else out
    # Candidate translations from real point minus empirical template samples.
    seeds=[prior.copy()];sample=P[np.linspace(0,n-1,min(n,35),dtype=int)];hyp=(sample[:,None,:]-mod.template[None,::6,:]).reshape(-1,2);cells=np.round(hyp/.01).astype(int);u,c=np.unique(cells,axis=0,return_counts=True)
    for h in u[np.argsort(c)[-20:][::-1]]*.01:
        if np.linalg.norm(h-prior)>.4 or any(np.linalg.norm(h-x)<.025 for x in seeds):continue
        seeds.append(h)
        if len(seeds)>=cfg.get('starts',6):break
    solutions=[]
    for seed in seeds:
        fit=minimize(objective,seed,method='Nelder-Mead',bounds=[(prior[0]-.4,prior[0]+.4),(prior[1]-.4,prior[1]+.4)],options=dict(maxiter=80,xatol=.0002,fatol=1e-5,initial_simplex=np.array([seed,seed+[.01,0],seed+[0,.01]])))
        candidate=dict(anchor=fit.x,cost=float(fit.fun)) if fit.fun<=objective(seed) else dict(anchor=seed,cost=objective(seed))
        close=[v for v in solutions if np.linalg.norm(candidate['anchor']-v['anchor'])<.006]
        if not close:solutions.append(candidate)
        elif candidate['cost']<close[0]['cost']:close[0].update(candidate)
    solutions.sort(key=lambda x:x['cost']);best=solutions[0];anchor=best['anchor'];_,ps,vals=objective(anchor,True)
    if cfg.get('source_weight','equal')=='robust' and len(sources)>=3:
        # Source-disagreement robustification does not let a dense bad source win.
        vv=np.array(vals);med=np.median(vv);weights=1/(1+(np.maximum(vv-med,0)/max(.1,med))**2)
        for src,w in zip(sources,weights):src['weight']*=w
        norm=sum(src['weight'] for src in sources)
        for src in sources:src['weight']/=norm
        fit=minimize(objective,anchor,method='Powell',options=dict(maxiter=15,xtol=.0003));anchor=fit.x;best=dict(anchor=anchor,cost=float(fit.fun));_,ps,vals=objective(anchor,True)
    H=np.zeros((2,2));coverage_ids=[];residuals=[]
    for src in sources:
        d,ids=mod.tree.query(P[src['ids']]-anchor);normals=mod.normals[ids];ww=src['weight']*src['point_weight']*rho_weights(d/src['sigma'],loss)
        observed=P[src['ids']]
        if len(observed)>=3:
            ev,evec=np.linalg.eigh(np.cov(observed.T))
            if ev[0]<.08*max(ev[1],1e-12):normals=np.repeat(evec[:,0][None],len(ids),axis=0)
        # Effective independent profile information capped per source: angular
        # sampling is correlated, it is not n raw-point independent evidence.
        H+=min(8,len(ids))*np.einsum('n,ni,nj->ij',ww,normals,normals)/(src['sigma']**2);coverage_ids.extend(ids.tolist());residuals.extend(d.tolist())
    eig,U=np.linalg.eigh(H);cov=U@np.diag(1/np.maximum(eig,1/.20**2))@U.T+np.eye(2)*.003**2;conditional_covariance=cov.copy()
    modes=[best]+[m for m in solutions if np.linalg.norm(m['anchor']-anchor)>.008 and m['cost']<best['cost']+.15][:2]
    # Include mode separation in uncertainty, but do NOT average their locations.
    for m in modes[1:]:delta=m['anchor']-anchor;cov+=.5*np.outer(delta,delta)
    probs=np.average(np.array(ps),axis=0,weights=[s['weight'] for s in sources]);visible=MODES[int(np.argmax(probs))];ext=np.ptp(P,axis=0)
    if n<=3:visible='SPARSE_MULTI'
    elif min(ext)<.015:visible='ONE_FACE'
    coverage=len(set(coverage_ids))/len(mod.template);res=float(np.median(residuals));status='LOW_SUPPORT' if n<4 else ('PROFILE_AMBIGUOUS' if len(modes)>1 or np.linalg.eigvalsh(cov).max()>.05**2 else 'INLIER')
    if res>.06:status='OUTLIER'
    return dict(anchor=anchor,covariance=cov,conditional_covariance=conditional_covariance,modes=modes,visibility=visible,visibility_probabilities=probs,n=n,sources=len(sources),coverage=coverage,residual=res,state=status,hessian=H,source_weights=[dict(source=int(source[s['ids'][0]]),weight=s['weight'],cost=float(v)) for s,v in zip(sources,vals)],ms=(time.perf_counter()-begin)*1000,fusion_ms=fusion_ms)
