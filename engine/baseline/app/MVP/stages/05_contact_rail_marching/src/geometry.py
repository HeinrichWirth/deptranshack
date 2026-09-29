"""Pure geometry: no data paths, labels, poses, or trajectory access."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def unit(x):
    x=np.asarray(x,dtype=float); n=np.linalg.norm(x)
    if n<1e-9 or not np.isfinite(n):raise ValueError('LOCAL_FRAME_DEGENERATE')
    return x/n

def transport(B,t):
    t=unit(t); a=B[:,0]; c=np.clip(a@t,-1.,1.); cross=np.cross(a,t)
    if c<-.999999:raise ValueError('LOCAL_FRAME_DEGENERATE')
    K=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    Q=np.eye(3)+K+K@K/(1+c)
    return Q@B

def binned(points,axis,bin_size):
    s=points@axis; bi=np.floor((s-s.min())/bin_size).astype(int)
    return np.array([np.median(points[bi==i],axis=0) for i in np.unique(bi)])

def pca(points,previous,robust=True):
    p=np.asarray(points); c=np.median(p,axis=0) if robust else p.mean(axis=0)
    if len(p)<3:raise ValueError('LOCAL_FRAME_DEGENERATE')
    q=p-c
    for k in range(3 if robust else 1):
        _,sv,V=np.linalg.svd(q,full_matrices=False); t=V[0]
        if sv[0]<.02:raise ValueError('LOCAL_FRAME_DEGENERATE')
        if robust:
            d=np.linalg.norm((p-c)-np.outer((p-c)@t,t),axis=1)
            med=np.median(d); mad=1.4826*np.median(abs(d-med))
            keep=d<=med+max(.005,3*mad)
            if keep.sum()<3:break
            c=np.mean(p[keep],axis=0); q=p[keep]-c
    if t@previous<0:t=-t
    return unit(t),c

def scatter(points,B,objective='trace'):
    yz=points@B[:,1:]; yz-=np.median(yz,axis=0)
    r2=np.sum(yz*yz,axis=1); cut=np.quantile(r2,.90); q=yz[r2<=cut]
    cov=q.T@q/max(1,len(q)); eig=np.maximum(np.linalg.eigvalsh(cov),1e-10)
    if objective=='trace':return float(eig.sum())
    if objective=='logdet':return float(np.log(eig+1e-6).sum())
    if objective in ('ellipse90','ellipse95'):
        mahal=np.einsum('ij,jk,ik->i',yz,np.linalg.inv(cov+np.eye(2)*1e-6),yz)
        return float(np.pi*np.sqrt(np.prod(eig+1e-6))*np.quantile(mahal,.90 if objective=='ellipse90' else .95))
    if objective=='mad':
        r=np.sqrt(r2); med=np.median(r); mad=1.4826*np.median(abs(r-med))
        return float(np.mean(r2[r<=med+max(.003,2.5*mad)]))
    raise ValueError(objective)

def refine(points,B,bound_deg,objective):
    bound=np.deg2rad(bound_deg); trace=[]
    def fun(v):
        BB=B@Rotation.from_rotvec([0.,v[0],v[1]]).as_matrix()
        val=scatter(points,BB,objective); trace.append([*np.rad2deg(v),val]); return val
    opt=minimize(fun,[0.,0.],method='Powell',bounds=[(-bound,bound)]*2,options={'maxiter':12,'xtol':1e-5,'ftol':1e-5})
    return B@Rotation.from_rotvec([0.,*opt.x]).as_matrix(),dict(success=bool(opt.success),candidates=trace,objective=float(opt.fun))

def angles(A,B):
    # Rotation-vector components resolved in the previous (u,v,w) frame.
    d=np.rad2deg(Rotation.from_matrix(A.T@B).as_rotvec())
    return dict(delta_roll=float(d[0]),delta_pitch=float(d[1]),delta_yaw=float(d[2]))
