import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares,minimize
from geometry_math import unit,bishop

def plan_frame(initial,up):
    z=unit(up);x=unit(initial[:,0]-z*(z@initial[:,0]));y=np.cross(z,x)
    return np.column_stack((x,y,z))

def grid(end,step=1.):
    s=np.arange(0,end+1e-8,step);s=s[s<end-1e-7]
    return np.r_[s,end] if end>1e-7 else np.array([0.])

def cluster(s,A,window=.25):
    groups=[];start=0
    for i in range(1,len(s)+1):
        if i==len(s) or s[i]-s[start]>window:
            groups.append(np.arange(start,i));start=i
    ss=np.array([np.mean(s[g]) for g in groups]);aa=np.array([np.mean(A[g],axis=0) for g in groups])
    # Endpoints retain the source horizon; clustering changes constraints, not data.
    if len(ss)>1:ss[0]=s[0];ss[-1]=s[-1]
    return ss,aa,groups

def curve_frames(s,C,initial):
    if len(s)<2:raise ValueError('Curve needs two stations')
    # Close floating-point endpoint duplicates are removed before this routine.
    assert np.all(np.diff(s)>1e-7)
    return bishop(unit(np.gradient(C,s,axis=0,edge_order=1)),initial)

def rho_weights(r,loss):
    r=np.maximum(np.asarray(r),1e-12)
    if loss=='huber':return np.minimum(1,1.5/r)
    if loss=='cauchy':return 1/(1+(r/2.)**2)
    if loss=='tukey':return np.maximum(0,1-(r/4.685)**2)**2
    if loss=='student':return 5/(4+r*r)
    raise ValueError(loss)

def bspline_design(s,end,spacing,degree=3):
    internal=np.arange(spacing,end-1e-8,spacing);knots=np.r_[np.zeros(degree+1),internal,np.full(degree+1,end)]
    base=BSpline(knots,np.eye(len(knots)-degree-1),degree,extrapolate=False)
    return base(s),base

def smooth_spline(obs_s,obs,obs_cov,query,frame,cfg):
    """IRLS measurement factors and separately squared physical regularization.

    Soft radius prior is optimized, NEVER implemented by clipping curvature.
    The optional projection variant uses actual SLSQP nonlinear inequalities.
    """
    end=float(query[-1]);X,base=bspline_design(obs_s,end,cfg.get('knots',4));Q=base(query);fine=grid(end,.5);D1=base.derivative(1)(fine);D2=base.derivative(2)(fine);D3=base.derivative(3)(fine)
    Y=obs@frame;sig=np.sqrt(np.maximum(np.einsum('ia,nij,ja->na',frame,obs_cov,frame),1e-6));n=X.shape[1];coef=np.linalg.lstsq(X,Y,rcond=None)[0];prior=cfg.get('prior','all');lam=cfg.get('smooth',1.)
    L=[[],[],[]]
    if prior in ('kappa','kappa_prime','all'):
        for j in (0,1):L[j]+=[D2*np.sqrt(lam)/.025]
    if prior in ('kappa_prime','all'):
        for j in (0,1):L[j]+=[D3*np.sqrt(lam)/.004]
    if prior in ('vertical','all'):L[2]+=[D2*np.sqrt(lam)/cfg.get('vertical_prior',.015),D3*np.sqrt(lam)/cfg.get('vertical_prior',.015)]
    L=[np.vstack(v) if v else np.zeros((0,n)) for v in L];history=[]
    for iteration in range(6):
        residual=(X@coef-Y)/sig;w=rho_weights(np.linalg.norm(residual,axis=1),cfg.get('loss','huber'));w=np.maximum(w,1e-5)
        for j in range(3):
            W=np.sqrt(w)/sig[:,j];M=np.vstack([X*W[:,None],L[j]]);b=np.r_[Y[:,j]*W,np.zeros(len(L[j]))];coef[:,j]=np.linalg.lstsq(M,b,rcond=None)[0]
        history.append(float(np.sum(w[:,None]*((X@coef-Y)/sig)**2)))
    def curvature(c):
        v=D1@c;a=D2@c;return (v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(np.sum(v[:,:2]**2,axis=1)**1.5,1e-8)
    if cfg.get('physics',False):
        W=np.sqrt(w)[:,None]/sig
        def fun(flat):
            c=flat.reshape(n,3);r=[((X@c-Y)*W).ravel()]+[L[j]@c[:,j] for j in range(3)]
            k=curvature(c);r+=[np.maximum(abs(k)-.01,0)/.001]
            return np.concatenate(r)
        fit=least_squares(fun,coef.ravel(),max_nfev=35,ftol=1e-6,xtol=1e-7,gtol=1e-6);coef=fit.x.reshape(n,3)
        if cfg.get('hard_projection',False):
            fit2=minimize(lambda x:np.sum(fun(x)**2),coef.ravel(),method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:.01-np.abs(curvature(x.reshape(n,3))))],options=dict(maxiter=80,ftol=1e-7))
            coef=fit2.x.reshape(n,3);history.append(dict(hard_success=bool(fit2.success),max_abs_kappa=float(max(abs(curvature(coef))))))
    cov=[]
    # Conditional fit covariance plus observation residual floor; no count of raw
    # duplicate LiDAR returns enters this information matrix.
    for j in range(3):
        W=np.sqrt(w)/sig[:,j];H=(X*W[:,None]).T@(X*W[:,None])+L[j].T@L[j]+np.eye(n)*1e-8
        cov.append(np.einsum('ij,jk,ik->i',Q,np.linalg.pinv(H),Q))
    local=np.stack(cov,axis=1);world=np.einsum('ij,nj,kj->nik',frame,np.maximum(local,.003**2),frame)
    final_objective=float(np.sum(w[:,None]*((X@coef-Y)/sig)**2)+sum(np.sum((L[j]@coef[:,j])**2) for j in range(3)))
    if cfg.get('physics',False):final_objective+=float(np.sum((np.maximum(abs(curvature(coef))-.01,0)/.001)**2))
    return Q@coef@frame.T,world,dict(iterations=6,objective_history=history,final_objective=final_objective,weights=w.tolist())
