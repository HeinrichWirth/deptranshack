"""Pure station/plan/vertical curve models. No labels, dataset paths, or identities.

The vertical axis is the saved registration world-up PROXY, not measured gravity.
PCHIP is component-wise in this explicitly defined plan frame, not rotation invariant.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline,CubicHermiteSpline,BSpline
from scipy.optimize import least_squares,minimize
from scipy.sparse import lil_matrix
from track_geometry import unit,bishop

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

def interpolation(s,A,query,frame,kind,window=0.):
    if window>0:s,A,_=cluster(s,A,window)
    V=A@frame
    if kind=='polyline':Y=np.column_stack([np.interp(query,s,V[:,j]) for j in range(3)])
    elif kind=='natural':Y=CubicSpline(s,V,bc_type='natural')(query)
    elif kind=='pchip':Y=PchipInterpolator(s,V)(query)
    elif kind=='hermite':
        # Hyman monotonicity limiter independently on longitudinal, plan lateral,
        # and vertical derivatives; no invented per-axis physical slope clipping.
        d=np.diff(V,axis=0)/np.diff(s)[:,None];m=np.vstack([d[0],(d[:-1]+d[1:])/2,d[-1]])
        for i in range(len(d)):
            for j in range(3):
                if abs(d[i,j])<1e-12:m[i:i+2,j]=0;continue
                a,b=m[i:i+2,j]/d[i,j]
                if a<0:m[i,j]=0;a=0
                if b<0:m[i+1,j]=0;b=0
                r=np.hypot(a,b)
                if r>3:m[i:i+2,j]*=3/r
        Y=CubicHermiteSpline(s,V,m)(query)
    else:raise ValueError(kind)
    return Y@frame.T

def curve_frames(s,C,initial):
    if len(s)<2:raise ValueError('Curve needs two stations')
    # Close floating-point endpoint duplicates are removed before this routine.
    assert np.all(np.diff(s)>1e-7)
    return bishop(unit(np.gradient(C,s,axis=0,edge_order=1)),initial)

def assign_stations(points,s,A,source,march_s):
    """Piecewise projection initialized by C4 longitudinal station, source-wise order.

    Sort within each source by the fixed initial chain progression, never XYZ row
    order (LAS order is acquisition order, not track progression).
    """
    v=np.diff(A,axis=0);vv=np.sum(v*v,axis=1);d=points[:,None,:]-A[:-1];t=np.clip(np.einsum('nki,ki->nk',d,v)/np.maximum(vv,1e-12),0,1)
    dist=np.sum((d-t[:,:,None]*v)**2,axis=2);ss=s[:-1]+t*np.diff(s)
    init=np.asarray(march_s,float);init=np.where(np.isfinite(init),init,ss[np.arange(len(points)),np.argmin(dist,axis=1)])
    permitted=abs(ss-init[:,None])<=8.;dist=np.where(permitted,dist,np.inf)
    no=~np.isfinite(dist).any(axis=1)
    if no.any():dist[no]=np.sum((d[no]-t[no,:,None]*v)**2,axis=2)
    out=ss[np.arange(len(points)),np.argmin(dist,axis=1)];corrections=0
    for src in np.unique(source):
        ids=np.flatnonzero(source==src);order=ids[np.argsort(init[ids],kind='stable')]
        corrected=np.maximum.accumulate(out[order]);corrections+=int(np.sum(corrected-out[order]>1e-6));out[order]=corrected
    return out,corrections

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

def factor_graph(obs_s,obs,obs_cov,query,frame,cfg):
    """States [x,y,heading,kappa,z,grade] with integration transition factors."""
    ns=grid(float(query[-1]),cfg.get('knots',4));n=len(ns);Y=obs@frame
    init=np.column_stack([np.interp(ns,obs_s,Y[:,j]) for j in range(3)]);d=np.gradient(init,ns,axis=0);psi=np.unwrap(np.arctan2(d[:,1],d[:,0]));kap=np.gradient(psi,ns);state=np.column_stack([init[:,:2],psi,kap,init[:,2],d[:,2]])
    W=np.zeros((len(obs_s),n));ix=np.clip(np.searchsorted(ns,obs_s,side='right')-1,0,n-2);t=(obs_s-ns[ix])/(ns[ix+1]-ns[ix]);W[np.arange(len(ix)),ix]=1-t;W[np.arange(len(ix)),ix+1]=t
    sig=np.sqrt(np.maximum(np.einsum('ia,nij,ja->na',frame,obs_cov,frame),1e-6));ds=np.diff(ns);loss=cfg.get('loss','huber');vp=cfg.get('vertical_prior',.015);w=np.ones(len(Y));history=[]
    def residual(flat):
        u=flat.reshape(n,6);loc=u[:,[0,1,4]];midpsi=(u[:-1,2]+u[1:,2])/2
        r=[(((W@loc-Y)/sig)*np.sqrt(w[:,None])).ravel(),(np.diff(u[:,0])-ds*np.cos(midpsi))/.01,(np.diff(u[:,1])-ds*np.sin(midpsi))/.01,(np.diff(u[:,2])-ds*(u[:-1,3]+u[1:,3])/2)/.002,(np.diff(u[:,4])-ds*(u[:-1,5]+u[1:,5])/2)/.01,np.diff(u[:,3])/np.sqrt(ds)/.002,np.diff(u[:,5])/np.sqrt(ds)/vp]
        if cfg.get('physics',True):r.extend([np.maximum(abs(u[:,3])-.01,0)/.001,u[:,3]/.025])
        return np.concatenate(r)
    # Numerical Jacobian with sparsity graph: measurements touch 2 nodes,
    # transitions touch 2 nodes, priors touch one node.
    rows=3*len(Y)+6*(n-1)+(2*n if cfg.get('physics',True) else 0);sp=lil_matrix((rows,n*6),dtype=int);row=0
    for a in ix:sp[row:row+3,a*6:(a+2)*6]=1;row+=3
    for block in range(6):
        for a in range(n-1):sp[row,a*6:(a+2)*6]=1;row+=1
    while row<rows:
        a=(row-(3*len(Y)+6*(n-1)))%n;sp[row,a*6:(a+1)*6]=1;row+=1
    for _ in range(3):
        fit=least_squares(residual,state.ravel(),jac_sparsity=sp.tocsr(),max_nfev=45,ftol=1e-5,xtol=1e-6,gtol=1e-5);state=fit.x.reshape(n,6);r=np.linalg.norm((W@state[:,[0,1,4]]-Y)/sig,axis=1);w=np.maximum(rho_weights(r,loss),1e-5);history.append(float(fit.cost))
    # Hermite positions use the solved heading/grade, not a generic XYZ spline.
    tang=np.column_stack((np.cos(state[:,2]),np.sin(state[:,2]),state[:,5]));C=CubicHermiteSpline(ns,state[:,[0,1,4]],tang)(query)@frame.T
    cov=np.repeat(np.mean(obs_cov,axis=0)[None],len(query),axis=0)
    return C,cov,dict(iterations=3,objective_history=history,solver_success=bool(fit.success),node_s=ns.tolist(),states=state.tolist(),weights=w.tolist())

def physics(s,C,frame,linear=None,vertical_prior=.015):
    Y=C@frame;v=np.gradient(Y,s,axis=0);a=np.gradient(v,s,axis=0);speed=np.maximum(np.linalg.norm(v[:,:2],axis=1),1e-8);heading=np.unwrap(np.arctan2(v[:,1],v[:,0]));k=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/speed**3;grade=v[:,2]/speed;vk=np.gradient(grade,s);kp=np.gradient(k,s);rad=1/np.maximum(abs(k),1e-12)
    jump={str(d):np.abs(np.interp(np.minimum(s+d,s[-1]),s,heading)-heading) for d in (1,2,4,8)};back=v[:,0]<0;cross=False
    # Nonadjacent 2D segment intersections (strict; collinear adjacency excluded).
    for i in range(len(Y)-3):
        for j in range(i+2,len(Y)-1):
            aa=Y[i,:2];bb=Y[i+1,:2];cc=Y[j,:2];dd=Y[j+1,:2]
            cr=lambda x,y:x[0]*y[1]-x[1]*y[0]
            if cr(bb-aa,cc-aa)*cr(bb-aa,dd-aa)<0 and cr(dd-cc,aa-cc)*cr(dd-cc,bb-cc)<0:cross=True
    flag=(abs(k)>.01)|(abs(vk)>3*vertical_prior)|back;score=np.maximum(abs(k)/.01-1,0)+np.maximum(abs(vk)/(3*vertical_prior)-1,0)+abs(kp)/.01+back*10
    over=np.linalg.norm(C-linear,axis=1) if linear is not None else np.zeros(len(C));summary=dict(radius_min=float(min(rad)),radius_p1=float(np.percentile(rad,1)),radius_p5=float(np.percentile(rad,5)),curvature_violation_fraction=float(np.mean(abs(k)>.01)),physical_violation_fraction=float(np.mean(flag)),max_grade=float(max(abs(grade))),max_vertical_curvature=float(max(abs(vk))),max_curvature_derivative=float(max(abs(kp))),backward_count=int(sum(back)),self_intersection=bool(cross),station_monotonic=bool(np.all(np.diff(s)>0)),max_tangent_change_deg=float(np.rad2deg(np.max(np.arccos(np.clip(np.sum(unit(v)[1:]*unit(v)[:-1],axis=1),-1,1))))),overshoot_p50=float(np.median(over)),overshoot_p95=float(np.percentile(over,95)),overshoot_max=float(max(over)),S_phys=float(np.mean(score)))
    for d in jump:summary['heading_jump_'+d+'m_deg']=float(np.rad2deg(max(jump[d])))
    return summary,dict(curvature_h=k,radius=rad,grade=grade,curvature_v=vk,curvature_derivative=kp,physical_flags=flag,physical_score=score,overshoot=over,heading=heading)
