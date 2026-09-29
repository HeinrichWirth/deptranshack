"""Pure cross-section geometry: no labels, files, trajectories or dataset identity."""
import numpy as np
from scipy.interpolate import CubicSpline

def unit(v):
    v=np.asarray(v,dtype=float);norm=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(norm<1e-10):raise ValueError('Degenerate direction')
    return v/norm

def transport(B,t):
    t=unit(t);a=B[:,0];c=float(np.clip(a@t,-1,1))
    if c<-.999999:raise ValueError('Tangent reversal')
    v=np.cross(a,t);K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    Q=np.eye(3)+K+K@K/(1+c);b=Q@B[:,1];b=unit(b-t*(t@b))
    return np.column_stack((t,b,np.cross(t,b)))

def bishop(tangents,initial):
    result=[];B=np.asarray(initial,dtype=float)
    for t in tangents:B=transport(B,t);result.append(B.copy())
    return np.array(result)

def cross_axes(B,alpha):
    c=np.cos(alpha)[...,None];s=np.sin(alpha)[...,None]
    return c*B[...,1]+s*B[...,2],-s*B[...,1]+c*B[...,2]

def rails(C,B,x,q):
    """x=[alpha,d,h_near,g,h_far]; columns of B are tangent/e1/e2."""
    x=np.asarray(x);b,n=cross_axes(B,x[...,0]);d,h,g,hf=[x[...,i,None] for i in range(1,5)]
    near=C-q*d*b-h*n;far=C-q*(d+g)*b-hf*n
    return np.stack((near,far),axis=-2)

def offsets(C,B,pair,q):
    near,far=pair
    b=unit(q*(near-far));b=unit(b-B[:,0]*(b@B[:,0]));n=np.cross(B[:,0],b)
    alpha=np.arctan2(b@B[:,2],b@B[:,1]);v=C-near;vf=C-far
    return np.array([alpha,q*(v@b),v@n,q*((near-far)@b),vf@n])

def curve_model(anchors,initial,spacing=1.):
    """Monotone common station along CR anchor polyline; never assigns equal rail arc lengths."""
    a=np.asarray(anchors,float)
    if len(a)<2:return None
    keep=np.r_[True,np.linalg.norm(np.diff(a,axis=0),axis=1)>.05];a=a[keep]
    if len(a)<2:return None
    knots=np.r_[0,np.cumsum(np.linalg.norm(np.diff(a,axis=0),axis=1))]
    curve=CubicSpline(knots,a,axis=0,bc_type='natural',extrapolate=False)
    station=np.unique(np.r_[np.arange(0,knots[-1],spacing),knots[-1]])
    # C2 interpolation passes through EVERY unchanged C4 anchor. Fine transport
    # reduces discretization of the rotation-minimizing frame for offset curves.
    fine=np.unique(np.r_[np.arange(0,knots[-1],.25),station]);fineB=bishop(unit(curve.derivative()(fine)),initial)
    C=curve(station);B=fineB[np.searchsorted(fine,station)]
    return dict(s=station,C=C,B=B,knots=knots,anchors=a)

def sample_curve(model,s):
    return CubicSpline(model['knots'],model['anchors'],axis=0,bc_type='natural',extrapolate=False)(s)

def rail_covariance(C,B,x,q,Px,Pc,tangent_sigma):
    """Analytic first-order J P J^T, including CR position and two tangent errors."""
    b,n=cross_axes(B,x[0]);out=[];JJ=[]
    for isfar in (False,True):
        d=x[1]+(x[3] if isfar else 0);h=x[4] if isfar else x[2]
        r=-q*d*b-h*n;J=np.zeros((3,5));J[:,0]=-q*d*n+h*b;J[:,1]=-q*b
        J[:,4 if isfar else 2]=-n
        if isfar:J[:,3]=-q*b
        T=np.column_stack((np.cross(b,r),np.cross(n,r)))
        out.append(Pc+J@Px@J.T+T@T.T*tangent_sigma**2);JJ.append(J)
    return np.array(out),np.array(JJ)

def wrapped(a):return (np.asarray(a)+np.pi/2)%np.pi-np.pi/2
