import numpy as np
from geometry_math import rails,rail_covariance,cross_axes,wrapped,unit,offsets
INPUT_KEYS={'s','C','B','legacy_B','cr_state','support','position_sigma','tangent_sigma','world_up_proxy'}

def predict(seed,data,profiles,side,cfg,global_prior=None,curvature_prior=None):
    """Arguments are numeric CR/seed data only; dataset identity is deliberately absent."""
    if set(data)-INPUT_KEYS:raise ValueError('Only numeric CR geometry may enter rail prediction; raw clouds/labels are forbidden')
    s=data['s'];C=data['C'];B=data['B'];N=len(s);x0=np.array(seed['x0']);P0=np.array(seed['covariance'])
    if cfg['family']=='legacy':
        xx=[]
        for obs in seed['observations']:
            k=np.argmin(abs(s-obs['s']));xx.append(offsets(np.array(obs['cr']),data['legacy_B'][k],np.array(obs['heads']),side))
        x0=np.median(xx,axis=0)
    x=np.tile(x0,(N,1));P=np.tile(P0,(N,1,1))
    dist=np.maximum(s-float(seed['seed_length']),0);slope=np.asarray(seed['slope']);family=cfg['family'];modelB=B.copy()
    beta=np.array([p.get('beta',np.nan) for p in profiles]);beta=np.array([np.nan if b is None else b for b in beta],float)
    beta_sigma=np.array([p.get('sigma',np.deg2rad(10)) for p in profiles]);use=np.array([p.get('informative',False) for p in profiles],bool)&np.isfinite(beta)
    near=use&(s<=seed['seed_length']);beta0=float(np.median(beta[near])) if near.any() else np.nan
    alpha_measurement=beta+(x0[0]-beta0);q=np.deg2rad(cfg['roll_q_deg_sqrt_m'])**2
    P[:,0,0]+=dist*q;P[:,1:,1:]+=np.eye(4)[None]*dist[:,None,None]*cfg['offset_q_m_sqrt_m']**2
    if family=='legacy':modelB=data['legacy_B'];x[:,0]=x0[0]
    elif family in ('sensor_up','world_up'):
        up=np.array([0.,0.,1.]) if family=='sensor_up' else data['world_up_proxy'];t=B[:,:,0]
        normals=unit(up-t*(t@up)[:,None]);bb=unit(np.cross(normals,t));a=np.arctan2(np.einsum('ij,ij->i',bb,B[:,:,2]),np.einsum('ij,ij->i',bb,B[:,:,1]))
        a=np.unwrap(a);x[:,0]=a+(x0[0]-float(np.median(a[s<=seed['seed_length']])))
    elif family in ('smooth','spline'):
        L=cfg['damping_m'];v=slope[0]*cfg['slope_scale']
        # C2 join to the constant seed estimate: value and first two derivatives
        # match at the seed boundary. The 2 m onset is fixed before benchmark.
        elapsed=dist**3/(dist**2+4.)
        if family=='spline':
            # Cubic Hermite continuation, smoothly flattening its initial slope at L.
            z=np.minimum(elapsed/L,1);x[:,0]+=v*L*(z-z*z+z*z*z/3)
        else:x[:,0]+=v*L*(1-np.exp(-elapsed/L))
        slope_uncert=max(np.deg2rad(.005),np.sqrt(P0[0,0])/max(2,seed['seed_length']))
        P[:,0,0]+=(L*(1-np.exp(-dist/L))*slope_uncert)**2
    elif family in ('profile','profile_filter') and np.isfinite(beta0):
        state=x0[0];var=P0[0,0];prior_s=s[0]
        for k in range(N):
            var+=q*max(0,s[k]-prior_s);prior_s=s[k]
            if use[k]:
                R=beta_sigma[k]**2+P0[0,0]+np.deg2rad(.2)**2
                if family=='profile':state=alpha_measurement[k];var=R
                else:
                    innovation=float(wrapped(alpha_measurement[k]-state));gain=var/(var+R)
                    # A robust observation guard, fixed before development evaluation.
                    if abs(innovation)<=max(np.deg2rad(2),3*np.sqrt(var+R)):state+=gain*innovation;var*=1-gain
            x[k,0]=state;P[k,0,0]=var
    elif family=='frenet':
        dt=np.gradient(B[:,:,0],s,axis=0);norm=np.linalg.norm(dt,axis=1);valid=norm>1e-4
        a=np.arctan2(np.einsum('ij,ij->i',dt,B[:,:,2]),np.einsum('ij,ij->i',dt,B[:,:,1]));a=x0[0]+wrapped(a-x0[0]);x[valid,0]=a[valid]
        P[:,0,0]+=np.deg2rad(5)**2
    elif family=='curvature' and curvature_prior is not None:
        dt=np.gradient(B[:,:,0],s,axis=0);kh=np.einsum('ij,ij->i',dt,B[:,:,1]);x[:,0]+=curvature_prior['coef']*(kh-kh[0]);P[:,0,0]+=curvature_prior['residual_sigma']**2
    if cfg['offset_model']=='O2' and global_prior is not None:
        mean=np.array(global_prior['mean']);var=np.array(global_prior['variance']);w=np.clip(np.diag(P0)[1:]/(np.diag(P0)[1:]+var),0,.5)
        x[:,1:]=(1-w)*x[:,1:]+w*mean;P[:,1:,1:]*=(1-w)[None,:,None]*(1-w)[None,None,:]
    # Track gaps cannot retain the same certainty as observed CR support.
    gap_age=0.;gap_age_values=[]
    for k in range(N):
        gap_age=gap_age+((s[k]-s[k-1]) if k else 0) if data['cr_state'][k]!=2 else 0.
        gap_age_values.append(gap_age);P[k,0,0]+=np.deg2rad(.1)**2*gap_age
    pair=rails(C,modelB,x,side);cov=[];center_cov=[]
    for k in range(N):
        pc=np.eye(3)*(data['position_sigma'][k]+.004*gap_age_values[k])**2
        cc,JJ=rail_covariance(C[k],modelB[k],x[k],side,P[k],pc,data['tangent_sigma'][k]);cov.append(cc*cfg['corridor_scale']**2)
        bb,nn=cross_axes(modelB[k],x[k,0]);rr=pair[k].mean(axis=0)-C[k];T=np.column_stack((np.cross(bb,rr),np.cross(nn,rr)));J=JJ.mean(axis=0)
        center_cov.append((pc+J@P[k]@J.T+T@T.T*data['tangent_sigma'][k]**2)*cfg['corridor_scale']**2)
    cov=np.array(cov);b,n=cross_axes(modelB,x[:,0]);trans=np.stack((b,n),axis=2)
    cov2=np.einsum('nij,nrjk,nkl->nril',np.swapaxes(trans,1,2),cov,trans)
    return dict(s=s,C=C,B=modelB,pair=pair,center=pair.mean(axis=1),state=x,state_cov=P,rail_cov=cov,center_cov=np.array(center_cov),
        sigma_lateral=np.sqrt(np.maximum(cov2[:,:,0,0],0)),sigma_vertical=np.sqrt(np.maximum(cov2[:,:,1,1],0)),
        alpha_sigma=np.sqrt(np.maximum(P[:,0,0],0)),alpha_measurement=alpha_measurement,profile_informative=use,
        cr_state=data['cr_state'],cr_gap_age=np.array(gap_age_values),support=data['support'],q=np.array(side),predicted_geometry=np.ones((N,2),np.uint8))
