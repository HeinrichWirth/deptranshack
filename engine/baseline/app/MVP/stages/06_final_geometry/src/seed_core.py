import numpy as np
from geometry_math import offsets

def seed_rebase(seed,s,C,B,q,initial):
    """Same frozen robust estimator, applied to the SAME provided head sections.

    Only the reference curve/plane changes. No fresh label or raw rail search.
    """
    obs=[];u=C@initial[:,0]
    for old in seed['observations']:
        heads=np.asarray(old['heads']).copy();long=float(np.mean(heads@initial[:,0]));station=float(np.interp(long,u,s));cr=np.array([np.interp(station,s,C[:,j]) for j in range(3)]);bb=B[int(np.argmin(abs(s-station)))];heads-=np.outer((heads-cr)@bb[:,0],bb[:,0]);x=offsets(cr,bb,heads,q);obs.append(dict(s=station,x=x,heads=heads,cr=cr,support=old['support']))
    xx=np.array([o['x'] for o in obs]);ss=np.array([o['s'] for o in obs]);xx[:,0]=np.unwrap(xx[:,0]);med=np.median(xx,axis=0);mad=1.4826*np.median(abs(xx-med),axis=0);P=np.cov(xx.T)/max(1,len(xx)/2)+np.diag(np.array([np.deg2rad(.1),.004,.004,.006,.004])**2);slope=np.polyfit(ss-ss.mean(),xx,1)[0]
    return dict(seed,x0=med,covariance=P,mad=mad,slope=slope,observations=obs,reference_s=float(np.median(ss)))
