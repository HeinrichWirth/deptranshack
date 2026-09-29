"""Isolated single-cloud rail-pair detector. No LAS, labels, run ID or GT imports.

Input is ONLY current-frame (v,w) points and fixed development configuration.
The three-dimensional motion basis is a separate geometric operation.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from time import perf_counter


DEFAULT = dict(bin_m=.02, kernel_m=.04, gauge_m=1.56, height_m=-1.,
               vmin=-1.4, vmax=1.4, wmin=-1.35, wmax=-.65,
               center_max=.30, gauge_soft=[1.48,1.62], cant_max=.18,
               min_support=6, confidence_threshold=0., head_quantile=.75)


def basis(direction):
    f = np.asarray(direction, dtype=float)
    n = np.linalg.norm(f)
    if not np.isfinite(n) or n < .01:
        raise ValueError('stationary_or_invalid_direction')
    f = f/n
    up = np.array([0.,0.,1.]); up -= np.dot(up,f)*f
    if np.linalg.norm(up) < .1:
        raise ValueError('near_vertical_direction')
    up /= np.linalg.norm(up)
    lateral = np.cross(up,f); lateral /= np.linalg.norm(lateral)
    up = np.cross(f,lateral)
    return np.column_stack((f,lateral,up))


def density(points, cfg):
    b=cfg['bin_m']
    ve=np.arange(cfg['vmin'], cfg['vmax']+b*.5,b)
    we=np.arange(cfg['wmin'], cfg['wmax']+b*.5,b)
    hist=np.histogram2d(points[:,0],points[:,1],bins=(ve,we))[0]
    smoothed=gaussian_filter(hist,cfg['kernel_m']/b,mode='constant')
    return hist, smoothed, (ve[:-1]+ve[1:])/2, (we[:-1]+we[1:])/2


def candidates(points, cfg, representation):
    start=perf_counter()
    p=points[(points[:,0]>=cfg['vmin'])&(points[:,0]<=cfg['vmax'])&
             (points[:,1]>=cfg['wmin'])&(points[:,1]<=cfg['wmax'])]
    if len(p)<2: return [], [], 0., 0.
    tree=cKDTree(p)
    if representation=='density':
        _, sm, v, w=density(p,cfg)
        ix=np.argwhere((sm==maximum_filter(sm,size=3,mode='constant'))&(sm>0))
        centers=np.column_stack((v[ix[:,0]],w[ix[:,1]])); strength=sm[tuple(ix.T)]
    else:
        # Each proposal is an actual raw point. Quantization only removes nearby
        # redundant proposals, never replaces the cloud used for kernel support.
        _, ix=np.unique(np.floor(p/.015).astype(np.int32),axis=0,return_index=True)
        centers=p[ix]
        ds,_=tree.query(centers,k=min(96,len(p)),distance_upper_bound=3*cfg['kernel_m'])
        if ds.ndim==1: ds=ds[:,None]
        strength=np.exp(-.5*(ds/cfg['kernel_m'])**2).sum(axis=1)
    density_ms=(perf_counter()-start)*1000; start=perf_counter()
    sides=[]
    for sign in (-1,1):
        select=np.flatnonzero((sign*centers[:,0]>.32)&(sign*centers[:,0]<1.2))
        order=select[np.argsort(strength[select])[::-1]]
        peaks=[]
        for i in order:
            c=centers[i]
            if any(np.linalg.norm(c-x['mode'])<.065 for x in peaks): continue
            idx=tree.query_ball_point(c,.085)
            near=p[idx]
            if len(near)<cfg['min_support']: continue
            d=np.linalg.norm(near-c,axis=1)
            support=float(np.exp(-.5*(d/cfg['kernel_m'])**2).sum())
            # Local upper surface estimate has the SAME fixed convention for all
            # point-supported variants; labels never select a neighbourhood.
            center=np.array([np.median(near[:,0]),np.quantile(near[:,1],cfg['head_quantile'])])
            cov=np.cov(near.T) if len(near)>2 else np.eye(2)*.01
            ev=np.linalg.eigvalsh(cov)
            peaks.append(dict(mode=c,center=center,support=support,n=len(near),
                              extent=np.ptp(near,axis=0),pca=ev,
                              scatter=float(np.sqrt(max(ev[-1],0)))))
            if len(peaks)>=14: break
        sides.append(peaks)
    return *sides, density_ms, (perf_counter()-start)*1000


def detect(points, cfg=None, variant='proposed', representation='density'):
    cfg=DEFAULT if cfg is None else cfg
    start=perf_counter()
    if variant=='fixed':
        pair=np.array([[-cfg['gauge_m']/2,cfg['height_m']],[cfg['gauge_m']/2,cfg['height_m']]])
        return dict(pair=pair,confidence=1.,status='ok',reason='',density_ms=0.,search_ms=0.,detector_ms=0.)
    left,right,density_ms,search_ms=candidates(np.asarray(points),cfg,representation)
    result=dict(pair=None,confidence=0.,status='refusal',reason='insufficient_bilateral_support',
                density_ms=density_ms,search_ms=search_ms,detector_ms=(perf_counter()-start)*1000)
    if not left or not right: return result
    options=[]
    for l in left:
        for r in right:
            pair=np.array([l['center'],r['center']]); g=pair[1,0]-pair[0,0]
            c=pair[:,0].mean(); h=pair[:,1].mean(); dz=pair[1,1]-pair[0,1]
            if variant!='independent' and (abs(c)>cfg['center_max'] or not 1.30<=g<=1.80 or abs(dz)>cfg['cant_max']): continue
            a,b=l['support'],r['support']
            score=np.log1p(a)+np.log1p(b)
            if variant!='independent':
                outside=max(cfg['gauge_soft'][0]-g,0,g-cfg['gauge_soft'][1])
                score += .5*min(a,b)/max(a,b)
                score -= .12*((g-cfg['gauge_m'])/.07)**2 + 2*(outside/.07)**2
                score -= .12*(c/.30)**2 + .08*((h-cfg['height_m'])/.35)**2 + .10*(dz/.18)**2
            if variant=='proposed': score -= .25*(l['scatter']+r['scatter'])/.05
            options.append((float(score),pair,l,r))
    if not options:
        result['reason']='no_supported_geometrically_compatible_pair'; return result
    options.sort(key=lambda x:x[0],reverse=True)
    score,pair,l,r=options[0]
    alt=next((o[0] for o in options[1:] if np.max(np.linalg.norm(o[1]-pair,axis=1))>.06),score-3)
    margin=max(score-alt,0)
    support=min(l['support'],r['support']); balance=support/max(l['support'],r['support'])
    confidence=float((1-np.exp(-support/20))*(.3+.7*(1-np.exp(-margin)))*np.sqrt(balance))
    result.update(pair=pair,confidence=confidence,status='ok',reason='',margin=margin,
                  support_left=l['support'],support_right=r['support'],
                  width_left=float(l['extent'][0]),width_right=float(r['extent'][0]),
                  height_left=float(l['extent'][1]),height_right=float(r['extent'][1]))
    if confidence<cfg['confidence_threshold']:
        result.update(status='refusal',reason='ambiguous_or_weak_pair')
    result['detector_ms']=(perf_counter()-start)*1000
    result['search_ms']=result['detector_ms']-density_ms
    return result
