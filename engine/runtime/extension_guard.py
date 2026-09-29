"""Conservative evidence gate, never invents or changes a rail trajectory.

Only the contiguous extrapolated prefix with raw support on BOTH rail heads
may participate in obstacle confirmation. The remaining corridor is unknown.
Raw clouds are causal and contain no class labels or synthetic curve points.
"""
import time
import numpy as np

CONFIG=dict(section_m=2.,lateral_m=.055,vertical_m=.03,min_points=3,
            min_longitudinal_span_m=.5,min_cells=3,cell_m=.03)

def guard(pair,points,observed_end,config=None):
    t=time.perf_counter();cfg=dict(CONFIG,**(config or {}));pair=np.asarray(pair,float)
    center=pair.mean(1);u=-center[:,1];end=-np.asarray(observed_end).mean(0)[1]
    if np.any(np.diff(u)<=1e-6):raise ValueError('Nonmonotonic path cannot be validated')
    n=max(2,min(len(pair),int(np.searchsorted(u,end+1e-5,side='right'))))
    original=n;accepted=n;sections=[]
    if n<len(pair):
        # Cheap broad XYZ filter, then small independent head neighborhoods.
        p=np.asarray(points)[:,:3];pu=-p[:,1]
        keep=np.isfinite(p).all(1)&(pu>=u[n-1])&(pu<=u[-1])
        keep&=(p[:,0]>=pair[:,:,0].min()-.1)&(p[:,0]<=pair[:,:,0].max()+.1)
        keep&=(p[:,2]>=pair[:,:,2].min()-.05)&(p[:,2]<=pair[:,:,2].max()+.05)
        p=p[keep];pu=pu[keep]
        start=n-1
        while start<len(pair)-1:
            stop=min(len(pair)-1,int(np.searchsorted(u,u[start]+cfg['section_m'])))
            q=p[(pu>=u[start])&(pu<=u[stop])];qu=-q[:,1];head=[]
            for side in (0,1):
                x=np.interp(qu,u,pair[:,side,0]);z=np.interp(qu,u,pair[:,side,2])
                m=(abs(q[:,0]-x)<=cfg['lateral_m'])&(abs(q[:,2]-z)<=cfg['vertical_m']);hit=q[m]
                span=float(np.ptp(hit[:,1])) if len(hit) else 0.
                cells=len(np.unique(np.floor(hit/cfg['cell_m']).astype(np.int64),axis=0)) if len(hit) else 0
                head.append(dict(points=len(hit),span_m=span,cells=cells,accepted=len(hit)>=cfg['min_points'] and span>=cfg['min_longitudinal_span_m'] and cells>=cfg['min_cells']))
            ok=all(h['accepted'] for h in head);sections.append(dict(from_m=float(u[start]),to_m=float(u[stop]),heads=head,accepted=ok))
            if not ok:break
            accepted=stop+1;start=stop
    return pair[:accepted].copy(),dict(status='CLIPPED_UNSUPPORTED_EXTENSION' if accepted<len(pair) else 'SUPPORTED_OR_NO_EXTENSION',
        observed_vertices=original,accepted_vertices=accepted,total_vertices=len(pair),observed_end_m=float(u[original-1]),
        accepted_end_m=float(u[accepted-1]),previous_end_m=float(u[-1]),unknown_from_m=float(u[accepted-1]) if accepted<len(pair) else None,
        sections=sections,ms=(time.perf_counter()-t)*1000,config=cfg)
