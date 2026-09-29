"""Pooled physics and all missing common planes, including old error there."""
from lc_common import *

def main():
    best=load(OUT/'research_freeze.json')['best'];physics=[];losses=[];station_audit=[]
    for group in ('phase_d','phase_e'):
        starts=load(OUT/group/'per_start.json')
        for name in ['C0_CURRENT']+load(OUT/'research_freeze.json')['selected']:
            n=0;bad=0;allbad=0
            for r in starts:
                if r['method']!=name or r['status']!='AVAILABLE':continue
                folder=OUT/group/name/key(r['run'],r['frame'])
                with np.load(folder/'curve.npz') as z:n+=len(z['s']);bad+=int(sum(abs(z['curvature_h'])>.01));allbad+=int(sum(z['physical_flags']))
                if name!=best:continue
                with np.load(folder/'evaluation.npz') as z:missing=np.flatnonzero(~z['alignment_available']);s=z['station']
                oldgroup='phase_b' if group=='phase_d' else 'phase_e'
                with np.load(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/key(r['run'],r['frame'])/'evaluation.npz') as z:
                    for j in missing:losses.append(dict(group=group,run=r['run'],frame=r['frame'],station=float(s[j]),old_GT_available=bool(z['valid'][j]),old_far_error=float(z['error'][j,1]) if z['valid'][j] else None,after_seed=bool(s[j]>=8),reason='New curve does not intersect this old plane inside its saved horizon; not silently extrapolated.'))
                # Endpoint assignment audit: points are projected to the finite
                # curve. Document longitudinal residuals excluded by a 2D slab
                # approximation rather than claiming a full 3D likelihood.
                info=load(folder/'details.json');profile=info.get('profiles',[])
                if profile:
                    p,pr=read_c4(r['run'],r['frame'])
                    with np.load(folder/'prediction.npz') as z:C=z['C'];B=z['B'];ss=z['s']
                    for prof in (profile[0],profile[-1]):
                        ids=np.array(prof['point_ids'],int)
                        if not len(ids):continue
                        j=int(np.argmin(abs(ss-prof['station'])));longitudinal=abs((pr['xyz'][ids]-C[j])@B[j,:,0]);station_audit.append(dict(group=group,run=r['run'],frame=r['frame'],station=prof['station'],point_count=len(ids),longitudinal_residual_p95=float(np.percentile(longitudinal,95)),outside_2m=int(sum(longitudinal>2)),note='Finite-chain endpoint projection audit; profile factor is transverse 2D, not full 3D point likelihood.'))
            physics.append(dict(group=group,method=name,stations=n,radius_violation_stations=bad,radius_violation_fraction=bad/max(n,1),physical_flag_stations=allbad,physical_flag_fraction=allbad/max(n,1)))
    write_csv(OUT/'physics_pooled_summary.csv',physics);save(OUT/'physics_pooled_summary.json',physics);write_csv(OUT/'alignment_losses.csv',losses);write_csv(OUT/'station_endpoint_audit.csv',station_audit);save(OUT/'geometry_audit.json',dict(physics=physics,missing_common_planes=len(losses),missing_after_seed=sum(r['after_seed'] for r in losses),missing_after30=sum(r['station']>=30 for r in losses),missing_old_evaluable=sum(r['old_GT_available'] for r in losses),endpoint_profile_windows=len(station_audit),endpoint_windows_with_points_outside_2m=sum(r['outside_2m']>0 for r in station_audit),production_refit=False));print('GEOMETRY AUDIT',len(losses),'missing planes;',len(station_audit),'endpoint profiles')

if __name__=='__main__':main()
