"""Actual observation provenance, held-out visibility diagnostics, cold runtime."""
from lc_common import *
from latent_inference import read_causal,fit_profiles,infer_arrays
from curve_geometry import *
from concurrent.futures import ProcessPoolExecutor

def provenance_case(task):
    r,best=task;run,i=r['run'],r['frame'];m,a,p,pr=read_causal(run,i)
    if m['status']!='AVAILABLE':return None
    timestamp=np.empty(len(pr['xyz']),dtype=np.float64)
    for source in np.unique(pr['source_frame']):
        path=dataset()/run/frames(run)[int(source)]['file']
        with frozen.laspy.open(path) as f:h=f.header
        raw=np.memmap(path,mode='r',dtype=h.point_format.dtype(),offset=h.offset_to_point_data,shape=(h.point_count,));ids=np.flatnonzero(pr['source_frame']==source);timestamp[ids]=raw['sensor_timestamp'][pr['source_row'][ids]];del raw
    group='phase_d' if run in load(OUT/'protocol.json')['split']['development'] else 'phase_e';folder=OUT/group/best/key(run,i)
    with np.load(folder/'curve.npz') as z:cs=z['s'];C=z['C']
    station,ncorr=assign_stations(pr['xyz'],cs,C,pr['source_frame'],p['s_from_seed']);dest=OUT/'provenance'/key(run,i);dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'points.npz',**pr,sensor_timestamp=timestamp,source_header_time_ns=np.array([int(frames(run)[int(j)]['header_time_ns']) for j in pr['source_frame']],np.int64),march_step=p['point_step'],template_residual=p['template_residual'],confidence=p['confidence'],state=p['point_state'],assigned_station=station);save(dest/'PROVENANCE.json',dict(run=run,frame=i,method=best,points=len(timestamp),sources=len(np.unique(pr['source_frame'])),max_source=int(pr['source_frame'].max()),causal=bool(np.all(pr['source_frame']<=i)),sensor_timestamp='Copied unchanged from original LAS sensor_timestamp; not replaced by frame timestamp.',state={1:'TENTATIVE reduced weight',2:'CONFIRMED full weight',3:'GAP no raw measurement'},source_registration_quality='No per-source covariance available in frozen LAS metadata. Fit disagreement, range, ring diversity and pose status are proxies, not measured registration accuracy.',station_order_corrections=ncorr,sha256=sha(dest/'points.npz')));return dict(run=run,frame=i,points=len(timestamp),sources=len(np.unique(pr['source_frame'])),causal=True)

def visibility():
    protocol=load(OUT/'protocol.json');best=load(OUT/'research_freeze.json')['best'];rows=[]
    for r in protocol['development']:
        run,i=r['run'],r['frame'];folder=OUT/'phase_d'/best/key(run,i)
        if load(folder/'prediction.json')['status']!='AVAILABLE':continue
        path=ROOT/'results_contact_marching/future_gt'/(key(run,i)+'_reference_v2.npz')
        if not path.exists():continue
        with np.load(path) as z:points=z['future']
        with np.load(folder/'prediction.npz') as z:s=z['s'];C=z['C'];B=z['B']
        for station in np.arange(10,s[-1],10):
            j=int(np.argmin(abs(s-station)));rel=(points-C[j])@B[j];P=rel[(abs(rel[:,0])<.5)&(np.linalg.norm(rel[:,1:],axis=1)<.4),1:]
            if not len(P):mode='UNKNOWN';ext=[0,0]
            else:
                ext=np.ptp(P,axis=0);med=np.median(P,axis=0)
                if len(P)<4:mode='SPARSE_MULTI'
                elif ext[0]<.02 and ext[1]>.04:mode='VERTICAL_ONLY'
                elif ext[1]<.025:mode='TOP_ONLY' if med[1]>-.015 else 'BOTTOM_ONLY'
                elif ext[0]>.055 and ext[1]>.11:mode='FULL'
                elif min(ext)<.025:mode='ONE_FACE'
                else:mode='CORNER'
            rows.append(dict(run=run,frame=i,station=float(station),points=len(P),extent_v=float(ext[0]),extent_w=float(ext[1]),visibility_diagnostic=mode,GT_used=True,selection_allowed=False,definition='Geometric proxy from actual future class2 points after prediction; not human visibility GT.'))
    write_csv(OUT/'actual_class2_visibility.csv',rows)

def runtime():
    protocol=load(OUT/'protocol.json');cfg=next(c for c in protocol['candidates'] if c['name']==load(OUT/'research_freeze.json')['best']);rows=[]
    for r in protocol['development'][::10]:
        m,a,p,pr=read_causal(r['run'],r['frame'])
        if m['status']!='AVAILABLE':continue
        start=time.perf_counter();F=plan_frame(a['initial_basis'],a['world_up_proxy']);s=grid(a['s'][-1],2);C=interpolation(a['anchor_knots'],a['anchor_xyz'],s,F,'pchip',.25);B=curve_frames(s,C,a['initial_basis']);ps,_=assign_stations(pr['xyz'],a['anchor_knots'],a['anchor_xyz'],pr['source_frame'],p['s_from_seed']);ss,aa,cv,profiles,timing=fit_profiles(s,C,B,ps,pr,p,m['q'],cfg['profile']);profile_elapsed=(time.perf_counter()-start)*1000
        t=time.perf_counter();smooth_spline(ss,aa,cv,grid(a['s'][-1]),F,cfg);curve_ms=(time.perf_counter()-t)*1000;rows.append(dict(run=r['run'],frame=r['frame'],raw_points=len(pr['xyz']),stations=len(s),horizon=float(s[-1]),profile_likelihood_ms=timing['profile_ms'],multi_view_fusion_ms=timing['fusion_ms'],anchor_estimation_ms=profile_elapsed,curve_optimization_ms=curve_ms,total_ms=(time.perf_counter()-start)*1000,cold_profile_cache=True,excludes='Frozen C4 execution, original LAS I/O, downstream rail reconstruction; all input arrays already loaded.'))
    write_csv(OUT/'runtime_cold.csv',rows);save(OUT/'runtime_cold.json',dict(cases=rows,summary={field:stats([r[field] for r in rows]) for field in ('profile_likelihood_ms','multi_view_fusion_ms','anchor_estimation_ms','curve_optimization_ms','total_ms')}))

def main():
    freeze=load(OUT/'research_freeze.json');p=load(OUT/'protocol.json');tasks=[(r,freeze['best']) for r in p['development']+p['benchmark']];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,r in enumerate(pool.map(provenance_case,tasks),1):
            if r:rows.append(r)
            if j%200==0:print('PROVENANCE',j,len(tasks),flush=True)
    write_csv(OUT/'provenance/index.csv',rows);visibility();runtime();save(OUT/'audit/EXTRA_COMPLETE.json',dict(time_ns=time.time_ns(),provenance_cases=len(rows),actual_class2_visibility=True,cold_runtime=True));print('EXTRA COMPLETE')

if __name__=='__main__':main()
