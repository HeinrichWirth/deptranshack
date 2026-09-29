"""Sequential fresh-process timings after all parallel research jobs finish."""
from long_common import *
from fusion import CausalSource,assemble
from roi_source import RoiSource,point_keys
from long_tracker import infer
from study import predict_cloud
from bootstrap import production_seed,pose_record
from contact_rail_step2 import ContactRailDetector
import subprocess,argparse,ctypes,inspect
from ctypes import wintypes

def peak_rss():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(s,ctypes.c_size_t) for s in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters();c.cb=ctypes.sizeof(c);get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=wintypes.HANDLE
    func=ctypes.windll.psapi.GetProcessMemoryInfo;func.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not func(get(),ctypes.byref(c),c.cb):raise ctypes.WinError()
    return int(c.PeakWorkingSetSize)

def profile_steps(call,baseline):
    # Separate cached diagnostic pass. Full-start timings above remain uninstrumented.
    from tracker import _engine
    fn=_engine if baseline else infer
    lines,first=inspect.getsourcelines(fn)
    loop=next(first+j for j,line in enumerate(lines) if "for k in range(1,cfg['max_steps']+1):" in line)
    stoptext="if steps and steps[-1]['status']!='ACCEPTED':" if baseline else 'if not finished:finished=active or [path]'
    stop=next(first+j for j,line in enumerate(lines) if stoptext in line)
    timings=[];active=None;current=None
    def trace(frame,event,arg):
        nonlocal active,current
        if frame.f_code is not fn.__code__:return None
        if event=='line' and frame.f_lineno in (loop,stop):
            now=time.perf_counter()
            if active is not None and current==frame.f_locals.get('k'):
                timings.append(dict(step=current,ms=(now-active)*1000))
            if frame.f_lineno==loop:active=now;current=frame.f_locals.get('k',0)+1
            else:active=None
        return trace
    old=sys.gettrace();start=time.perf_counter()
    try:sys.settrace(trace);value=call()
    finally:sys.settrace(old)
    return value,timings,(time.perf_counter()-start)*1000

def child(name,run,i,group='phase_e_benchmark'):
    cfg=load(OUT/'configs.json')[name];det=ContactRailDetector();t=time.perf_counter()
    if cfg.get('baseline'):
        src=CausalSource(run,i);current,timing=src.read(i)
        seed=production_seed(current['xyz'],pose_record(src.past[-1]),src.direction_next,load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json'),det)
        original=src.read;src.read=lambda k:(current,dict(timing,read_ms=0.,transform_ms=0.)) if k==i else original(k)
        c,_=assemble(src,dict(frames=cfg['frames']),seed);p=predict_cloud(c,seed,dict(frames=cfg['frames']));c['keys']=point_keys(c)
        read_ms=c['meta']['read_ms']+timing['read_ms'];transform_ms=c['meta']['transform_ms']+timing['transform_ms']
        points_processed=None
    else:
        source=RoiSource(run,i);src=source.source
        seed=production_seed(source.current['xyz'],pose_record(src.past[-1]),src.direction_next,load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json'),det)
        p,c=infer(source,seed,det.template,cfg)
        read_ms=sum(x['read_ms'] for x in c['meta']['io_details']);transform_ms=sum(x['transform_ms'] for x in c['meta']['io_details']);points_processed=source.processed
    elapsed=(time.perf_counter()-t)*1000
    expected=read_prediction(OUT/group/name/key(run,i))
    with np.load(OUT/group/name/key(run,i)/'provenance.npz') as z:expectedkeys=(z['source_frame'].astype(np.uint64)<<np.uint64(32))|z['source_row'].astype(np.uint64)
    np.testing.assert_array_equal(c['keys'][p['indices']],expectedkeys);assert p['reason']==expected['reason']
    if not cfg.get('baseline'):np.testing.assert_array_equal(p['point_state'],expected['point_state'])
    cached=[]
    for _ in range(2):
        start=time.perf_counter()
        if cfg.get('baseline'):q=predict_cloud(c,seed,dict(frames=cfg['frames']))
        else:q,_=infer(source,seed,det.template,cfg)
        cached.append((time.perf_counter()-start)*1000);np.testing.assert_array_equal(q['indices'],p['indices'])
    call=(lambda:predict_cloud(c,seed,dict(frames=cfg['frames']))) if cfg.get('baseline') else (lambda:infer(source,seed,det.template,cfg)[0])
    profiled,full_steps,profile_ms=profile_steps(call,bool(cfg.get('baseline')));np.testing.assert_array_equal(profiled['indices'],p['indices'])
    row=dict(variant=name,run=run,start_frame=i,start_ms=elapsed,cached_march_ms=float(np.median(cached)),instrumented_march_ms=profile_ms,instrumentation_ratio=profile_ms/max(.001,float(np.median(cached))),seed_ms=seed['seed_ms'],read_ms=read_ms,transform_ms=transform_ms,
      points=len(c['xyz']),historical_roi_points=c['meta'].get('historical_roi_points'),points_projected_in_roi=points_processed,
      actual_frames=len(c['meta']['source_frames']),peak_rss_bytes=peak_rss(),os_file_cache='warm/uncontrolled; new process; interpreter startup excluded')
    return dict(start=row,steps=[dict(variant=name,run=run,start_frame=i,**s,scope='full spatial iteration including all beam parents; separate instrumented cached pass') for s in full_steps],selected_path_steps=[dict(variant=name,run=run,start_frame=i,step=s['step_index'],status=s['status'],ms=s.get('total_ms')) for s in p['steps']])

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--child':print(json.dumps(clean(child(sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5] if len(sys.argv)>5 else 'phase_e_benchmark'))));return
    lock=load(OUT/'research_freeze.json');names=['B0','B2','B4','B8']+lock['top3'];allrows=load(OUT/'phase_e_benchmark/analysis_rows.json');chosen=[]
    for run in SPLIT['validation']:
        rr=sorted([r for r in allrows if r['variant']=='B0' and r.get('seed_available') and r['run']==run],key=lambda r:r['start_frame'])
        chosen += [rr[j] for j in np.unique(np.linspace(0,len(rr)-1,6).astype(int))]
    result=[];steps=[];path_steps=[]
    for name in names:
        for r in chosen:
            out=subprocess.run([sys.executable,'-B',__file__,'--child',name,r['run'],str(r['start_frame'])],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            v=json.loads(out.stdout);result.append(v['start']);steps+=v['steps'];path_steps+=v['selected_path_steps'];print('RUNTIME',len(result),len(names)*len(chosen),name,round(v['start']['start_ms']),flush=True)
    csv_write(OUT/'runtime.csv',result);csv_write(OUT/'runtime/steps.csv',steps);csv_write(OUT/'memory.csv',[{k:r[k] for k in ('variant','run','start_frame','peak_rss_bytes','points','actual_frames')} for r in result])
    csv_write(OUT/'runtime/selected_path_steps.csv',path_steps)
    summary={n:dict(starts=len(chosen),**{f:percentiles([r[f] for r in result if r['variant']==n]) for f in ('start_ms','cached_march_ms','read_ms','transform_ms','seed_ms','peak_rss_bytes','points','instrumentation_ratio')},step_ms=percentiles([r['ms'] for r in steps if r['variant']==n and r['ms'] is not None])) for n in names}
    save(OUT/'runtime/summary.json',summary)
    screen=load(OUT/'audit/screen.json');chosen=[screen[j] for j in np.unique(np.linspace(0,len(screen)-1,12).astype(int))];roi=[]
    for name in ('B8','AF8','AF16'):
        for r in chosen:
            out=subprocess.run([sys.executable,'-B',__file__,'--child',name,r['run'],str(r['frame']),'phase_a_screen'],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            v=json.loads(out.stdout);roi.append(v['start']);print('ROI_RUNTIME',len(roi),36,name,round(v['start']['start_ms']),flush=True)
    csv_write(OUT/'runtime/roi_comparison.csv',roi)
    save(OUT/'runtime/roi_summary.json',{n:dict(cohort='12 fixed development starts; same starts for all three',**{f:percentiles([r[f] for r in roi if r['variant']==n]) for f in ('start_ms','cached_march_ms','peak_rss_bytes','points')}) for n in ('B8','AF8','AF16')})

if __name__=='__main__':main()
