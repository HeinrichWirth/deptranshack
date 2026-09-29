"""Sequential online simulation and already-assembled cloud timing, fresh processes."""
from fusion_common import *
from fusion import CausalSource,assemble
from study import predict_cloud
from bootstrap import production_seed,pose_record
from contact_rail_step2 import ContactRailDetector
import subprocess,ctypes,argparse
from ctypes import wintypes

def peak_rss():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(s,ctypes.c_size_t) for s in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters();c.cb=ctypes.sizeof(c);get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=wintypes.HANDLE
    func=ctypes.windll.psapi.GetProcessMemoryInfo;func.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not func(get(),ctypes.byref(c),c.cb):raise ctypes.WinError()
    return int(c.PeakWorkingSetSize)

def child(name,run,i):
    config=load(OUT/'configs.json')[name];start=time.perf_counter();src=CausalSource(run,i);current,timing=src.read(i)
    seed=production_seed(current['xyz'],pose_record(src.past[-1]),src.direction_next,load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json'),ContactRailDetector())
    original=src.read
    src.read=lambda k:(current,dict(timing,read_ms=0.,transform_ms=0.)) if k==i else original(k)
    cloud,seed=assemble(src,config,seed)
    # Frozen fusion_ms starts after concatenation. For the benchmark, measure
    # all assembly overhead (including concatenate/provenance), subtracting only
    # reads and transforms actually performed inside this assembly call.
    assembly_overhead=max(0.,cloud['meta']['assembly_ms']-cloud['meta']['read_ms']-cloud['meta']['transform_ms'])
    cloud['meta']['read_ms']+=timing['read_ms'];cloud['meta']['transform_ms']+=timing['transform_ms'];cloud['meta']['fusion_ms']=assembly_overhead
    pred=predict_cloud(cloud,seed,config);uncached=(time.perf_counter()-start)*1000
    expected=load_prediction(OUT/'heldout'/name/key(run,i))
    np.testing.assert_array_equal(pred['indices'],expected['indices'])
    assert pred['reason']==expected['reason']
    cached=[]
    for _ in range(2):
        start=time.perf_counter();same=predict_cloud(cloud,seed,config);cached.append((time.perf_counter()-start)*1000);np.testing.assert_array_equal(pred['indices'],same['indices'])
    rows=dict(variant=name,run=run,start_frame=i,uncached_start_ms=uncached,cached_march_ms=float(np.median(cached)),seed_ms=seed['seed_ms'],read_ms=cloud['meta']['read_ms'],read_history_ms=cloud['meta']['read_history_ms'],transform_ms=cloud['meta']['transform_ms'],fusion_ms=cloud['meta']['fusion_ms'],points=len(cloud['xyz']),actual_frames=cloud['meta']['actual_frames'],array_bytes=cloud['meta']['input_bytes'],peak_process_rss_bytes=peak_rss(),os_file_cache='warm/uncontrolled; application cloud cache bypassed')
    return dict(start=rows,steps=[dict(variant=name,run=run,start_frame=i,**{k:s.get(k) for k in ('step_index','status','pca_ms','bishop_ms','projection_ms','template_ms','total_ms')}) for s in pred['steps']])

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--child':print(json.dumps(clean(child(sys.argv[2],sys.argv[3],int(sys.argv[4])))));return
    lock=load(OUT/'research_lock.json');default_names=list(dict.fromkeys(['F0','F2','F3','F4',lock['selected'],lock['best_distance'],'F8']))
    ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+');ap.add_argument('--append',action='store_true');args=ap.parse_args();names=args.variants or default_names
    assert all(n in lock['heldout_variants'] for n in names)
    available=[r for r in load(OUT/'analysis_rows.json') if r['variant']=='F0' and r['seed_available']];chosen=[]
    for run in SPLIT['validation']:
        rr=sorted([r for r in available if r['run']==run],key=lambda r:r['start_frame']);chosen += [rr[j] for j in np.unique(np.linspace(0,len(rr)-1,6).astype(int))]
    result=[];steps=[]
    for name in names:
        for r in chosen:
            out=subprocess.run([sys.executable,'-B',__file__,'--child',name,r['run'],str(r['start_frame'])],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            value=json.loads(out.stdout);result.append(value['start']);steps.extend(value['steps']);print('BENCH',len(result),len(names)*len(chosen),name,round(value['start']['uncached_start_ms'],1),flush=True)
    old=csv_read(OUT/'runtime.csv') if args.append else [];old_steps=csv_read(OUT/'runtime/steps.csv') if args.append else []
    assert not set(names)&{r['variant'] for r in old},'Do not silently duplicate or replace benchmark runs'
    combined=old+result;csv_write(OUT/'runtime.csv',combined);csv_write(OUT/'runtime/steps.csv',old_steps+steps);csv_write(OUT/'memory.csv',[{k:r[k] for k in ('variant','run','start_frame','points','actual_frames','array_bytes','peak_process_rss_bytes')} for r in combined])
    summary={name:dict(starts=len(chosen),**{field:percentiles([r[field] for r in result if r['variant']==name]) for field in ('uncached_start_ms','cached_march_ms','seed_ms','read_ms','read_history_ms','transform_ms','fusion_ms','points','peak_process_rss_bytes')},per_step={field:percentiles([s[field] for s in steps if s['variant']==name and s.get(field) is not None]) for field in ('pca_ms','projection_ms','template_ms','total_ms')}) for name in names}
    previous=load(OUT/'runtime/summary.json') if args.append else {};previous.update(summary);save(OUT/'runtime/summary.json',previous)
if __name__=='__main__':main()
