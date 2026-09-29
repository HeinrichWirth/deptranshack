"""Sequential wall-clock benchmark, frozen C4 is read from disk, not rerun."""
from rr_common import *
from seed_input import build_input
from rail_models import predict,INPUT_KEYS
from run_inference import check_freeze
import ctypes
from ctypes import wintypes

def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak_wset','rss','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE;psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    result=Counters();result.cb=ctypes.sizeof(result)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(result),result.cb):raise ctypes.WinError(ctypes.get_last_error())
    return result

def main():
    lock=check_freeze();method=lock['best_development'];cfg=load(OUT/'configs.json')[method];prior=load(OUT/'models/global_prior.json')['default'];starts=load(OUT/'phase_e/per_start.json');rr=[r for r in starts if r['method']==method and r['status']=='AVAILABLE'];rr.sort(key=lambda r:r['max_predicted_range']);selected=[rr[k] for k in np.unique(np.linspace(0,len(rr)-1,12,dtype=int))];rows=[]
    for r in selected:
        run,i=r['run'],r['frame'];t=time.perf_counter();meta,a=build_input(run,i);prepare=(time.perf_counter()-t)*1000;times=[]
        for _ in range(10):
            t=time.perf_counter();p=predict(meta['seed'],{k:a[k] for k in INPUT_KEYS},meta['profiles'],meta['q'],cfg,prior);times.append((time.perf_counter()-t)*1000)
        mem=memory();cp,_=read_c4(run,i)
        rows.append(dict(run=run,frame=i,method=method,stations=len(p['s']),CR_points=len(a['cr_points']),seed_points=len(a['seed_points']),prepare_with_io_seed_profile_ms=prepare,geometry_median_ms=float(np.median(times)),geometry_p95_ms=float(np.quantile(times,.95)),stage6_total_ms=prepare+float(np.median(times)),rss_mb=mem.rss/2**20,peak_rss_mb=getattr(mem,'peak_wset',mem.rss)/2**20,C4_saved_original_ms=cp.get('total_ms'),C4_rerun=False,workers=1,geometry_repeats=10))
        print('RUNTIME',len(rows),len(selected),flush=True)
    write_csv(OUT/'runtime.csv',rows);save(OUT/'runtime/summary.json',dict(time_ns=time.time_ns(),sequential=True,starts=len(rows),geometry_ms=stats([r['geometry_median_ms'] for r in rows]),total_ms=stats([r['stage6_total_ms'] for r in rows]),peak_process_mb=max(r['peak_rss_mb'] for r in rows),C4_cost_excluded=True,scope='C4 file read + SHA, seed calibration and optional CR profile diagnostics + rail geometry. Profile diagnostics are computed even though selected model does not use them. No GT, report plotting or LAS export.',background_processes='Other project work may share CPU/disk; not a real-time hardware guarantee.'))

if __name__=='__main__':main()
