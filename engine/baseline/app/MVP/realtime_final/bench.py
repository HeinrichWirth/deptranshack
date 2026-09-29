from . import ROOT,OUT
from .memory_engine import MemoryEngine,SharedRing
from MVP.c4_v2.native_engine import NativeEngine
import long_common as lc
import numpy as np
import time,argparse,hashlib,platform,os,cProfile

def delta(a,b):
    return {k:(delta(v,b.get(k,{})) if isinstance(v,dict) else v-b.get(k,0) if isinstance(v,(int,float)) else v) for k,v in a.items()}
def run(e,index):
    profiler=cProfile.Profile() if os.environ.get('COPY_CPROFILE') and not getattr(e,'copy_profiled',False) and index>=50 else None
    before=e.marcher.stats();t=time.perf_counter();cpu=time.process_time();thread_cpu=time.thread_time()
    if profiler:profiler.enable()
    r=e.process_frame(index)
    if profiler:
        profiler.disable();profiler.dump_stats(os.environ['COPY_CPROFILE']+f'_{index}.prof');e.copy_profiled=True
    dt=time.perf_counter()-t
    row=dict(r['summary'],wall_ms=dt*1000,cpu_seconds=time.process_time()-cpu,counters=delta(e.marcher.stats(),before),step_count=len(r['c4'].get('steps',[])))
    row['worker_thread_cpu_seconds']=time.thread_time()-thread_cpu
    row['active_cores']=row['cpu_seconds']/dt;row['c4_ms']=r['summary']['timing']['T_C4_MARCHING_INCLUSIVE_HISTORY_IO']*1000
    row['history_raw_points']=sum(len(x.world) for x in e.cache.frames.values());row['materialized_history_points']=r['cloud']['meta'].get('historical_roi_points',0)
    assert r['summary']['physical_reads_total']==0 if isinstance(e,MemoryEngine) else True
    return r,row
def same(a,b):
    ca,cb=a['c4'],b['c4'];eq=lambda x,y:np.array_equal(x,y,equal_nan=True)
    checks=dict(ids=eq(a['cloud']['keys'][ca['indices']],b['cloud']['keys'][cb['indices']]),reason=ca['reason']==cb['reason'])
    for k in ('curve','local_uvw','point_state','point_anchor','template_residual'):
        checks[k]=eq(ca.get(k,[]),cb.get(k,[]))
    checks['provenance']=np.array_equal(a['provenance'],b['provenance'])
    pa,pb=a['prediction'],b['prediction'];checks['availability']=(pa is None)==(pb is None)
    checks['final_exact']=pa is None and pb is None or pa is not None and pb is not None and all(eq(pa[k],pb[k]) for k in ('C','B','pair','s','q'))
    return {**checks,'c4_exact':all(v for k,v in checks.items() if k not in ('provenance','final_exact','availability'))}
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','ablation','development','profile']);a=p.parse_args()
    cohort=lc.load(ROOT/'results_performance_final2/benchmark_cohort.json');selection=lc.load(OUT/'ORIGINAL_SELECTION_LOCK.json')
    cases=cohort['final'] if a.action=='development' else cohort['clean']
    assert set(c['run'] for c in cases)<=set(selection['development'])
    if a.action=='smoke':cases=cases[:3]
    variants={'memory_broad':(False,False),'memory_selective':(False,True),'memory_exact':(True,True)} if a.action=='ablation' else {'memory_exact':(True,True)}
    rows=[];arrivals=[];proof=[]
    for name in dict.fromkeys(c['run'] for c in cases):
        directory=lc.dataset()/name;old=NativeEngine(8,12,0);old.start_run(directory)
        es={k:MemoryEngine(8,*v,profile=a.action=='profile') for k,v in variants.items()}
        for e in es.values():e.start_run(directory)
        ring=SharedRing(old.records)
        for case in (c for c in cases if c['run']==name):
            index=case['frame'];io=ring.prepare_from_files(directory,index)
            base,br=run(old,index);br['variant']='offline_original';rows.append(br)
            for variant,e in es.items():
                e.prepare(ring,index);r,row=run(e,index);checks=same(base,r);row.update(variant=variant,**checks);rows.append(row);proof.append(dict(run=name,frame=index,variant=variant,**checks))
                print(a.action,name,index,variant,round(row['wall_ms'],1),checks,flush=True)
            # Persist progress, not only a final success marker.
            lc.save(OUT/(a.action+'_progress.json'),dict(rows=rows,equality=proof))
        arrivals.extend(dict(run=name,**r) for r in ring.arrivals)
    lc.save(OUT/(a.action+'.json'),dict(rows=rows,arrivals=arrivals,equality=proof,platform=platform.platform(),time_ns=time.time_ns()))
    lc.csv_write(OUT/(a.action+'_equality.csv'),proof)
    if not all(r['c4_exact'] and r['availability'] and r['final_exact'] for r in proof):raise RuntimeError('EXACT_OPTIMIZATION_CHANGED_OUTPUT')
    print(a.action.upper()+'_PASS',len(proof),flush=True)
if __name__=='__main__':main()
