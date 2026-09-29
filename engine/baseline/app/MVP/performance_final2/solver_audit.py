from . import OUT
from .engine import Engine
from . import tracker_backend
from MVP.performance_final.profile_c4 import COHORT
import long_common as lc
import time

def main():
    rows=[];summary=[];original=tracker_backend.least_squares
    try:
        for run,index in COHORT:
            def measured(fun,x,**kw):
                counters=dict(fun_seconds=0.,jac_seconds=0.,fun_calls=0,jac_calls=0)
                def wrap(fn,key):
                    def invoke(*a,**k):
                        t=time.perf_counter();r=fn(*a,**k);counters[key+'_seconds']+=time.perf_counter()-t;counters[key+'_calls']+=1;return r
                    return invoke
                if callable(kw.get('jac')):kw['jac']=wrap(kw['jac'],'jac')
                t=time.perf_counter();r=original(wrap(fun,'fun'),x,**kw);wall=time.perf_counter()-t
                rows.append(dict(run=run,frame=index,solver_seconds=wall,orchestration_seconds=wall-counters['fun_seconds']-counters['jac_seconds'],nfev=r.nfev,njev=r.njev,parameters=len(x),**counters))
                return r
            tracker_backend.least_squares=measured;e=Engine('batch');e.start_run(lc.dataset()/run);r=e.process_frame(index);e.close()
            rr=[x for x in rows if x['run']==run];overhead=sum(x['orchestration_seconds'] for x in rr)
            summary.append(dict(run=run,frame=index,C4_seconds=r['summary']['timing']['T_C4_MARCHING'],full_seconds=r['summary']['timing']['T_TOTAL'],solver_overhead_seconds=overhead,
                fraction=overhead/r['summary']['timing']['T_C4_MARCHING']))
    finally:tracker_backend.least_squares=original
    lc.csv_write(OUT/'solver_audit.csv',rows);lc.save(OUT/'solver_budget.json',summary)

if __name__=='__main__':main()
