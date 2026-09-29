from . import OUT,ROOT
from MVP.performance_final.engine import Engine
from MVP.performance_final.ablation import fingerprint
from MVP.final_pipeline.benchmark import COHORT
from MVP.performance_final.run_full import RUNS
import long_common as lc
import numpy as np
import time

def main():
    cohort=[dict(run=r,frame=i,category=c) for r,start,c in COHORT for i in range(start,start+8)]
    final=[dict(run=r,frame=int(i)) for r in RUNS for i in np.unique(np.linspace(0,len(lc.frames(r))-1,40,dtype=int))]
    lc.save(OUT/'benchmark_cohort.json',dict(clean=cohort,final=final,selection='fixed before tests; uniformly spaced, no labels/quality'))
    for backend in ('python','optimized','spatial','native','native_spatial'):
        rows=[]
        for run,start,category in COHORT:
            e=Engine(backend);e.start_run(lc.dataset()/run)
            for i in range(start,start+8):
                result=e.process_frame(i);rows.append(dict(run=run,frame=i,hash=fingerprint(result),**result['summary']['timing']))
        prior=lc.load(ROOT/'results_performance_final/ablation'/f'{backend}.json')
        assert [r['hash'] for r in rows]==[r['hash'] for r in prior]
        lc.save(OUT/'baseline'/f'{backend}.json',rows)
        print('BASELINE',backend,round(np.median([r['T_TOTAL'] for r in rows])*1000,1),flush=True)
    lc.save(OUT/'baseline/COMPLETE.json',dict(time_ns=time.time_ns(),exact_to_previous=True))

if __name__=='__main__':main()
