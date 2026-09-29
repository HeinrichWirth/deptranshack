"""Evaluate each immutable completed start while later starts are being inferred."""
from long_common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from assess_long import one as assess
from observability import one as observe
from oracle_threshold import one as oracle
import subprocess

def one(r):
    run=r['run'];i=r['frame'];names=['B0','B2','B4','B8']+load(OUT/'research_freeze.json')['top3'];group='phase_e_benchmark'
    folders=[OUT/group/n/key(run,i) for n in names]
    while not all((p/'PREDICTION_COMPLETE.json').exists() for p in folders):time.sleep(3)
    for folder in folders:assess(str(folder))
    observe((run,i,group));oracle((run,i,group))
    return run,i

def main():
    cohort=load(OUT/'audit/cohort.json')['heldout'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(one,r) for r in cohort]
        for n,f in enumerate(as_completed(jobs),1):
            v=f.result()
            if n%20==0 or n==len(cohort):print('FINAL POSTPROCESS',n,len(cohort),v,round(time.perf_counter()-t,1),flush=True)
    for name,args in [('observability.py',['phase_e_benchmark','--workers','4']),('oracle_threshold.py',['phase_e_benchmark','--workers','4']),('collect_long.py',['phase_e_benchmark']),('artifact_data.py',[])]:
        print('BEGIN',name,flush=True)
        subprocess.run([sys.executable,'-B',str(STAGE/'src'/name)]+args,check=True)
    print('FINAL_TABLES_AND_EXAMPLE_SELECTION_COMPLETE',flush=True)

if __name__=='__main__':main()
