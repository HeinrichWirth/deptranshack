from lc_common import *
from latent_inference import run_case,producer_hash
from concurrent.futures import ProcessPoolExecutor
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['phase_a','phase_b','phase_d','phase_e']);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--names',nargs='*');ap.add_argument('--limit',type=int);a=ap.parse_args();require_reproduction();check_baseline();protocol=load(OUT/'protocol.json');configs=protocol['candidates']
    if a.phase=='phase_a':cohort=protocol['phase_a']
    elif a.phase=='phase_b':cohort=protocol['development'];allowed=load(OUT/'selection/phase_a.json')['selected'];configs=[c for c in configs if c['name'] in allowed]
    else:
        cohort=protocol['benchmark'] if a.phase=='phase_e' else protocol['development'];freeze=load(OUT/'research_freeze.json')
        assert freeze['producer']==producer_hash();configs=freeze['configs']
    if a.names:configs=[c for c in configs if c['name'] in a.names]
    if a.limit:cohort=cohort[:a.limit]
    group=a.phase if not a.limit else a.phase+'_pilot';tasks=[(r,group,configs) for r in cohort];start=time.perf_counter();predictions=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for j,rows in enumerate(pool.map(run_case,tasks),1):
            predictions+=rows;print('INFERENCE',group,j,len(tasks),round(time.perf_counter()-start,1),flush=True)
    save(OUT/group/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),cohort=cohort,methods=[c['name'] for c in configs],producer=producer_hash(),predictions=len(predictions),wall_seconds=time.perf_counter()-start,all_GT_closed_until_barrier=True));write_csv(OUT/group/'physics_before_GT.csv',[dict(run=r['run'],frame=r['i'],method=r['method'],status=r['status'],**r['physics']) for r in predictions])
    # All predictions/physical flags persisted before importing evaluation code.
    from study_evaluation import evaluate_case,summarize
    tasks=[(r,group,[c['name'] for c in configs]) for r in cohort];summaries=[];rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for j,(summ,rr) in enumerate(pool.map(evaluate_case,tasks),1):summaries+=summ;rows+=rr;print('EVALUATION',group,j,len(tasks),flush=True)
    save(OUT/group/'per_start.json',summaries);write_csv(OUT/group/'per_start.csv',summaries);save(OUT/group/'per_station.json',rows);write_csv(OUT/group/'per_station.csv',rows);save(OUT/group/'range_metrics.json',summarize(rows));write_csv(OUT/group/'range_metrics.csv',summarize(rows));save(OUT/group/'EVALUATION_COMPLETE.json',dict(time_ns=time.time_ns(),starts=len(cohort),station_rows=len(rows),wall_seconds=time.perf_counter()-start));check_baseline();print('COMPLETE',group,flush=True)

if __name__=='__main__':main()
