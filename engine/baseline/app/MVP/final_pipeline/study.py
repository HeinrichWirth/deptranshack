"""Reproducible RAW inference cohorts. Evaluation is a separate command."""
from . import ROOT
from .pipeline import Pipeline, write_frame
from .verify_raw import exact, normalized
import long_common as lc
import numpy as np
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT=ROOT/'results_final_pipeline'


def run_task(task):
    run,indices,phase,engine=task
    pipeline=Pipeline(dict(engine=engine))
    pipeline.start_run(lc.dataset()/run)
    done=[]
    for i in indices:
        folder=OUT/phase/lc.key(run,i)
        if (folder/'PREDICTION_COMPLETE.json').exists():
            lc.check_marker(folder)
            done.append(lc.load(folder/'summary.json'))
            continue
        r=pipeline.process_frame(i)
        for group in ('phase_d_combinations','phase_e_benchmark'):
            prior=ROOT/'results_contact_marching_long_range'/group/'C4'/lc.key(run,i)
            if (prior/'PREDICTION_COMPLETE.json').exists():
                exact(r['c4'],lc.read_prediction(prior))
                r['summary']['c4_exact']=True
                break
        write_frame(folder,r)
        done.append(r['summary'])
        print('INFERENCE',phase,run,i,r['summary']['status'],round(r['summary']['timing']['T_TOTAL'],3),flush=True)
    return done


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['baseline','development','benchmark','full_runs','curved_run'])
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--engine',choices=['baseline','cache'],default='baseline')
    args=ap.parse_args()
    if args.phase=='baseline':rows=lc.load(ROOT/'results_final_geometry_rescue/protocol.json')['screen']
    elif args.phase in ('full_runs','curved_run'):
        selected=lc.load(OUT/'selected_las_runs.json')['runs']
        if args.phase=='curved_run':selected=[r for r in selected if r['role']=='curved_run']
        rows=[dict(run=r['run'],frame=i) for r in selected for i in range(r['frames'])]
    else:rows=lc.load(ROOT/'results_contact_marching_long_range/audit/cohort.json')['development' if args.phase=='development' else 'heldout']
    phase='deployable_'+args.phase
    if args.engine=='cache':phase+='_cache'
    tasks=[(run,sorted(r['frame'] for r in rows if r['run']==run),phase,args.engine) for run in sorted(set(r['run'] for r in rows))]
    before=time.perf_counter();results=[]
    (OUT/phase).mkdir(parents=True,exist_ok=True)
    if args.workers==1:
        for task in tasks:results.extend(run_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_task,tasks):results.extend(result)
    lc.save(OUT/phase/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),starts=rows,results=results,
        workers=args.workers,engine=args.engine,elapsed_seconds=time.perf_counter()-before,
        annotation_labels=False,future_clouds=False,future_pose_delay_allowed=True))


if __name__=='__main__':main()
