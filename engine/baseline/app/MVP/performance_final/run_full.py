from . import ROOT,OUT
from .engine import Engine
from .ablation import fingerprint
from MVP.final_pipeline.spatial_scheduler import SpatialScheduler
from MVP.final_pipeline.pipeline import write_frame
from MVP.final_pipeline.memory import rss
import long_common as lc
import numpy as np
import argparse,time,os

RUNS=['roundT_squareT_pressureGate_squareT','roundT_pressureGate_roundT','squareT_platform_squareT_switch',
      'new_data_part_01a2','new_data_part_01d1']


def old_saved(run,i):
    for phase in ('deployable_benchmark_cache','deployable_curved_run_cache','deployable_development_cache'):
        p=ROOT/'results_final_pipeline'/phase/lc.key(run,i)
        if (p/'PREDICTION_COMPLETE.json').exists():return p


def compare_saved(result,folder):
    if folder is None:return None
    old=lc.load(folder/'summary.json')
    assert old['final_geometry_available']==result['summary']['final_geometry_available']
    for name,key in (('prediction','prediction'),('curve','c4_smooth'),('c4','c4')):
        path=folder/(name+'.npz')
        if not path.exists():continue
        with np.load(path) as z:
            for k in z.files:np.testing.assert_array_equal(z[k],result[key][k],err_msg=str(folder)+' '+name+' '+k)
    return True


def run(run,backend,mode):
    directory=OUT/'runs'/(mode+'_'+backend)/run
    if (directory/'INFERENCE_COMPLETE.json').exists():return
    directory.mkdir(parents=True,exist_ok=True)
    engine=Engine(backend,orientation=mode=='scheduler')
    scheduler=SpatialScheduler(engine) if mode=='scheduler' else None
    (scheduler or engine).start_run(lc.dataset()/run)
    rows=[];start=time.perf_counter();cpu=time.process_time();serialization=0.;comparisons=0
    for i in range(len(engine.records)):
        before=time.perf_counter();pc=time.process_time()
        if scheduler:
            value=scheduler.process_frame(i);row=value['summary'];fresh=value['fresh_result']
        else:
            fresh=engine.process_frame(i);row=dict(run=run,frame=i,status=fresh['summary']['status'],fresh=fresh['prediction'] is not None,
                valid_geometry_state=fresh['prediction'] is not None,low_motion_reuse=False,stale=False,solve_attempted=True,
                wall_seconds=time.perf_counter()-before,cpu_seconds=time.process_time()-pc,solve_seconds=fresh['summary']['timing']['T_TOTAL'])
        if fresh is not None:
            row['timing']=fresh['summary']['timing'];row['output_hash']=fingerprint(fresh)
            if not scheduler or engine.direction_info.get('mode')=='FROZEN_T_TPLUS1':
                row['old_frozen_match']=compare_saved(fresh,old_saved(run,i));comparisons+=int(row['old_frozen_match'] is True)
            if mode=='scheduler':
                before=time.perf_counter();write_frame(directory/f'{i:06d}',fresh,'research');serialization+=time.perf_counter()-before
        row['peak_rss_bytes']=rss()['peak_rss_bytes'];rows.append(row)
        if (i+1)%25==0:
            lc.save(directory/'progress.json',dict(frame=i,frames=len(engine.records),wall=time.perf_counter()-start,
                fresh=sum(r['fresh'] for r in rows),reuse=sum(r['low_motion_reuse'] for r in rows)))
            print('FULL',mode,backend,run,i+1,len(engine.records),'sec',round(time.perf_counter()-start,1),flush=True)
        del fresh
        if scheduler:del value
    lc.save(directory/'INFERENCE_COMPLETE.json',dict(run=run,backend=backend,mode=mode,rows=rows,
        wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,serialization_seconds=serialization,
        frozen_comparisons=comparisons,frame_count=len(rows),time_ns=time.time_ns(),PID=os.getpid(),
        labels_used=False,future_clouds=False,all_consecutive_frames=True))
    print('COMPLETE',mode,backend,run,len(rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,choices=RUNS);ap.add_argument('--backend',default='python')
    ap.add_argument('--mode',choices=['scheduler','every_frame'],default='scheduler');a=ap.parse_args();run(a.run,a.backend,a.mode)


if __name__=='__main__':main()
