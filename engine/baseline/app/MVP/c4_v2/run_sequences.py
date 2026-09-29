"""Full chronological sequence inference; GT is never imported here."""
from . import ROOT,OUT
from .persistent import PersistentEngine
from .native_engine import NativeEngine
from MVP.performance_final2.engine import Engine as Reference
import long_common as lc
import numpy as np
import argparse,time,shutil,json
from concurrent.futures import ProcessPoolExecutor,as_completed


def sequence(task):
    run,group,config,limit=task;name=config['name'];root=OUT/group/name
    engine=PersistentEngine(**{k:v for k,v in config.items() if k!='name'})
    engine.start_run(lc.dataset()/run);n=len(engine.records) if not limit else min(limit,len(engine.records))
    old=lc.load(ROOT/'results_performance_final2/benchmark_cohort.json')['final'];selected={c['frame'] for c in old if c['run']==run and c['frame']<n}
    if not selected:selected=set(np.unique(np.linspace(0,n-1,40).astype(int)).tolist())
    rows=[]
    for i in range(n):
        r=engine.process_frame(i)
        if i in selected:
            folder=root/run/f'{i:06d}';folder.mkdir(parents=True,exist_ok=True)
            if r['prediction'] is not None:
                np.savez_compressed(folder/'prediction.npz',**r['prediction'])
                if engine.state is not None:
                    s=engine.state;np.savez_compressed(folder/'persistent_map.npz',anchors=s.anchors,source_keys=s.keys,observations=s.points,origin_pose=s.origin_pose,sensor_station=s.sensor_station,side=s.side)
            ref=ROOT/'results_performance_final2/equivalence'/run/f'{i:06d}'/'original.npz'
            if ref.exists():shutil.copyfile(ref,folder/'reference.npz')
            lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={p.name:lc.sha(p) for p in folder.glob('*.npz')},labels_used=False))
            rows.append(dict(run=run,frame=i,**{k:v for k,v in r['summary'].items() if k not in ('run','frame')}))
        if i%50==0 or i==n-1:
            print(name,run,i,n,engine.stats[-1]['status'],round(engine.stats[-1]['total_ms']),flush=True)
            lc.save(root/run/'progress.json',dict(frame=i,total=n,last=engine.stats[-1]))
    engine.close();lc.save(root/run/'timings.json',engine.stats);lc.csv_write(root/run/'seams.csv',engine.seams)
    lc.save(root/run/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=rows,frames=n,config=config))
    return dict(run=run,rows=rows,frames=n)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='development');ap.add_argument('--name',default='combined_r8_o8');ap.add_argument('--repair',type=float,default=8);ap.add_argument('--overlap',type=float,default=8);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--iterations',type=int,default=12);ap.add_argument('--grid',type=int,default=0);ap.add_argument('--control',default='combined');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int);ap.add_argument('--run');a=ap.parse_args()
    config=dict(name=a.name,repair=a.repair,overlap=a.overlap,threads=a.threads,iterations=a.iterations,grid=a.grid,control=a.control)
    plan=lc.load(ROOT/'MVP/c4_v2/plan.json');runs=[a.run] if a.run else plan[a.group]
    root=OUT/a.group/a.name;root.mkdir(parents=True,exist_ok=True);lc.save(root/'CONFIG.json',dict(config=config,created_ns=time.time_ns(),concurrent_runs=a.workers))
    tasks=[(run,a.group,config,a.limit) for run in runs];results=[]
    if a.workers==1:
        for t in tasks:results.append(sequence(t))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for result in pool.map(sequence,tasks):results.append(result)
    lc.save(root/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=[r for x in results for r in x['rows']],frames=sum(x['frames'] for x in results),config=config))

if __name__=='__main__':main()
