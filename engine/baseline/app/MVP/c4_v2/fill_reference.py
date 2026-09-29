"""Reference RAW inference for the same held-out frames; no annotation access."""
from . import ROOT,OUT
from MVP.performance_final2.engine import Engine
import long_common as lc
import numpy as np
import time,argparse
from concurrent.futures import ProcessPoolExecutor


def one(task):
    root,run,indices=task;engine=Engine('native_batch_spatial');engine.start_run(lc.dataset()/run);rows=[]
    for i in indices:
        folder=root/run/f'{i:06d}';folder.mkdir(parents=True,exist_ok=True)
        r=engine.process_frame(i)
        if r['prediction'] is not None:np.savez_compressed(folder/'reference.npz',**r['prediction'])
        rows.append(r['summary']);print('REFERENCE',run,i,flush=True)
        if (folder/'PREDICTION_COMPLETE.json').exists():
            mark=lc.load(folder/'PREDICTION_COMPLETE.json');mark['reference_inference_ns']=time.time_ns();mark['files']={p.name:lc.sha(p) for p in folder.glob('*.npz') if p.name!='evaluation.npz'};lc.save(folder/'PREDICTION_COMPLETE.json',mark)
    engine.close();lc.save(root/run/'reference_timings.json',rows);return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();root=OUT/a.folder
    barrier=lc.load(root/'INFERENCE_COMPLETE.json');runs=sorted(set(r['run'] for r in barrier['rows']))
    tasks=[(root,run,sorted(r['frame'] for r in barrier['rows'] if r['run']==run)) for run in runs]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    lc.save(root/'REFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=[r for rs in results for r in rs]))

if __name__=='__main__':main()
