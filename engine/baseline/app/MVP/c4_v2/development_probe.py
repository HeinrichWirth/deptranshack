"""Compare full-native variants with stored RAW reference on development only."""
from . import ROOT,OUT
from .native_engine import NativeEngine
import long_common as lc
import numpy as np
import argparse,time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=0);ap.add_argument('--iterations',type=int,default=12);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--stride',type=int,default=5);a=ap.parse_args()
    plan=lc.load(ROOT/'MVP/c4_v2/plan.json');cases=lc.load(ROOT/'results_performance_final2/benchmark_cohort.json')['final']
    selected=[]
    for run in plan['development']:
        selected.extend([c for c in cases if c['run']==run][::a.stride])
    name=f'full_g{a.grid}_i{a.iterations}';root=OUT/'development_probe'/name;rows=[];run=None
    for case in selected:
        if run!=case['run']:
            run=case['run'];engine=NativeEngine(a.threads,a.iterations,a.grid);engine.start_run(lc.dataset()/run)
        r=engine.process_frame(case['frame']);folder=root/run/f'{case["frame"]:06d}';folder.mkdir(parents=True,exist_ok=True)
        if r['prediction'] is not None:np.savez_compressed(folder/'prediction.npz',**r['prediction'])
        reference=ROOT/'results_performance_final2/equivalence'/run/f'{case["frame"]:06d}'/'original.npz'
        if reference.exists():
            with np.load(reference) as z:np.savez_compressed(folder/'reference.npz',**{k:z[k] for k in z.files})
        lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={p.name:lc.sha(p) for p in folder.glob('*.npz')},labels_used=False))
        row=dict(**r['summary'],variant=name,native_stats=engine.marcher.stats());rows.append(row)
        print(name,len(rows),len(selected),run,case['frame'],round(r['summary']['timing']['T_TOTAL']*1000),r['summary']['available_c4_observed_range'],flush=True)
    lc.save(root/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=rows));lc.save(root/'timings.json',rows)

if __name__=='__main__':main()
