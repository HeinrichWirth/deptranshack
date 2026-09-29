"""Serial latency comparison, only after all full-run processes have ended."""
from . import ROOT, OUT
import long_common as lc
import argparse
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-full',action='store_true')
    args=parser.parse_args()
    barrier=OUT/'batch_full.json'
    while not barrier.exists():
        if not args.wait_full:raise RuntimeError('Complete batch_full before latency benchmark')
        time.sleep(5)
    batch=lc.load(barrier)
    assert len(batch['rows'])==10 and all(r['returncode']==0 for r in batch['rows'])
    initial=OUT/'ablation_initial'
    if not initial.exists():shutil.copytree(OUT/'ablation',initial)
    start=time.time_ns();rows=[]
    for backend in ('python','optimized','spatial','native','native_spatial'):
        with (OUT/'ablation'/f'{backend}.log').open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-B','-m','MVP.performance_final.ablation','--backend',backend],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        rows.append(dict(backend=backend,completed_time_ns=time.time_ns()))
        print('CLEAN_ABLATION_COMPLETE',backend,flush=True)
    subprocess.run([sys.executable,'-B','-m','MVP.performance_final.native_test'],cwd=ROOT,check=True)
    lc.save(OUT/'clean_benchmark.json',dict(start_ns=start,end_ns=time.time_ns(),rows=rows,
        full_run_barrier_ns=barrier.stat().st_mtime_ns,backend_processes=1,
        competing_full_run_processes=0,note='Warm filesystem cache; one sequential pass of the same 32 starts per backend.'))


if __name__=='__main__':main()
