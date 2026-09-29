"""Clean, sequential timing and profiling, after parameter selection."""
from . import ROOT,OUT
from .native_engine import NativeEngine
from .persistent import PersistentEngine
from MVP.performance_final2.engine import Engine as Reference
import long_common as lc
import numpy as np
import time,cProfile,pstats,argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scaling-only',action='store_true');ap.add_argument('--skip-scaling',action='store_true');ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    cases=[('roundT_pressureGate_roundT',40),('roundT_squareT_pressureGate_squareT',100),('squareT_platform_squareT_switch',60),('new_data_part_01a2',150)]
    rows=[];fingerprints={}
    for threads in (() if a.skip_scaling else (1,2,4,8,16)):
        for run,start in cases:
            e=NativeEngine(threads,12,0);e.start_run(lc.dataset()/run)
            for i in (start,start+1,start+2):
                r=e.process_frame(i);t=r['summary']['timing'];p=r['prediction']
                fp=None if p is None else lc.sha_bytes(p['pair'].tobytes()) if hasattr(lc,'sha_bytes') else __import__('hashlib').sha256(p['pair'].tobytes()).hexdigest()
                if threads==1:fingerprints[(run,i)]=fp
                rows.append(dict(threads=threads,run=run,frame=i,cold=i==start,total_ms=t['T_TOTAL']*1000,c4_ms=t['T_C4_MARCHING']*1000,
                    active_cores=t['CPU_TOTAL']/t['T_TOTAL'],deterministic=fingerprints[(run,i)]==fp))
            e.close()
    if rows:lc.csv_write(OUT/'full_thread_scaling.csv',rows)
    if a.scaling_only:return
    times=[]
    for backend in ('reference','v2'):
        for run,_ in cases[:3]:
            e=Reference('native_batch_spatial') if backend=='reference' else PersistentEngine(repair=8,threads=a.threads)
            e.start_run(lc.dataset()/run)
            for i in range(90):
                r=e.process_frame(i);t=r['summary']['timing']
                row=dict(backend=backend,run=run,frame=i,total_ms=t['T_TOTAL']*1000,active_cores=t['CPU_TOTAL']/t['T_TOTAL'],timing=t)
                if backend=='v2':row.update(r['persistent_stats'])
                times.append(row)
            e.close();print('CLEAN_TIMING',backend,run,flush=True)
    lc.save(OUT/'clean_timing.json',times)
    for backend in ('reference','v2'):
        e=Reference('native_batch_spatial') if backend=='reference' else PersistentEngine(repair=8,threads=a.threads)
        e.start_run(lc.dataset()/cases[0][0])
        for i in range(20):e.process_frame(i)
        cp=cProfile.Profile();cp.enable()
        for i in range(20,40):e.process_frame(i)
        cp.disable();cp.dump_stats(str(OUT/f'{backend}.prof'))
        with (OUT/f'{backend}_profile.txt').open('w',encoding='utf-8') as f:pstats.Stats(cp,stream=f).sort_stats('cumulative').print_stats(35)
        e.close()

if __name__=='__main__':main()

