from . import ROOT,OUT
from .engine import Engine
from MVP.performance_final.ablation import fingerprint
from MVP.final_pipeline.memory import rss
import long_common as lc
import time,sys,subprocess
from concurrent.futures import ProcessPoolExecutor

def job(case):
    run,start,backend=case;e=Engine(backend);e.start_run(lc.dataset()/run);rows=[];t=time.perf_counter()
    for i in range(start,start+4):rows.append(fingerprint(e.process_frame(i)))
    e.close();return dict(run=run,hashes=rows,seconds=time.perf_counter()-t,peak_RSS_bytes=rss()['peak_rss_bytes'])

def main():
    backend=lc.load(OUT/'selection.json')['backend']
    runs=sorted(p.name for p in lc.dataset().iterdir() if p.is_dir() and list(p.glob('*.frames.json')))[:8]
    tasks=[(r,len(lc.frames(r))//2,backend) for r in runs]
    lc.save(OUT/'offline_cohort.json',dict(tasks=tasks,selection='first eight lexicographic physical runs, middle four consecutive frames; no label selection'))
    rows=[];reference=None
    for workers in (1,2,4,8):
        t=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(job,tasks))
        elapsed=time.perf_counter()-t;hashes=[r['hashes'] for r in results]
        if reference is None:reference=hashes
        assert hashes==reference
        rows.append(dict(processes=workers,runs=len(tasks),solves=4*len(tasks),wall_seconds=elapsed,
            solves_per_second=4*len(tasks)/elapsed,exact=True,includes_process_startup=True,
            peak_worker_RSS_bytes=max(r['peak_RSS_bytes'] for r in results),
            concurrent_RSS_upper_estimate=workers*max(r['peak_RSS_bytes'] for r in results)))
        print('OFFLINE_THROUGHPUT',rows[-1],flush=True)
    lc.csv_write(OUT/'offline_throughput.csv',rows)

if __name__=='__main__':main()
