"""Standalone Linux RAW pipeline. The default implementation never reuses C4 geometry."""
from . import OUT
from .memory_engine import MemoryEngine,SharedRing
from MVP.final_pipeline.pipeline import write_frame
from pathlib import Path
import argparse,time,json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=1);p.add_argument('--threads',type=int,default=8);p.add_argument('--workers',type=int,default=1);p.add_argument('--mode',choices=['offline','live-replay'],default='offline');a=p.parse_args()
    src=a.run.resolve();dest=a.output.resolve()
    if dest==src or dest.is_relative_to(src):raise ValueError('Derived output must be outside source LAS directory')
    dest.mkdir(parents=True,exist_ok=True)
    if a.mode=='live-replay':
        from .live import exercise
        import long_common as lc
        result=exercise(src,a.start,a.count,a.workers,a.threads);lc.save(dest/'live_result.json',result);print(json.dumps(result['summary']));return
    e=MemoryEngine(a.threads);e.start_run(src);ring=SharedRing(e.records)
    for i in range(a.start,min(a.start+a.count,len(e.records))):
        t=time.perf_counter();ring.prepare_from_files(src,i);e.prepare(ring,i);r=e.process_frame(i);r['summary']['offline_full_ms']=(time.perf_counter()-t)*1000
        write_frame(dest/e.run/f'{i:06d}',r,'online');print(json.dumps(r['summary']),flush=True)
if __name__=='__main__':main()
