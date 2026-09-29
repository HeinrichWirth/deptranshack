from . import ROOT
from .worker import GeometryWorker
from MVP.performance_final2.async_simulation import service
from pathlib import Path
import long_common as lc
import argparse,os,time,json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--c4-backend',choices=('reference','v2'),default=os.environ.get('C4_BACKEND','reference'))
    ap.add_argument('--config',type=Path);ap.add_argument('--paced',action='store_true');ap.add_argument('--limit',type=int)
    a=ap.parse_args();source=Path(a.run).resolve();output=Path(a.output).resolve()
    if output==source or output.is_relative_to(source):raise ValueError('Output must be separate from source LAS')
    for forbidden in ('stages','final_pipeline','performance_final','performance_final2'):
        if output.is_relative_to((ROOT/'MVP'/forbidden).resolve()):raise ValueError('Frozen code path cannot be an output')
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    config=lc.load(a.config) if a.config else dict(threads=8,iterations=12,grid=0,repair=8,overlap=8)
    worker=GeometryWorker(source,a.c4_backend,config);rows=[];start=time.perf_counter();records=worker.engine.records
    try:
        for i,record in enumerate(records[:a.limit] if a.limit else records):
            due=start+(int(record['header_time_ns'])-int(records[0]['header_time_ns']))/1e9
            if a.paced and due>time.perf_counter():time.sleep(due-time.perf_counter())
            before=time.perf_counter();snap=worker.on_frame(i);_,n=service(source,records,i,snap)
            rows.append(dict(frame=i,raw_points=n,service_seconds=time.perf_counter()-before,state_available=snap is not None,
                geometry={} if snap is None else {k:v for k,v in snap.items() if k!='prediction'}))
    finally:worker.close()
    lc.save(output/'RUN_COMPLETE.json',dict(backend=a.c4_backend,config=config,frames=rows,events=worker.policy.events,max_pending=worker.policy.max_pending,
        actual_background_worker=True,paced=a.paced,wall_seconds=time.perf_counter()-start,production_default='reference'))
    print(json.dumps(dict(frames=len(rows),backend=a.c4_backend,max_pending=worker.policy.max_pending)),flush=True)

if __name__=='__main__':main()

