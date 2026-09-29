from . import OUT
from .async_worker import GeometryWorker
from .async_simulation import service
import long_common as lc
import argparse,time,json,os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--backend',default=os.environ.get('C4_BACKEND','native_batch_spatial'));ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--paced',action='store_true',help='Replay source timestamps; otherwise ingress is as fast as raw read/transform')
    a=ap.parse_args();directory=Path(a.run).resolve();output=Path(a.output).resolve()
    if output==directory or output.is_relative_to(directory):raise ValueError('Use a new output path outside source data')
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty')
    output.mkdir(parents=True,exist_ok=True);worker=GeometryWorker(directory,a.backend,a.workers);records=worker.engine.records;rows=[];start=time.perf_counter()
    try:
        for i,record in enumerate(records):
            due=start+(int(record['header_time_ns'])-int(records[0]['header_time_ns']))/1e9
            if a.paced and due>time.perf_counter():time.sleep(due-time.perf_counter())
            t=time.perf_counter();snap=worker.on_frame(i);dt,n=service(directory,records,i,snap)
            row=dict(frame=i,raw_points=n,service_seconds=time.perf_counter()-t,state_available=snap is not None,
                geometry={} if snap is None else {k:v for k,v in snap.items() if k!='prediction'})
            rows.append(row);print(json.dumps(lc.clean(row)),flush=True)
    finally:worker.close()
    lc.save(output/'RUN_COMPLETE.json',dict(backend=a.backend,workers=a.workers,frames=rows,events=worker.policy.events,max_pending=worker.policy.max_pending,
        obstacle_detector_implemented=False,paced=a.paced))

if __name__=='__main__':main()
