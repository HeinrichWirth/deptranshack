"""Identical-work geometry benchmark, not an online replay or pose estimator.

Uses previously causal registered poses as fixed inputs; no GT labels.
Every listed frame is solved once, so scheduler skips cannot improve this score.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys,json,sqlite3,time,argparse,hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
sys.path.insert(0,os.environ.get('COPY_RUNTIME_ROOT','/candidate'));sys.path.insert(0,'/work/adapter')
from cdr_cloud import decode
from MVP.realtime_final.memory_engine import MemoryEngine,SharedRing
from MVP.realtime_final.bench import run
from MVP.final_pipeline.frame_data import FrameData
from fusion import CausalSource

def digest(value):
    if value is None:return None
    if isinstance(value,dict):return {k:digest(v) for k,v in value.items()}
    a=np.ascontiguousarray(value);return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    poses=[json.loads(s) for s in Path('/work/reference/poses.jsonl').read_text().splitlines()];records=[]
    for row in poses:records.append(dict(file='ROS_MESSAGE_'+str(row['message_id']),header_time_ns=row['header_time_ns'],lidar_pose_in_folder=row['pose'] if row['pose'] is not None else np.full((4,4),np.nan).tolist(),pose_status=row['status'],pose_uses_future=False))
    session=out/'session';session.mkdir(exist_ok=True);(session/'input.frames.json').write_text(json.dumps(dict(frames=records)))
    engine=MemoryEngine(args.threads);engine.start_run(session);records=engine.records
    connection=sqlite3.connect('file:/tmp/input.db3?mode=ro',uri=True)
    cases=[50,100,116,118,120,200,218,232,250,368,370,400,500,600,755,813,814,1000,1100,1263,1303,1336,1400,1499]
    results=[];checks=[]
    for i in cases:
        history,_=CausalSource('',i,records[:i+1]).history(dict(frames=12));ring=SharedRing(records)
        for k in sorted(history):
            assert poses[k]['pose'] is not None
            _,p=decode(connection.execute('SELECT data FROM messages WHERE id=?',(poses[k]['message_id'],)).fetchone()[0]);xyz=np.column_stack([p[n] for n in ('x','y','z')]);ids=np.flatnonzero(np.isfinite(xyz).all(axis=1)&np.any(xyz!=0,axis=1)).astype(np.uint32);P=np.asarray(poses[k]['pose']);world=xyz[ids].astype(float)@P[:3,:3].T+P[:3,3];ring_ids=np.full(len(ids),-1,np.int16);fields={'intensity_raw':p['intensity'][ids].copy()}
            for a in (world,ids,ring_ids,*fields.values()):a.setflags(write=False)
            ring.arrive(FrameData(world,ring_ids,ids,MappingProxyType(fields),'fixed_raw_benchmark',k,len(p)*16))
        engine.prepare(ring,i);r,row=run(engine,i);results.append(row)
        checks.append(dict(frame=i,status=row['status'],reason=row['original_reason'],prediction=digest(r['prediction']),curve=digest(r['c4_smooth']),provenance=digest(r['provenance']),selected_keys=digest(r['cloud']['keys'][r['c4']['indices']])))
        print('FIXED',i,round(row['wall_ms'],2),row['status'],flush=True)
    (out/'BENCHMARK.json').write_text(json.dumps(dict(rows=results,checks=checks,frames=cases,threads=args.threads,fixture='previous causal poses plus source DB3; fixed work, not end-to-end'),indent=2)+'\n')
    print('FIXED_COMPLETE',float(np.median([r['wall_ms'] for r in results])),flush=True)

if __name__=='__main__':main()
