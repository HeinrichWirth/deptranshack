"""Diagnostic replay of fixed causal snapshots, never a live-speed benchmark."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,sqlite3
from pathlib import Path
from types import MappingProxyType
import numpy as np
sys.path[:0]=['/candidate','/work/adapter']
from cdr_cloud import decode
from MVP.realtime_final.memory_engine import MemoryEngine,SharedRing
from MVP.realtime_final.bench import run
from MVP.final_pipeline.frame_data import FrameData
from fusion import CausalSource

source=Path('/work/results/INPUT10HZ_SOLVE5_REPLAY/payload')
out=Path('/work/results/HORIZON_DIAGNOSIS');out.mkdir(exist_ok=True)
poses=[json.loads(s) for s in (source/'poses.jsonl').read_text().splitlines()]
records=[dict(file='ROS_MESSAGE_'+str(p['message_id']),header_time_ns=p['header_time_ns'],lidar_pose_in_folder=p['pose'] if p['pose'] is not None else np.full((4,4),np.nan).tolist(),pose_status=p['status'],pose_uses_future=False) for p in poses]
conn=sqlite3.connect('file:/source/cloud_with_fake_obj_0.db3?mode=ro',uri=True)
results=[]
for i in (4,39,79):
    session=Path('/tmp')/f'horizon_{i}';session.mkdir();(session/'input.frames.json').write_text(json.dumps(dict(frames=records[:i+2])))
    ring=SharedRing(records[:i+2]);history,_=CausalSource('',i,records[:i+1]).history(dict(frames=12))
    for k in sorted(history):
        _,p=decode(conn.execute('SELECT data FROM messages WHERE id=?',(poses[k]['message_id'],)).fetchone()[0]);xyz=np.column_stack([p[n] for n in ('x','y','z')]);ids=np.flatnonzero(np.isfinite(xyz).all(axis=1)&np.any(xyz!=0,axis=1)).astype('uint32');P=np.asarray(poses[k]['pose']);world=xyz[ids].astype(float)@P[:3,:3].T+P[:3,3];rings=np.full(len(ids),-1,np.int16)
        for a in (world,ids,rings):a.setflags(write=False)
        ring.arrive(FrameData(world,rings,ids,MappingProxyType({}),'raw_diagnostic',k,len(p)*16))
    for tight in (True,False):
        engine=MemoryEngine(2,tight=tight);engine.start_run(session);engine.prepare(ring,i)
        engine.marcher.diagnostics(True,True);r,row=run(engine,i)
        with np.load(source/'solves'/f'{i:06d}'/'prediction.npz') as old:
            exact=all(np.array_equal(old[k],r['prediction'][k],equal_nan=True) for k in ('s','C','B','pair'))
        assert exact,(i,tight)
        trace=engine.marcher.trace();roi=[dict(step=e['step'],points=len(e['ids'])) for e in trace if e['operation']=='ROI_exact']
        details=dict(frame=i,tight_roi=tight,exact_saved_prediction=exact,reason=r['c4']['reason'],horizon_m=float(r['prediction']['s'][-1]),observed_range_m=r['c4']['max_confirmed_observed_range'],roi_calls=roi,history=history,
            packs=[dict(step=e['step'],values=e['values']) for e in trace if e['operation']=='pack'],
            blocks=[dict(step=s['step_index'],n=s['new_support_n'],anchor=np.asarray(s['anchor_3d']).tolist(),basis=np.asarray(s['basis']).tolist(),plane_origin=np.asarray(s['plane_origin']).tolist(),state=s['state']) for s in r['c4']['steps']])
        results.append(details)
        if tight:
            keep=[e for e in trace if e['operation'] in ('input_frame','pca_covariance','pca_result','ROI_exact','pack','winner') or e['operation'].startswith('stop:')]
            (out/f'trace_{i:06d}.json').write_text(json.dumps(keep))
            seed=r['c4']['seed'];(out/f'seed_{i:06d}.json').write_text(json.dumps(dict(anchor=np.asarray(seed['anchor']).tolist(),basis=np.asarray(seed['basis']).tolist(),side=seed['side'])))
        print('DIAGNOSIS',json.dumps(details),flush=True)
        engine.close()
(out/'native_checks.json').write_text(json.dumps(dict(pass_=True,cases=results,fixture='raw clouds <=T plus previously causal poses <=T+1; diagnostic only; not live timing'),indent=2))
