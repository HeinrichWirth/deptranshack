from . import OUT
from .engine import Engine
from .async_worker import Policy,GeometryWorker
from MVP.performance_final.run_full import RUNS
from MVP.final_pipeline.frame_data import FrameData
from MVP.final_pipeline.memory import rss
from fusion import valid_edge
import long_common as lc
import numpy as np
import argparse,time

def service(directory,records,i,snapshot):
    t=time.perf_counter();raw=FrameData.read(directory/records[i]['file'],i);P=np.asarray(records[i]['lidar_pose_in_folder'])
    current_xyz=(raw.world-P[:3,3])@P[:3,:3]
    assert len(current_xyz)==len(raw.point_index)
    return time.perf_counter()-t,len(current_xyz)

def simulate(run,backend):
    engine=Engine(backend);directory=lc.dataset()/run;engine.start_run(directory);records=engine.records;policy=Policy(records)
    times=np.array([(int(r['header_time_ns'])-int(records[0]['header_time_ns']))/1e9 for r in records])
    completion=None;frames=[];solves=[];start=time.perf_counter()
    def dispatch(clock):
        request=policy.start(clock)
        if request is None:return None
        wall=time.perf_counter();result=engine.process_frame(request['frame']);elapsed=time.perf_counter()-wall
        solves.append(dict(frame=request['frame'],started=clock,finished=clock+elapsed,wall_seconds=elapsed,
            timing=result['summary']['timing'],source_frames=result['summary']['source_frames'],ready_pose=request['ready_pose']))
        assert all(j<=request['frame'] for j in result['summary']['source_frames'])
        return (clock+elapsed,result)
    for i,t in enumerate(times):
        while completion is not None and completion[0]<=t:
            when,result=completion;policy.complete(result,when);completion=dispatch(when)
        policy.arrival(i,float(t));policy.offer(i-1,i,float(t))
        before=time.perf_counter();snap=policy.snapshot(i);transform=time.perf_counter()-before
        dt,n=service(directory,records,i,snap);speed=None
        if i and valid_edge(records[i-1],records[i]):
            speed=float(np.linalg.norm(np.asarray(records[i]['lidar_pose_in_folder'])[:3,3]-np.asarray(records[i-1]['lidar_pose_in_folder'])[:3,3])/(times[i]-times[i-1]))
        meta={} if snap is None else {k:v for k,v in snap.items() if k!='prediction'}
        horizon=meta.get('remaining_horizon')
        frames.append(dict(run=run,backend=backend,frame=i,source_seconds=float(t),state_available=snap is not None,
            service_seconds=dt+transform,current_raw_points=n,geometry_transform_seconds=transform,speed_m_s=speed,
            time_to_horizon=None if horizon is None or speed is None or speed<=0 else horizon/speed,
            worker_busy=policy.worker_busy,pending_depth=int(policy.pending_latest_frame is not None),**meta))
        if completion is None:completion=dispatch(float(t))
        if (i+1)%100==0:print('ASYNC',backend,run,i+1,len(records),len(solves),flush=True)
    # Completions after the final raw timestamp are recorded, never retroactively
    # applied to consumers; no offline drain can inflate on-stream availability.
    if completion is not None:policy.complete(completion[1],completion[0])
    engine.close()
    lc.save(OUT/'async'/backend/run/'result.json',dict(frames=frames,solves=solves,events=policy.events,max_pending=policy.max_pending,
        source_duration=float(times[-1]),simulation_host_wall=time.perf_counter()-start,peak_RSS_bytes=rss()['peak_rss_bytes'],
        model='discrete event, actual source timestamps and measured per-solve wall; independent consumer resource model',
        final_pending_not_drained=policy.pending_latest_frame is not None))

def live(run,backend,count=200):
    directory=lc.dataset()/run;worker=GeometryWorker(directory,backend);records=worker.engine.records;rows=[];start=time.perf_counter()
    try:
        for i in range(min(count,len(records))):
            target=start+(int(records[i]['header_time_ns'])-int(records[0]['header_time_ns']))/1e9
            if target>time.perf_counter():time.sleep(target-time.perf_counter())
            before=time.perf_counter();snap=worker.on_frame(i);dt,n=service(directory,records,i,snap);end=time.perf_counter()
            rows.append(dict(frame=i,arrival_lateness_seconds=before-target,service_seconds=end-before,state_available=snap is not None,
                age_seconds=None if snap is None else snap['geometry_age_seconds']))
    finally:worker.close()
    lc.save(OUT/'async_live.json',dict(run=run,backend=backend,frames=rows,events=worker.policy.events,max_pending=worker.policy.max_pending,
        wall_seconds=time.perf_counter()-start,actual_background_thread=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',default='fd');ap.add_argument('--run');ap.add_argument('--live',action='store_true');a=ap.parse_args()
    if a.live:live(a.run or RUNS[0],a.backend)
    else:
        for run in [a.run] if a.run else RUNS[:3]:simulate(run,a.backend)

if __name__=='__main__':main()
