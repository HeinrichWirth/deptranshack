"""Latest-value discrete event replay with measured solves and raw service."""
from . import ROOT,OUT
from .persistent import PersistentEngine
from .worker import Policy
from MVP.performance_final2.async_simulation import service
from MVP.performance_final2.async_worker import remaining
import long_common as lc
import numpy as np
import time,argparse


def simulate(run):
    config=lc.load(OUT/'selected_config.json');engine=PersistentEngine(**config);directory=lc.dataset()/run;engine.start_run(directory)
    records=engine.records;policy=Policy(records);times=np.array([(int(r['header_time_ns'])-int(records[0]['header_time_ns']))/1e9 for r in records]);frames=[];solves=[];completion=None;source_horizon={}
    def dispatch(clock):
        request=policy.start(clock)
        if request is None:return None
        start=time.perf_counter();result=engine.process_frame(request['frame']);dt=time.perf_counter()-start
        horizon=0 if result['prediction'] is None else remaining(result['prediction'])
        source_horizon[request['frame']]=horizon
        solves.append(dict(result['persistent_stats'],started=clock,finished=clock+dt,wall_seconds=dt,source_horizon=horizon))
        return clock+dt,result
    for i,t in enumerate(times):
        while completion is not None and completion[0]<=t:
            when,result=completion;policy.complete(result,when);completion=dispatch(when)
        policy.arrival(i,float(t));policy.offer(i-1,i,float(t));before=time.perf_counter();snap=policy.snapshot(i);transform=time.perf_counter()-before
        dt,n=service(directory,records,i,snap);meta={} if snap is None else {k:v for k,v in snap.items() if k!='prediction'}
        horizon=meta.get('remaining_horizon');known=source_horizon.get(meta.get('geometry_source_frame'),0)
        reason='NO_VALID_GEOMETRY' if horizon is None else 'POSITIVE_HORIZON' if horizon>0 else 'OBSERVATION_SHORT_HORIZON' if known<=8.5 else 'COMPUTE_EXPIRED'
        frames.append(dict(run=run,frame=i,source_seconds=float(t),state_available=snap is not None,service_seconds=dt+transform,current_raw_points=n,pending_depth=int(policy.pending_latest_frame is not None),reason=reason,known_source_horizon=known,**meta))
        if completion is None:completion=dispatch(float(t))
        if i%100==0:print('ASYNC_V2',run,i,len(records),flush=True)
    if completion is not None:policy.complete(completion[1],completion[0])
    engine.close();lc.save(OUT/'async'/run/'result.json',dict(frames=frames,solves=solves,events=policy.events,max_pending=policy.max_pending,source_duration=float(times[-1]),model='actual measured solve duration; event replay; no offline drain added to on-stream metrics'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run');a=ap.parse_args()
    for run in ([a.run] if a.run else ['roundT_squareT_pressureGate_squareT','roundT_pressureGate_roundT','squareT_platform_squareT_switch']):simulate(run)

