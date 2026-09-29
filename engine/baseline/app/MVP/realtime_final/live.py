"""Bounded concurrent full solves. Only raw history is shared; C4 restarts every time."""
from . import ROOT,OUT
from .memory_engine import MemoryEngine,SharedRing
from .native_api import load
from .bench import run
from MVP.final_pipeline.frame_data import FrameData
from MVP.final_pipeline.spatial_scheduler import transform_prediction
from MVP.performance_final2.async_worker import remaining
from fusion import valid_edge
from concurrent.futures import ThreadPoolExecutor
import numpy as np,time,argparse,resource
import long_common as lc

class Scheduler:
    def __init__(self,directory,workers=4,threads=1):
        self.engines=[MemoryEngine(threads) for _ in range(workers)]
        for e in self.engines:e.start_run(directory)
        self.records=self.engines[0].records;self.ring=SharedRing(self.records);self.pool=ThreadPoolExecutor(max_workers=workers)
        self.workers=workers;self.threads=threads;self.free=list(range(workers));self.active={};self.pending=None;self.state=None;self.latest_published=-1;self.generation=0;self.last=None
        self.events=[];self.solves=[];self.max_active=0;self.max_pending=0;self.superseded=0;self.discarded_stale=0
    def eligible(self,i):
        row=self.records[i]
        if row.get('pose_status') not in ('ok','origin') or row.get('pose_uses_future',False):return False
        P=np.asarray(row['lidar_pose_in_folder'])
        return np.isfinite(P).all() and (self.state is None or np.linalg.norm(P[:3,3]-self.state['pose'][:3,3])>=.5)
    def poll(self):
        for f in list(self.active):
            if not f.done():continue
            job=self.active.pop(f);self.free.append(job['worker']);r,row=f.result();now=time.perf_counter();i=job['frame']
            row.update(requested=job['requested'],started=job['started'],completed=now,generation=job['generation'],worker=job['worker'],published=False)
            if job['generation']!=self.generation:status='INVALIDATED_GAP'
            elif i<=self.latest_published:status='DISCARDED_OLDER_COMPLETION';self.discarded_stale+=1
            elif r['prediction'] is None:status='UNAVAILABLE_KEEP_LAST'
            else:
                P=np.asarray(self.records[i]['lidar_pose_in_folder']);self.state=dict(prediction=transform_prediction(r['prediction'],P[:3,:3],P[:3,3],True),pose=P.copy(),frame=i,published=now)
                self.latest_published=i;row['published']=True;status='PUBLISHED'
            row['publish_status']=status;self.solves.append(row);self.events.append(dict(event=status,frame=i,clock=now))
        if self.pending is not None and self.free:
            job=self.pending;self.pending=None
            if job['generation']!=self.generation or not self.eligible(job['frame']):return
            w=self.free.pop(0);e=self.engines[w];e.cache=job.pop('cache');e.marcher=job.pop('marcher');e.native_frames=set(e.cache.frames);e.records=self.records[:job['frame']+2]
            job.update(worker=w,started=time.perf_counter());f=self.pool.submit(run,e,job['frame']);self.active[f]=job
            self.max_active=max(self.max_active,len(self.active))
    def arrival(self,raw):
        begin=time.perf_counter();i=raw.frame;self.poll()
        connected=self.last is not None and i==self.last+1 and valid_edge(self.records[self.last],self.records[i])
        if self.last is not None and not connected:
            self.generation+=1;self.state=None;self.pending=None;self.events.append(dict(event='GAP_RESET',frame=i,clock=begin))
        # Capture T's twelve-frame lease BEFORE T+1 retires the oldest raw entry.
        if connected and self.eligible(i-1):
            cache=self.ring.snapshot(i-1);m=load().Marcher(self.threads,12,0);m.options(True);m.attach(self.ring.store,sorted(cache.frames))
            if self.pending is not None:self.superseded+=1
            self.pending=dict(frame=i-1,generation=self.generation,requested=begin,cache=cache,marcher=m);self.max_pending=1
        self.ring.arrive(raw);self.last=i;self.poll();snapshot=self.snapshot(i)
        return snapshot,(time.perf_counter()-begin)*1000
    def snapshot(self,index):
        if self.state is None:return None
        P=np.asarray(self.records[index]['lidar_pose_in_folder']);s=self.state;p=transform_prediction(s['prediction'],P[:3,:3],P[:3,3],False)
        return dict(source_frame=s['frame'],remaining_horizon=remaining(p),age_seconds=(int(self.records[index]['header_time_ns'])-int(self.records[s['frame']]['header_time_ns']))/1e9)
    def close(self):
        self.pending=None
        self.pool.shutdown(wait=True);self.poll()
        for e in self.engines:e.close()

def exercise(directory,start,count,workers,threads):
    s=Scheduler(directory,workers,threads)
    # Offline replay fixture is private to the producer. No future raw is put in a snapshot.
    begin=time.perf_counter();raw=[FrameData.read(directory/s.records[i]['file'],i) for i in range(max(0,start-11),start+count)]
    preload=time.perf_counter()-begin
    for frame in raw:
        if frame.frame<start:s.ring.arrive(frame)
    arrivals=[];cpu=time.process_time();begin=time.perf_counter()
    for rawframe in (f for f in raw if f.frame>=start):
        deadline=begin+(rawframe.frame-start)*.1
        while time.perf_counter()<deadline:
            s.poll();time.sleep(min(.003,max(0,deadline-time.perf_counter())))
        entered=time.perf_counter();snap,service=s.arrival(rawframe);end=time.perf_counter()
        arrivals.append(dict(frame=rawframe.frame,raw_consumer_ms=service,arrival_lateness_ms=(entered-deadline)*1000,end_to_deadline_ms=(end-deadline)*1000,positive_horizon=snap is not None and snap['remaining_horizon']>0,remaining_horizon=None if snap is None else snap['remaining_horizon'],geometry_age=None if snap is None else snap['age_seconds'],geometry_source_frame=None if snap is None else snap['source_frame'],active=len(s.active),pending=int(s.pending is not None)))
    while time.perf_counter()<begin+count*.1:
        s.poll();time.sleep(.003)
    cutoff=time.perf_counter();cpu_used=time.process_time()-cpu
    published=[r for r in s.solves if r['published'] and r['completed']<=cutoff];finished=[r for r in s.solves if r['completed']<=cutoff]
    stat=dict(run=directory.name,start=start,frames=count,workers=workers,threads=threads,preload_seconds=preload,wall_seconds=cutoff-begin,cpu_seconds=cpu_used,active_cores=cpu_used/(cutoff-begin),publish_hz=len(published)/(cutoff-begin),solve_hz=len(finished)/(cutoff-begin),publishes=len(published),completed=len(finished),positive_horizon_fraction=float(np.mean([r['positive_horizon'] for r in arrivals])),raw_consumer_p95_ms=float(np.quantile([r['raw_consumer_ms'] for r in arrivals],.95)),raw_deadline_p95_ms=float(np.quantile([r['end_to_deadline_ms'] for r in arrivals],.95)),max_active=s.max_active,max_pending=s.max_pending,superseded=s.superseded,discarded_stale=s.discarded_stale,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,live_ring_bytes=s.engines[0].cache.memory_bytes(),native_ring=s.ring.store.stats())
    for q in (.5,.95):stat['solve_p'+str(int(q*100))+'_ms']=float(np.quantile([r['wall_ms'] for r in finished],q)) if finished else None
    stat['system_pass']=stat['raw_deadline_p95_ms']<100 and stat['positive_horizon_fraction']>=.9 and stat['max_pending']<=1
    published_ids=[r['frame'] for r in published];assert all(a<b for a,b in zip(published_ids,published_ids[1:]))
    s.close();return dict(summary=stat,arrivals=arrivals,solves=s.solves,events=s.events,index_arrivals=s.ring.arrivals)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=1);p.add_argument('--run',default='roundT_squareT_pressureGate_squareT');p.add_argument('--start',type=int,default=100);p.add_argument('--count',type=int,default=100);p.add_argument('--tag',default='development');a=p.parse_args()
    result=exercise(lc.dataset()/a.run,a.start,a.count,a.workers,a.threads);dest=OUT/'live';dest.mkdir(exist_ok=True)
    path=dest/f'{a.tag}__{a.run}__{a.workers}x{a.threads}.json';lc.save(path,result);print('LIVE_COMPLETE',result['summary'],flush=True)
if __name__=='__main__':main()
