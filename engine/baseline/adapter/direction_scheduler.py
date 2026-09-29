"""5 Hz cloud leases with an independent original T+1 direction pose.

Only orchestration changes; all frozen numerical geometry functions are reused.
"""
from stride_scheduler import StrideScheduler
from MVP.realtime_final.live import np,time,load,run,transform_prediction,valid_edge


class DirectionScheduler(StrideScheduler):
    def arrival(self,raw):
        begin=time.perf_counter();i=raw.frame;self.poll()
        connected=self.last is not None and i==self.last+1 and valid_edge(self.records[self.last],self.records[i])
        if self.last is not None and not connected:
            self.generation+=1;self.state=None;self.pending=None
            self.events.append(dict(event='GAP_RESET',frame=i,clock=begin))
        # No solve trigger on the next retained cloud (original T+2).
        self.ring.arrive(raw);self.last=i;self.poll()
        return self.snapshot(i),(time.perf_counter()-begin)*1000

    def reserve_direction(self,i):
        assert i==self.last and len(self.records)==i+1
        if not self.eligible(i):return None
        cache=self.ring.snapshot(i);marcher=load().Marcher(self.threads,12,0)
        marcher.options(True);marcher.attach(self.ring.store,sorted(cache.frames))
        return dict(frame=i,generation=self.generation,cache=cache,marcher=marcher,past=self.records[:i+1])

    def direction_arrival(self,i,record,lease=None):
        begin=time.perf_counter();self.poll()
        if lease is None:lease=self.reserve_direction(i)
        if lease is None or lease['generation']!=self.generation:return
        assert record['source_frame_index']==self.records[i]['source_frame_index']+1
        if not valid_edge(self.records[i],record):
            self.events.append(dict(event='DIRECTION_POSE_UNAVAILABLE',frame=i,clock=begin));return
        if not self.eligible(i):return
        if self.pending is not None:self.superseded+=1
        self.pending=dict(frame=i,generation=self.generation,requested=begin,cache=lease['cache'],marcher=lease['marcher'],
                          direction_records=lease['past']+[record])
        self.max_pending=1;self.poll()

    def poll(self):
        # Same completion/publication/gap policy as frozen Scheduler.poll.
        for future in list(self.active):
            if not future.done():continue
            job=self.active.pop(future);self.free.append(job['worker'])
            result,row=future.result();now=time.perf_counter();i=job['frame']
            row.update(requested=job['requested'],started=job['started'],completed=now,
                       generation=job['generation'],worker=job['worker'],published=False)
            if job['generation']!=self.generation:status='INVALIDATED_GAP'
            elif i<=self.latest_published:status='DISCARDED_OLDER_COMPLETION';self.discarded_stale+=1
            elif result['prediction'] is None:status='UNAVAILABLE_KEEP_LAST'
            else:
                P=np.asarray(self.records[i]['lidar_pose_in_folder'])
                self.state=dict(prediction=transform_prediction(result['prediction'],P[:3,:3],P[:3,3],True),pose=P.copy(),frame=i,published=now)
                self.latest_published=i;row['published']=True;status='PUBLISHED'
            row['publish_status']=status;self.solves.append(row);self.events.append(dict(event=status,frame=i,clock=now))
        if self.pending is not None and self.free:
            job=self.pending;self.pending=None
            if job['generation']!=self.generation or not self.eligible(job['frame']):return
            worker=self.free.pop(0);engine=self.engines[worker]
            engine.cache=job.pop('cache');engine.marcher=job.pop('marcher');engine.native_frames=set(engine.cache.frames)
            # Capture only retained history through T plus the original T+1 pose.
            engine.records=job.pop('direction_records')
            job.update(worker=worker,started=time.perf_counter())
            future=self.pool.submit(run,engine,job['frame']);self.active[future]=job
            self.max_active=max(self.max_active,len(self.active))
