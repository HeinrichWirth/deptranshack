"""Latest-only registration and direction orchestration, isolated from inspection."""
from rt_common import *
from registration import new,restore,capture,DeferredRecovery,defer,provisional_checkpoint,anchor_provisional
from direction_probe import fork
from concurrent.futures import ThreadPoolExecutor

def direction(ring,isolated,meta,stop):
    source=meta['source_frame']+1
    while not stop.is_set() and age(meta)<400:
        item=ring.read(source)
        if item is not None:break
        time.sleep(.002)
    else:return dict(status='DIRECTION_INPUT_TIMEOUT')
    nextmeta,points=item
    if nextmeta['generation']!=meta['generation']:return dict(status='DIRECTION_SOURCE_GAP')
    begin=time.perf_counter()
    try:pose=isolated.step(points[:,:3],isolated.index+1,nextmeta['header_time_ns'],update_reference=False)
    except ValueError as e:return dict(status='DIRECTION_UNAVAILABLE',reason=str(e))
    return dict(status='COMPLETE' if pose['pose'] is not None else 'DIRECTION_UNAVAILABLE',meta=nextmeta,pose=pose,
                ms=(time.perf_counter()-begin)*1000,completed=time.perf_counter())

def registration_worker(ring,inbox,posebox,geobox,recovery_in,recovery_out,retrybox,stop,ready,out,cfg,stress):
    logger=log_open(out,'registration');reg=new();history=[];generation=-1;last_source=-cfg['registration_stride']
    pool=ThreadPoolExecutor(max_workers=1);running=None;pending_direction=None;pending_recovery=None;token=0;injected=False;anchored=True
    ready.set();retry=False;last_requested=-1
    def accepted(meta,row,checkpoint=False,publish=True,emit_pose=True):
        nonlocal history,pending_direction,reg,retry,last_requested
        src=meta['source_frame'];history=[x for x in history if x['meta']['source_frame']<src]
        # Window continuity follows actual original indices, not compacted transport.
        if history and history[-1]['meta']['source_frame']!=src-cfg['registration_stride']:history=[]
        history.append(dict(meta=meta,pose=row['pose']));history=history[-48:]
        if not publish:return
        if False:
            posebox.put(dict(meta=meta,pose=row['pose'],published=time.perf_counter(),status='MEASURED',registration_ms=reg.last_timing_ms.get('total')))
        
        if meta.get('geometry_target',False) and src>last_requested and age(meta)<cfg['geometry_admission_ms']:
            expedited=retry;retry=False;last_requested=src
            before=time.perf_counter();isolated=fork(reg)
            if pending_direction is not None:log(logger,event='DIRECTION_PENDING_REPLACED',source_frame=pending_direction['meta']['source_frame'])
            pending_direction=dict(isolated=isolated,meta=meta,history=history[-cfg['history_raw_frames']:])
            log(logger,event='DIRECTION_PREPARE',source_frame=src,ms=(time.perf_counter()-before)*1000,expedited_retry=expedited)
    try:
        while not stop.is_set():
            failed=retrybox.take()
            if failed is not None and failed['generation']==generation:
                retry=True
                log(logger,event='RETRY_NEXT_FRESH',failed_source=failed['source_frame'])
                if anchored and history and history[-1]['meta']['source_frame']>=failed['source_frame']:
                    h=history[-1];accepted(h['meta'],dict(pose=h['pose']),emit_pose=False)
            if running is not None and running['future'].done():
                job=running;running=None;result=job['future'].result();meta=job['meta']
                log(logger,event='DIRECTION_COMPLETE',source_frame=meta['source_frame'],status=result['status'],ms=result.get('ms'),age_ms=age(meta))
                if result['status']=='COMPLETE' and meta['generation']==generation and age(meta)<=cfg['geometry_admission_ms']:
                    geobox.put(dict(meta=meta,history=job['history'],direction=result,requested=time.perf_counter()))
                elif result['status']=='COMPLETE':log(logger,event='GEOMETRY_ADMISSION_EXPIRED',source_frame=meta['source_frame'],age_ms=age(meta))
            if running is None and pending_direction is not None:
                job=pending_direction;pending_direction=None
                if age(job['meta'])<=cfg['geometry_admission_ms']:
                    job['future']=pool.submit(direction,ring,job.pop('isolated'),job['meta'],stop);running=job
            recovered=recovery_out.take()
            if recovered is not None:
                if pending_recovery is not None and recovered['token']==pending_recovery['token']:
                    pending=pending_recovery;pending_recovery=None
                    if recovered['status']=='COMPLETE' and recovered['generation']==generation:
                        meta=recovered['meta'];row=recovered['pose']
                        log(logger,event='RECOVERY_CHECKPOINT',source_frame=meta['source_frame'],pose=row,ms=recovered['ms'],age_ms=age(meta))
                        if row['pose'] is not None and pending.get('provisional_ok') and history:
                            delta=np.asarray(row['pose'])@np.linalg.inv(pending['anchor_prior'])
                            anchor_provisional(reg,delta)
                            history=[dict(h,pose=(delta@np.asarray(h['pose'])).tolist()) for h in history]
                            anchored=True;latest=history[-1]
                            log(logger,event='PROVISIONAL_CHAIN_ANCHORED',anchor_source_frame=meta['source_frame'],source_frame=latest['meta']['source_frame'],
                                translation_correction_m=float(np.linalg.norm(delta[:3,3])),clock=time.perf_counter())
                            accepted(latest['meta'],dict(pose=reg.pose.tolist()))
                        elif row['pose'] is not None:
                            reg=restore(recovered['state']);anchored=True;history=[];accepted(meta,row,checkpoint=True)
                        else:anchored=False;log(logger,event='RECOVERY_FAILED',status='NO_ACCEPTED_ANCHOR')
                    else:anchored=False;log(logger,event='RECOVERY_FAILED',status=recovered['status'])
                else:log(logger,event='RECOVERY_OBSOLETE_DISCARDED',token=recovered['token'])
            if pending_recovery is not None and (time.perf_counter()-pending_recovery['started'])*1000>cfg['recovery_watchdog_ms']:
                log(logger,event='RECOVERY_WATCHDOG',token=pending_recovery['token']);pending_recovery=None;anchored=False
            meta=inbox.take()
            if meta is None:time.sleep(.002);continue
            src=meta['source_frame']
            if src<=last_source:raise ValueError('Non-monotonic latest transport')
            if src>last_source+cfg['registration_stride']:log(logger,event='REGISTRATION_INPUT_REPLACED',frames=list(range(last_source+cfg['registration_stride'],src,cfg['registration_stride'])))
            last_source=src
            if meta['generation']!=generation:
                generation=meta['generation'];reg=new();history=[];pending_direction=None;pending_recovery=None;anchored=True
                log(logger,event='SOURCE_SESSION',generation=generation,source_frame=src)
            if pending_recovery is not None and not pending_recovery.get('provisional_ok'):
                log(logger,event='RECOVERY_PENDING_NO_POSE',source_frame=src,age_ms=age(meta));continue
            if pending_recovery is None and not anchored:
                log(logger,event='UNANCHORED_NO_POSE',source_frame=src,age_ms=age(meta));continue
            if age(meta)>cfg['registration_admission_ms']:
                log(logger,event='REGISTRATION_ADMISSION_EXPIRED',source_frame=src,age_ms=age(meta));continue
            item=ring.read(src)
            if item is None:log(logger,event='REGISTRATION_RING_EXPIRED',source_frame=src);continue
            _,points=item;begin=time.perf_counter()
            if stress and src>=200 and not injected:
                injected=True;log(logger,event='INJECT_MAIN_PAUSE',source_frame=src,seconds=1,clock=begin);time.sleep(1.)
            state=capture(reg);previous=reg.__dict__.copy();reg.fn['recover_endpoint']=defer
            try:
                row=reg.step(points[:,:3],reg.index+1,meta['header_time_ns'])
            except DeferredRecovery as deferred:
                reg.__dict__.clear();reg.__dict__.update(previous)
                if pending_recovery is not None:
                    pending_recovery['provisional_ok']=False;history=[]
                    log(logger,event='PROVISIONAL_CHAIN_FAILED',source_frame=src);continue
                token+=1
                recovery_in.put(dict(token=token,state=state,meta=meta,generation=generation))
                pending_recovery=dict(token=token,started=time.perf_counter(),anchor_prior=deferred.predicted.copy(),provisional_ok=True)
                provisional_checkpoint(reg,deferred.source,deferred.predicted,meta['header_time_ns']);anchored=False
                accepted(meta,dict(pose=reg.pose.tolist()),publish=False)
                log(logger,event='RECOVERY_SUBMITTED',source_frame=src,token=token,age_ms=age(meta));continue
            except ValueError as e:
                log(logger,event='REGISTRATION_UNAVAILABLE',source_frame=src,reason=str(e),age_ms=age(meta));continue
            log(logger,event='REGISTRATION_COMPLETE' if anchored else 'PROVISIONAL_RELATIVE_POSE',source_frame=src,pose=row,ms=(time.perf_counter()-begin)*1000,
                age_ms=age(meta),detail=reg.last_timing_ms,generation=generation)
            if row['pose'] is not None:accepted(meta,row,publish=anchored)
            else:
                history=[]
                if pending_recovery is not None:pending_recovery['provisional_ok']=False
    finally:
        pool.shutdown(wait=True);logger.close()

