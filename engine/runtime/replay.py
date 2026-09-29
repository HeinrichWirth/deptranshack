"""10 Hz independent input + independent near/far checks + bounded heavy work."""
from rt_common import *
import argparse,sqlite3,multiprocessing as mp,traceback
from cdr_cloud import decode

def guarded(function,args,errors):
    try:function(*args)
    except BaseException:
        text=traceback.format_exc();print(text,flush=True);errors.put(text);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--db',required=True);parser.add_argument('--out',required=True);parser.add_argument('--count',type=int,default=1510);parser.add_argument('--stress',action='store_true');a=parser.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cfg=json.loads((HERE/'config.json').read_text())
    origin=json.loads((HERE/'COPY_ORIGIN.json').read_text())
    assert all(sha(HERE/'baseline'/k)==v for k,v in origin['files'].items())
    code_paths=[HERE/'config.json']+[p for folder in ('runtime','native') for p in (HERE/folder).rglob('*') if p.is_file() and p.suffix!='.pyc']
    source_hashes={str(p.relative_to(HERE)):sha(p) for p in code_paths};dump(out/'CODE_BEFORE.json',source_hashes)
    # Only the copied runtime is importable; no baseline result/GT files are mounted.
    os.environ['COPY_REGISTRATION_ACCEL']='parallel';os.environ['COPY_VOXEL']='packed';os.environ['COPY_KD_WORKERS']='1'
    from registration import recovery_worker
    from heavy import registration_worker
    from batch_worker import geometry_worker

    ctx=mp.get_context('spawn');stop=ctx.Event();errors=ctx.Queue()
    ring=CloudRing(ctx,cfg['ring_slots'],cfg['max_points'])
    boxes={name:Latest(ctx,32*1024*1024 if name.startswith('recovery') else 256*1024) for name in ('near','registration','pose','geometry','model_near','model_far','recovery_in','recovery_out','retry','near_reference')}
    work=[('recovery',recovery_worker,(ring,boxes['recovery_in'],boxes['recovery_out'],stop)),
          ('registration',registration_worker,(ring,boxes['registration'],boxes['pose'],boxes['geometry'],boxes['recovery_in'],boxes['recovery_out'],boxes['retry'],stop)),
          ('geometry',geometry_worker,(ring,boxes['geometry'],[boxes['model_near'],boxes['model_far']],boxes['retry'],stop))]
    processes=[];started=time.perf_counter()
    for name,function,args in work:
        ready=ctx.Event();tail=(ready,str(out),a.stress) if name=='recovery' else (ready,str(out),cfg,a.stress) if name=='registration' else (ready,str(out),cfg)
        proc=ctx.Process(target=guarded,args=(function,args+tail,errors),name=name);proc.start();processes.append((proc,ready))
    for proc,ready in processes:
        while not ready.wait(.05):
            if not proc.is_alive() or time.perf_counter()-started>30:
                stop.set()
                for child,_ in processes:
                    child.join(timeout=1)
                    if child.is_alive():child.terminate();child.join()
                raise RuntimeError('Worker startup failed: '+proc.name)
    startup=time.perf_counter()-started
    con=sqlite3.connect(Path(a.db).resolve().as_uri()+'?mode=ro',uri=True)
    topics=con.execute('SELECT id,name,type FROM topics').fetchall();assert len(topics)==1 and topics[0][2]=='sensor_msgs/msg/PointCloud2'
    schedule=con.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT ?',(topics[0][0],a.count)).fetchall()
    logger=log_open(out,'input');begin=time.perf_counter();previous_stamp=None;generation=0;digest=hashlib.sha256();received=0;last_slot=-1;previous_target=-1
    try:
        i=0
        while i<len(schedule):
            message_id,stamp=schedule[i];deadline=begin+(stamp-schedule[0][1])/1e9
            while time.perf_counter()<deadline:time.sleep(min(.002,max(0,deadline-time.perf_counter())))
            # Never drain old payloads after an input stall: skip to the newest
            # already-arrived original message. Every omission is recorded.
            if time.perf_counter()-deadline>.1:
                old=i
                while i+1<len(schedule) and begin+(schedule[i+1][1]-schedule[0][1])/1e9<=time.perf_counter():i+=1
                if i!=old:
                    log(logger,event='SOURCE_INPUT_EXPIRED',frames=list(range(old,i)),clock=time.perf_counter());generation+=1;previous_stamp=None;continue
            entered=time.perf_counter();blob=con.execute('SELECT data FROM messages WHERE id=?',(message_id,)).fetchone()[0];read=time.perf_counter()
            header,points=decode(blob);decoded=time.perf_counter()
            if previous_stamp is not None and not 0<(header['header_time_ns']-previous_stamp)/1e9<.15:generation+=1
            previous_stamp=header['header_time_ns']
            meta=dict(source_frame=i,message_id=message_id,bag_time_ns=stamp,header_time_ns=header['header_time_ns'],deadline=deadline,generation=generation,points=len(points),source_frame_id=header['frame_id'])
            slot=int((stamp-schedule[0][1])*cfg['full_hz']//1_000_000_000)
            target=slot>last_slot
            meta.update(geometry_target=target,solve_slot=slot,batch_start=previous_target+1)
            if target:last_slot=slot;previous_target=i
            values=np.column_stack([points[k] for k in ('x','y','z','intensity')]);ring.put(meta,values);copied=time.perf_counter()
            if i%cfg['registration_stride']==0:boxes['registration'].put(meta)
            published=time.perf_counter();digest.update(blob);hashed=time.perf_counter();received+=1
            log(logger,event='INPUT',**meta,read_ms=(read-entered)*1000,decode_ms=(decoded-read)*1000,copy_ms=(copied-decoded)*1000,
                dispatch_ms=(published-copied)*1000,hash_ms=(hashed-published)*1000,start_lag_ms=(entered-deadline)*1000,delivered_age_ms=(published-deadline)*1000)
            if i%100==0:print('PROGRESS',out.name,i,'/',len(schedule),'input_age_ms',round((published-deadline)*1000,2),flush=True)
            if not errors.empty():raise RuntimeError(errors.get())
            i+=1
        input_end=time.perf_counter()
        # Allow the last on-time jobs to finish; this is not a catch-up queue.
        until=time.perf_counter()+.85
        while time.perf_counter()<until:
            if not errors.empty():raise RuntimeError(errors.get())
            time.sleep(.01)
    finally:
        stop.set();con.close();logger.close()
        for proc,ready in processes:
            proc.join(timeout=4)
            if proc.is_alive():proc.terminate();proc.join();raise RuntimeError('Worker watchdog terminated '+proc.name)
    if not errors.empty():raise RuntimeError(errors.get())
    if any(p.exitcode for p,_ in processes):raise RuntimeError('Worker failed')
    assert all(sha(HERE/k)==v for k,v in source_hashes.items())
    assert all(sha(HERE/'baseline'/k)==v for k,v in origin['files'].items())
    dump(out/'COMPLETE.json',dict(pass_=True,config=cfg,received=received,scheduled=len(schedule),stress=a.stress,
        input_seconds=input_end-begin,recording_seconds=(schedule[-1][1]-schedule[0][1])/1e9,wall_seconds=time.perf_counter()-begin,startup_seconds=startup,
        payload_sha256=digest.hexdigest(),pending_replacements={name:box.replaced.value for name,box in boxes.items()},
        source_db=str(a.db),no_saved_poses=True,no_annotation=True,baseline_files_unchanged=len(origin['files']),runtime_unchanged=True,transport='paced DB3, no DDS measurement'))
    print('COMPLETE',out.name,received,flush=True)

if __name__=='__main__':main()

