"""Sequential raw ROS bag -> causal ICP -> unchanged frozen geometry scheduler.

No saved registration, LAS/map, annotation or future cloud is an input.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,sqlite3,threading,queue,argparse,hashlib,traceback,weakref
from pathlib import Path
from types import MappingProxyType
import numpy as np
from cdr_cloud import decode
from causal_registration import CausalRegistration
sys.path.insert(0,os.environ.get('COPY_RUNTIME_ROOT','/candidate'))
from direction_scheduler import DirectionScheduler as StrideScheduler
from direction_probe import probe
def create_scheduler(directory):return StrideScheduler(directory,workers=int(os.environ.get('COPY_WORKERS','1')),threads=int(os.environ.get('COPY_THREADS','2')),stride=int(os.environ.get('COPY_SOLVE_STRIDE','1')),offset=int(os.environ.get('COPY_SOLVE_OFFSET','0')))
from MVP.final_pipeline.frame_data import FrameData
from MVP.final_pipeline.pipeline import write_frame

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distribution(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--db',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--registration-source',type=Path,required=True);parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--count',type=int,default=1510);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    input_stride=int(os.environ.get('COPY_INPUT_STRIDE','1'))
    assert input_stride==2
    solve_stride=int(os.environ.get('COPY_SOLVE_STRIDE','2'));solve_offset=int(os.environ.get('COPY_SOLVE_OFFSET','1'))
    if (out/'COMPLETE.json').exists() or (out/'arrivals.jsonl').exists():raise ValueError('New output directory required')
    manifest=json.loads((args.freeze/'FINAL_GEOMETRY_RELEASE_MANIFEST.json').read_text())
    audit={k:sha(Path('/app')/k)==v['sha256'] for k,v in manifest['runtime_files'].items()};assert all(audit.values())
    save(out/'freeze_before.json',dict(pass_=True,files=len(audit),marker_sha256=sha(args.freeze/'PRODUCTION_V1.freeze')))
    session=out/'session';session.mkdir();save(session/'input.frames.json',dict(frames=[]))
    transport=queue.Queue(maxsize=1);exports=queue.Queue(maxsize=16);stop=threading.Event();errors=[]
    reg=CausalRegistration(args.registration_source);poses=[];arrivals=[];all_solves=[];events=[];publications=[];audit_rows=[]
    metrics=dict(producer_started=None,producer_finished=None,source_messages=0,raw_points=0,eligible_points=0,exported_solves=0,expected_messages=0)
    io_times=[];direction_rows=[];direction_by_frame={}
    source_stat=args.db.stat();source_digest=hashlib.sha256()
    candidate_files={str(p):sha(p) for base in (Path(os.environ.get('COPY_RUNTIME_ROOT','/candidate')),Path('/work/adapter'),Path('/work/registration')) for p in base.rglob('*') if p.is_file()};save(out/'CANDIDATE_SOURCE.json',candidate_files)
    def put(item):
        while not stop.is_set():
            try:transport.put(item,timeout=.1);return
            except queue.Full:pass
    def producer():
        try:
            conn=sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
            topics=conn.execute('SELECT id,name,type,serialization_format FROM topics').fetchall();found=[r for r in topics if r[2]=='sensor_msgs/msg/PointCloud2'];assert len(found)==1 and found[0][3]=='cdr'
            full_schedule=conn.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT ?',(found[0][0],args.count)).fetchall()
            schedule=full_schedule[::input_stride]
            metrics['original_messages']=len(full_schedule);metrics['skipped_messages']=len(full_schedule)-len(schedule)
            metrics['original_recording_seconds']=(full_schedule[-1][1]-full_schedule[0][1])/1e9
            save(out/'source_index_map.json',dict(index_space='frame and provenance.source_frame are retained-stream indices; source_frame_index is original bag ordinal',input_stride=input_stride,frames=[dict(frame=i,source_frame_index=i*input_stride,message_id=m,bag_time_ns=t) for i,(m,t) in enumerate(schedule)]))
            metrics['expected_messages']=len(schedule);save(out/'input.json',dict(db=str(args.db),bytes=source_stat.st_size,mtime_ns=source_stat.st_mtime_ns,topics=topics,messages=len(schedule),clock='original bag timestamps; header timestamps used for geometry/ICP'))
            begin=time.perf_counter();metrics['producer_started']=begin
            events_to_read=[(k,m,t,False) for k,(m,t) in enumerate(schedule)]
            for k in range(solve_offset,len(schedule),solve_stride):
                original_next=k*input_stride+1
                if original_next<len(full_schedule):
                    m,t=full_schedule[original_next];events_to_read.append((k,m,t,True))
            events_to_read.sort(key=lambda r:(r[2],r[1]))
            for index,message_id,bag_stamp,direction_only in events_to_read:
                deadline=begin+(bag_stamp-schedule[0][1])/1e9
                while not stop.is_set() and time.perf_counter()<deadline:time.sleep(min(.005,max(0,deadline-time.perf_counter())))
                if stop.is_set():break
                entered=time.perf_counter();blob=conn.execute('SELECT data FROM messages WHERE id=?',(message_id,)).fetchone()[0];read_done=time.perf_counter()
                meta,points=decode(blob);xyz=np.column_stack([points[k] for k in ('x','y','z')]);decoded=time.perf_counter()
                if not direction_only:source_digest.update(blob)
                payload_sha=hashlib.sha256(blob).hexdigest() if direction_only else None
                hashed=time.perf_counter()
                if direction_only:
                    assert reg.index==index
                    pose,detail,probe_ms=probe(reg,xyz,meta['header_time_ns']);registered=time.perf_counter()
                    row=dict(kind='direction',frame=index,source_frame_index=index*input_stride+1,
                             target_source_frame_index=index*input_stride,message_id=message_id,bag_time_ns=bag_stamp,
                             header_time_ns=meta['header_time_ns'],payload_sha256=payload_sha,
                             read_ms=(read_done-entered)*1000,decode_ms=(decoded-read_done)*1000,
                             payload_hash_ms=(hashed-decoded)*1000,registration_ms=probe_ms,
                             registration_detail_ms=detail,producer_ms=(registered-entered)*1000,
                             raw_deadline=deadline,ready_clock=registered,pose=pose)
                    put((None,row));continue
                pose=reg.step(xyz,index,meta['header_time_ns']);registered=time.perf_counter()
                finite=np.isfinite(xyz).all(axis=1);zero=np.all(xyz==0,axis=1);ids=np.flatnonzero(finite&~zero).astype(np.uint32)
                raw=None
                if pose['pose'] is not None:
                    P=np.asarray(pose['pose']);world=xyz[ids].astype(np.float64)@P[:3,:3].T+P[:3,3]
                    ring=np.full(len(ids),-1,np.int16)
                    fields={'intensity_raw':np.asarray(points['intensity'][ids]).copy()} if 'intensity' in points.dtype.names else {}
                    for a in (world,ring,ids,*fields.values()):a.setflags(write=False)
                    raw=FrameData(world,ring,ids,MappingProxyType(fields),f'{args.db}#message={message_id}',index,len(points)*meta['point_step'])
                ready=time.perf_counter()
                row=dict(frame=index,source_frame_index=index*input_stride,message_id=message_id,bag_time_ns=bag_stamp,header_time_ns=meta['header_time_ns'],source_frame_id=meta['frame_id'],source_points=len(points),eligible_points=len(ids),zero_points=int(zero.sum()),nonfinite_points=int((~finite).sum()),read_ms=(read_done-entered)*1000,decode_ms=(decoded-read_done)*1000,payload_hash_ms=(hashed-decoded)*1000,registration_ms=(registered-hashed)*1000,registration_detail_ms=dict(reg.last_timing_ms),frame_data_ms=(ready-registered)*1000,producer_ms=(ready-entered)*1000,producer_lateness_ms=(entered-deadline)*1000,raw_deadline=deadline,ready_clock=ready,pose=pose)
                put((raw,row));metrics['source_messages']+=1;metrics['raw_points']+=len(points);metrics['eligible_points']+=len(ids)
            conn.close();metrics['producer_finished']=time.perf_counter();put(None)
        except BaseException:
            errors.append(traceback.format_exc());put(None)
    def writer():
        try:
            while True:
                item=exports.get()
                if item is None:exports.task_done();break
                kind,payload=item
                if kind=='solve':
                    result,identity=payload;directory=out/'solves'/f'{identity:06d}';io_begin=time.perf_counter()
                    result['summary']['source_frame_index']=identity*input_stride
                    result['summary']['source_identity_map']=[dict(frame=k,source_frame_index=k*input_stride,message_id=arrivals[k]['message_id']) for k in result['summary']['source_frames']]
                    result['summary']['input_stride']=input_stride
                    direction=direction_by_frame[identity]
                    result['summary']['direction_pose_source_frame_index']=direction['source_frame_index']
                    result['summary']['direction_pose_message_id']=direction['message_id']
                    result['summary']['direction_pose_policy']='original_T_plus_1_pose_only'
                    write_frame(directory,result,'online');io_times.append(dict(operation='write_solve',frame=identity,ms=(time.perf_counter()-io_begin)*1000));metrics['exported_solves']+=1
                else:
                    prediction,frame=payload;directory=out/'published';directory.mkdir(exist_ok=True);io_begin=time.perf_counter();np.savez_compressed(directory/f'{frame:06d}.npz',**prediction);io_times.append(dict(operation='write_publication',frame=frame,ms=(time.perf_counter()-io_begin)*1000))
                exports.task_done()
        except BaseException:errors.append(traceback.format_exc());stop.set()
    export_thread=threading.Thread(target=writer,name='diagnostic-export');export_thread.start()
    ingest=threading.Thread(target=producer,name='raw-causal-registration');ingest.start()
    runtime=None;records=[];attached=weakref.WeakSet();seen_published=set();latest=-1;session_id=0
    arrival_log=(out/'arrivals.jsonl').open('w');pose_log=(out/'poses.jsonl').open('w')
    def observe():
        if runtime is None:return
        for future,job in list(runtime.active.items()):
            if future in attached:continue
            attached.add(future);frame=job['frame'];engine=runtime.engines[job['worker']]
            clouds=sorted(engine.cache.frames);max_pose=len(engine.records)-1
            assert frame % runtime.solve_stride == runtime.solve_offset
            assert clouds and max(clouds)<=frame and min(clouds)>=frame-11 and max_pose==frame+1 and frame<=latest
            direction=direction_by_frame[frame]
            assert engine.records[-1]['source_frame_index']==frame*input_stride+1
            assert engine.records[-1]['header_time_ns']==direction['header_time_ns']
            assert all(records[k]['pose_status'] in ('ok','origin') for k in clouds)
            audit_rows.append(dict(frame=frame,source_frame_index=frame*input_stride,dispatched_after_frame=latest,
                                  max_pose_frame=max_pose,cloud_frames=clouds,original_cloud_frames=[k*input_stride for k in clouds],
                                  max_original_pose_frame=direction['source_frame_index'],
                                  direction_message_id=direction['message_id'],direction_pose_only=True,session=session_id))
            def completed(f,frame=frame,allowed=tuple(clouds)):
                try:
                    result,row=f.result()
                    assert set(result['summary']['source_frames'])<=set(allowed)
                    if len(result['provenance']):assert int(result['provenance']['source_frame'].max())<=frame
                    queued=time.perf_counter();exports.put(('solve',(result,frame)));io_times.append(dict(operation='queue_solve_wait',frame=frame,ms=(time.perf_counter()-queued)*1000))
                except BaseException:errors.append(traceback.format_exc());stop.set()
            future.add_done_callback(completed)
        if runtime.state is not None:
            state=runtime.state;frame=state['frame']
            if frame not in seen_published:
                seen_published.add(frame);stamp=state['published'];source=arrivals[frame]
                publications.append(dict(source_frame=frame,published_after_frame=latest,clock=stamp,latency_from_raw_ms=(stamp-source['raw_deadline'])*1000,latency_from_frame_data_ms=(stamp-source['ready_clock'])*1000))
                exports.put(('publish',(state['prediction'],frame)))
    def finish_session():
        nonlocal runtime
        if runtime is not None:
            observe();runtime.close();observe();all_solves.extend(dict(r,session=session_id) for r in runtime.solves);events.extend(dict(r,session=session_id) for r in runtime.events);runtime=None
    started=time.perf_counter();cpu_started=time.process_time()
    try:
        while True:
            if runtime is not None:observe();runtime.poll();observe()
            try:item=transport.get(timeout=.003)
            except queue.Empty:
                if errors:raise RuntimeError(errors[0])
                continue
            if item is None:break
            raw,row=item;index=row['frame'];pose=row['pose']
            if row.get('kind')=='direction':
                direction_rows.append(row);direction_by_frame[index]=row
                assert index==latest
                if runtime is not None:
                    P=pose['pose'] if pose['pose'] is not None else np.full((4,4),np.nan).tolist()
                    record=dict(file='POSE_ONLY_NO_CLOUD',header_time_ns=row['header_time_ns'],
                                lidar_pose_in_folder=P,pose_status=pose['status'],pose_uses_future=False,
                                source_frame_index=row['source_frame_index'])
                    runtime.direction_arrival(index,record);observe()
                continue
            latest=index;poses.append(pose)
            P=pose['pose'] if pose['pose'] is not None else np.full((4,4),np.nan).tolist()
            record=dict(file=f'ROS_MESSAGE_{row["message_id"]}',header_time_ns=row['header_time_ns'],lidar_pose_in_folder=P,pose_status=pose['status'],pose_uses_future=False,source_frame_index=row['source_frame_index'])
            records.append(record);arrivals.append(row);begin=time.perf_counter();snapshot=None
            if raw is None:
                finish_session();row['geometry_status']='NO_CAUSAL_POSE'
            else:
                if runtime is None:
                    session_id+=1;runtime=create_scheduler(session);runtime.records.extend(records)
                else:runtime.records.append(record)
                assert len(runtime.records)==index+1
                snapshot,service=runtime.arrival(raw);observe();row['core_arrival_ms']=service
                row['geometry_status']='AVAILABLE' if snapshot is not None else 'WAITING_FOR_GEOMETRY'
            row['consumer_ms']=(time.perf_counter()-begin)*1000;row['deadline_ms']=(time.perf_counter()-row['raw_deadline'])*1000
            row['snapshot']=snapshot;row['positive_horizon']=snapshot is not None and snapshot['remaining_horizon']>0
            io_begin=time.perf_counter();arrival_log.write(json.dumps({k:v for k,v in row.items() if k!='pose'})+'\n');arrival_log.flush();pose_log.write(json.dumps(dict(frame=index,source_frame_index=row['source_frame_index'],header_time_ns=row['header_time_ns'],message_id=row['message_id'],**pose))+'\n');pose_log.flush();io_times.append(dict(operation='write_arrival_and_pose',frame=index,ms=(time.perf_counter()-io_begin)*1000))
            if index%50==0:print('PROGRESS',index+1,'/',metrics['expected_messages'],'pose',pose['status'],'solves',len(seen_published),'lag_ms',round(row['deadline_ms'],1),'wall_s',round(time.perf_counter()-started,1),flush=True)
        finish_session()
        if errors:raise RuntimeError(errors[0])
    finally:
        stop.set();ingest.join(timeout=30);finish_session();arrival_log.close();pose_log.close();exports.put(None);export_thread.join(timeout=60)
    if errors:raise RuntimeError(errors[0])
    elapsed=time.perf_counter()-started;save(out/'COPY_IO_TIMINGS.json',io_times);source_after=args.db.stat();assert (source_after.st_size,source_after.st_mtime_ns)==(source_stat.st_size,source_stat.st_mtime_ns)
    checks={k:sha(Path('/app')/k)==v['sha256'] for k,v in manifest['runtime_files'].items()};assert all(checks.values())
    save(out/'direction_poses.json',direction_rows)
    save(out/'solves.json',all_solves);save(out/'events.json',events);save(out/'publications.json',publications);save(out/'causality_audit.json',dict(pass_=True,solves=audit_rows,no_future_clouds=True,pose_limit='original T+1 pose only',saved_poses_used=False,retroactive_interpolation=False))
    duration=(arrivals[-1]['bag_time_ns']-arrivals[0]['bag_time_ns'])/1e9
    summary=dict(frames=len(arrivals),recording_seconds=duration,wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu_started,valid_poses=sum(p['pose'] is not None for p in poses),invalid_pose_frames=[p['index'] for p in poses if p['pose'] is None],sessions=session_id,completed_solves=len(all_solves),available_solves=sum(r['final_geometry_available'] for r in all_solves),published=len(publications),publish_hz_wall=len(publications)/elapsed,positive_horizon_fraction=float(np.mean([r['positive_horizon'] for r in arrivals])),raw_deadline_p95_ms=distribution([r['deadline_ms'] for r in arrivals]).get('p95'),timing={key:distribution([r[key] for r in arrivals if key in r]) for key in ('read_ms','decode_ms','registration_ms','frame_data_ms','producer_ms','producer_lateness_ms','core_arrival_ms','consumer_ms','deadline_ms')},geometry_solve_ms=distribution([r['wall_ms'] for r in all_solves]),c4_ms=distribution([r['c4_ms'] for r in all_solves]),publication_latency_ms=distribution([r['latency_from_raw_ms'] for r in publications]),observed_horizon_m=distribution([r['available_c4_observed_range'] for r in all_solves if r['final_geometry_available']]),last_frame_solve='NOT_REQUESTED: no T+1',source_message_payload_sha256=source_digest.hexdigest(),source_unchanged=True,frozen_runtime_unchanged=True,registration_source_hashes=reg.source_hashes,annotation_used=False,ground_truth_accuracy='unavailable',**metrics)
    assert all(sha(Path(p))==digest for p,digest in candidate_files.items())
    summary['copy_main']=dict(workers=int(os.environ.get('COPY_WORKERS','1')),threads=int(os.environ.get('COPY_THREADS','2')),runtime_root=os.environ.get('COPY_RUNTIME_ROOT'),voxel=os.environ.get('COPY_VOXEL','original'),las_export=False,solve_stride=int(os.environ.get('COPY_SOLVE_STRIDE','1')),solve_offset=int(os.environ.get('COPY_SOLVE_OFFSET','0')))
    summary['realtime_10hz_deadline_pass']=summary['raw_deadline_p95_ms']<100
    summary['copy_main']['input_stride']=input_stride
    summary['retained_input_hz_nominal']=10/input_stride
    summary['retained_input_deadline_p95_pass']=summary['raw_deadline_p95_ms']<100*input_stride
    summary['index_space']='retained stream; original identities in source_index_map.json and source_frame_index'
    summary['direction_policy']='original T+1 pose only; isolated ICP probe, retained 5 Hz registration unchanged'
    summary['direction_pose_messages']=len(direction_rows)
    summary['total_payloads_read']=len(arrivals)+len(direction_rows)
    summary['not_read_messages']=metrics['original_messages']-summary['total_payloads_read']
    summary['last_frame_solve']='NOT_SELECTED' if (len(arrivals)-1)%solve_stride!=solve_offset else 'T+1 required'
    summary['direction_timings_ms']={key:distribution([r[key] for r in direction_rows]) for key in ('read_ms','decode_ms','registration_ms','producer_ms')}
    save(out/'SUMMARY.json',summary);save(out/'COMPLETE.json',dict(pass_=True,frames=len(arrivals),runtime_files_checked=len(checks),summary_sha256=sha(out/'SUMMARY.json')))
    print('COMPLETE',json.dumps({k:summary[k] for k in ('frames','wall_seconds','valid_poses','completed_solves','published','realtime_10hz_deadline_pass')}),flush=True)

if __name__=='__main__':main()
