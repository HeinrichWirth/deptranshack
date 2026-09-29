"""Full 3 Hz causal cycle. Disjoint fresh batches; raw 10 Hz C4 history."""
from rt_common import *
from rail_refresh import refresh
from sequence_policy import agreement,select
from path_state import remember,valid as valid_state
from obstacle_tracker import Tracker,clusters
from extension_guard import guard
import _obstacle_envelope as envelope
from exact_math import install
from concurrent.futures import ThreadPoolExecutor

def geometry_worker(ring,inbox,modelboxes,retrybox,stop,ready,out,cfg):
    from MVP.realtime_final.memory_engine import MemoryEngine,SharedRing
    from parallel_history import ParallelRing
    from MVP.realtime_final.bench import run
    from MVP.realtime_final.native_api import load
    from MVP.final_pipeline.frame_data import FrameData
    from MVP.final_pipeline.spatial_scheduler import transform_prediction
    from types import MappingProxyType
    from linear_extension import extend_curves
    root=Path(out);session=root/'session';session.mkdir();dump(session/'input.frames.json',dict(frames=[]))
    engine=MemoryEngine(8);engine.start_run(session);install(engine)
    near_pool=ThreadPoolExecutor(max_workers=1)
    prepare_pool=ThreadPoolExecutor(max_workers=4)
    tracker=Tracker();history=[];state=None;used_sources=set();cached_parts={};cached_poses={};shared=None;generation=None;logger=log_open(out,'geometry');farlog=log_open(out,'far');ready.set()
    for name in ('rail_refresh','batches','solves'):(root/name).mkdir()
    while not stop.is_set():
        job=inbox.take()
        if job is None:time.sleep(.002);continue
        meta=job['meta'];src=meta['source_frame'];begin=time.perf_counter()
        if age(meta)>cfg['geometry_admission_ms']:
            log(logger,event='GEOMETRY_ADMISSION_EXPIRED',source_frame=src,age_ms=age(meta));continue
        rows=job['history'];P=np.asarray(rows[-1]['pose']);batch=[h for h in rows if h['meta']['source_frame']>=meta['batch_start']]
        if [h['meta']['source_frame'] for h in batch]!=list(range(meta['batch_start'],src+1)):
            log(logger,event='BATCH_UNAVAILABLE',source_frame=src,reason='MISSING_ACCEPTED_REGISTRATION',expected_start=meta['batch_start'],available=[h['meta']['source_frame'] for h in batch]);continue
        assert job['direction']['meta']['source_frame']==src+1
        records=[dict(file='UNAVAILABLE',header_time_ns=0,lidar_pose_in_folder=np.full((4,4),np.nan).tolist(),pose_status='unavailable',pose_uses_future=False) for _ in range(src+2)]
        owners=[];merged=[];ok=True
        changed=generation!=meta['generation'] or any(k in cached_poses and not np.array_equal(cached_poses[k],h['pose']) for h in rows for k in [h['meta']['source_frame']])
        if changed:cached_parts={};cached_poses={};shared=None;generation=meta['generation']
        parts=cached_parts
        for h in rows:
            k=h['meta']['source_frame'];assert k<=src
            records[k]=record(h['meta'],h['pose'])
            if k not in parts:
                item=ring.read(k)
                if item is None:ok=False;break
                points=item[1];ids=np.flatnonzero(finite(points)).astype(np.uint32);K=np.asarray(h['pose'])
                world=points[ids,:3].astype(float)@K[:3,:3].T+K[:3,3]
                parts[k]=(world,ids,points[ids,3]);cached_poses[k]=np.asarray(h['pose'])
            world,ids,_=parts[k]
            if k>=meta['batch_start']:
                merged.append(world);owners.append(np.column_stack((np.full(len(ids),k,np.uint32),ids)))
        if not ok:log(logger,event='BATCH_UNAVAILABLE',source_frame=src,reason='RING_EXPIRED');continue
        owner=np.concatenate(owners);world=np.concatenate(merged);xyz=np.asarray((world-P[:3,3])@P[:3,:3],np.float32)
        # Frozen detectors keep their original single-T input. All raw past
        # scans remain separately indexed for causal C4 accumulation.
        records[src+1]=record(job['direction']['meta'],job['direction']['pose']['pose'])
        if shared is None:shared=ParallelRing(records)
        shared.records=records
        new_raws=[]
        for h in rows:
            k=h['meta']['source_frame']
            if k in shared.entries:continue
            if k<=shared.latest:raise RuntimeError('Missing immutable history row')
            w,ids,intensity=parts[k];rings=np.full(len(ids),-1,np.int16)
            for a in (w,ids,intensity,rings):a.setflags(write=False)
            new_raws.append(FrameData(w,rings,ids,MappingProxyType(dict(intensity_raw=intensity)),f'RAW_{k}',k,len(w)*16))
        shared.arrive_many(new_raws,prepare_pool)
        cached_parts={k:v for k,v in parts.items() if k>=src-11};cached_poses={k:v for k,v in cached_poses.items() if k>=src-11}
        engine.records=records;engine.cache=shared.snapshot(src);engine.native_frames=set(engine.cache.frames)
        engine.marcher=load().Marcher(8,12,0);engine.marcher.options(True);engine.marcher.attach(shared.store,sorted(engine.cache.frames))
        near_future=near_pool.submit(refresh,shared.entries[src][1])
        prepared=time.perf_counter();result,timing=run(engine,src);solved=time.perf_counter();prediction=result['prediction']
        contact_status=(result.get('step2') or {}).get('status','UNKNOWN_STEP1_UNAVAILABLE')
        model=None
        if prediction is not None:
            wp=transform_prediction(prediction,P[:3,:3],P[:3,3],True);curves=dict(left=wp['pair'][:,0],right=wp['pair'][:,1],contact=wp['C']);original=len(curves['left'])
            try:curves,extension=extend_curves(curves,cfg['extension_m'])
            except ValueError:extension=None
            model=dict(meta=meta,pair_world=np.stack((curves['left'],curves['right']),1),contact_world=curves['contact'],original_count=original,extension=extension,observed_end_world=wp['pair'][-1].copy())
            history=[m for m in history if m['meta']['generation']==meta['generation']][-2:]+[model]
        current_xyz=shared.entries[src][1];updated=near_future.result();refreshed=time.perf_counter();near=updated.get('near_pair') if updated['status']=='ACCEPTED' else None
        checks=[];selected=None;local=None;chosen=None;active=None;extension_guard=None
        if near is not None:
            for m in reversed(history):
                if age(m['meta'])>5000:continue
                candidate=(m['pair_world']-P[:3,3])@P[:3,:3];check=agreement(candidate,near,cfg['primary_vertical_agreement_m'],cfg['primary_lateral_agreement_m'])
                checks.append(dict(model_source=m['meta']['source_frame'],**check))
                if check['accepted']:selected=m;local=candidate;break
            cached=valid_state(state,meta,time.perf_counter());fallback=(state['pair_world']-P[:3,3])@P[:3,:3] if cached else local
            chosen=select(local,fallback,near,contact_status=contact_status,fresh_near=True,vertical_m=cfg['primary_vertical_agreement_m'],lateral_m=cfg['primary_lateral_agreement_m'])
            active=state if chosen['fallback'] and cached else selected;pair=chosen['pair']
            observed_end=(active['observed_end_world']-P[:3,3])@P[:3,:3] if active is not None else near[-1]
            if -observed_end[:,1].mean() < -near[-1,:,1].mean():observed_end=near[-1]
            pair,extension_guard=guard(pair,xyz,observed_end)
            if active is not None and len(pair)>len(near):state=remember(pair,P,meta,active)
            else:state=state if cached else None
        policy_done=time.perf_counter();hits=dict(count=None,eligible_points=len(xyz),all_indices=[]);components=[];rejected=[]
        if chosen is not None:
            env=envelope.Envelope(pair,cfg['width_m'],cfg['height_m'],cfg['penetration_m']+.000001,len(near));built=time.perf_counter();hits=env.query(xyz,np.eye(4));queried=time.perf_counter()
            ids=np.asarray(hits['all_indices'],int)
            components=clusters(xyz[ids].astype(float),ids,pair)
            for c in components:
                r=np.linalg.norm(c['points'],axis=1);weights=np.minimum(1.,(cfg['far_weight_start_m']/np.maximum(r,1e-9))**2)
                c.update(weighted_points=float(weights.sum()),weight_min=float(weights.min()),far_point_count=int((r>70).sum()),raw_source_frames=sorted(set(owner[c['indices'],0].tolist())))
                c['source_owners']=owner[c['indices']].tolist()
            rejected=[c for c in components if c['weighted_points']<cfg['min_weighted_points']]
            components=[c for c in components if c['weighted_points']>=cfg['min_weighted_points']]
        else:built=queried=policy_done
        clustered=time.perf_counter()
        fresh_sources={h['meta']['source_frame'] for h in batch};assert not fresh_sources&used_sources,'Repeated raw evidence in another detection batch';used_sources.update(fresh_sources)
        assignments=tracker.update(src,meta['header_time_ns']/1e9,P,meta['generation'],components);tracked=time.perf_counter()
        # Exact batch input, no LAS and no synthetic path points mixed into cloud.
        np.savez(root/'batches'/f'{src:06d}.npz',xyz=xyz,owners=owner,pose=P)
        dump(root/'batches'/f'{src:06d}.json',dict(meta=meta,history=rows,batch_sources=sorted(fresh_sources),direction_source_frame=src+1,uses_future_clouds=False,near_detector_source=src,obstacle_sources=sorted(fresh_sources),c4_history_sources=[h['meta']['source_frame'] for h in rows]))
        if chosen is not None:
            contact=(active['contact_world']-P[:3,3])@P[:3,:3] if active is not None else np.empty((0,3))
            contact=contact[-contact[:,1]<=extension_guard['accepted_end_m']+1e-5]
            np.savez(root/'rail_refresh'/f'{src:06d}.npz',pair=pair,near_pair=near,contact=contact,prior_pair=local if local is not None else near,model_source=active['meta']['source_frame'] if active else -1,pose=P,original_count=extension_guard['observed_vertices'],support_indices=np.flatnonzero((owner[:,0]==src)&np.isin(owner[:,1],parts[src][1][updated['support_indices']])))
        if prediction is not None:
            d=root/'solves'/f'{src:06d}';d.mkdir();np.savez_compressed(d/'prediction.npz',**prediction)
        finished=time.perf_counter();late=age(meta)>cfg['geometry_result_deadline_ms']
        status='UNKNOWN_NEAR_REFERENCE' if chosen is None else 'EXPIRED_RESULT' if late else 'INTRUSION' if components else 'NO_CONFIRMED_COMPONENT_THIS_BATCH'
        row=dict(event='GEOMETRY_COMPLETE',source_frame=src,status=status,prepare_ms=(prepared-begin)*1000,solve_ms=(solved-prepared)*1000,rail_refresh_ms=(refreshed-solved)*1000,path_policy_ms=(policy_done-refreshed)*1000,envelope_build_ms=(built-policy_done)*1000,query_ms=(queried-built)*1000,cluster_ms=(clustered-queried)*1000,tracker_ms=(tracked-clustered)*1000,export_ms=(finished-tracked)*1000,total_ms=(finished-begin)*1000,age_ms=age(meta),oldest_evidence_age_ms=age(batch[0]['meta']),stages=timing['timing'],contact_status=contact_status,reason=result['summary'].get('status'),source_frames=[h['meta']['source_frame'] for h in rows],batch_sources=sorted(fresh_sources),batch_points=len(xyz),direction_source_frame=src+1,far_available=chosen is not None and len(pair)>len(near),geometry_scope=chosen['scope'] if chosen else 'UNAVAILABLE',model_agreement_checks=checks,extension_guard=extension_guard,weighted_rejected_components=len(rejected),components=len(components),track_ids=assignments,generation=meta['generation'],clock=finished)
        log(logger,**row);log(farlog,**dict(row,event='FAR_RESULT'),**hits)
    dump(root/'tracks.json',dict(config=tracker.cfg,tracks=tracker.export(),independent_nonoverlapping_batches=True,raw_sources_used=len(used_sources)))
    near_pool.shutdown();prepare_pool.shutdown();engine.close();logger.close();farlog.close()

