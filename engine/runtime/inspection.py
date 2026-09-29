"""Independent current-cloud near and measured-pose path intrusion checks."""
from rt_common import *
import _envelope
from rail_refresh import refresh
from sequence_policy import agreement,select
from path_state import attach_near,remember,valid as valid_state

def reference(model):
    pair=np.asarray(model['source_pair'])
    if pair.ndim!=3 or pair.shape[1:]!=(2,3) or len(pair)<2 or not np.isfinite(pair).all():raise ValueError('Invalid rail pair')
    centers=pair.mean(axis=1)
    k=max(1,int(np.argmin(abs(np.linalg.norm(centers-centers[0],axis=1)-5))))
    forward=centers[k]-centers[0];across=pair[0,1]-pair[0,0]
    up=np.cross(forward,across)
    if np.linalg.norm(up)<1e-9:raise ValueError('Degenerate rail plane')
    up/=np.linalg.norm(up)
    if up[2]<0:up=-up
    forward=np.array([0.,-1.,0.]);forward-=up*np.dot(forward,up)
    if np.linalg.norm(forward)<1e-9:raise ValueError('Degenerate local forward')
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,up)
    return dict(up=up,forward=forward,right=right,floor=float(np.dot(centers[0],up)),meta=model['meta'],published=model['published'])

def near_worker(ring,inbox,models,nearbox,stop,ready,out,cfg):
    logger=log_open(out,'near');ref=None;last=-1;recent=[];ready.set()
    while not stop.is_set():
        event=models.take()
        model=event.get('model') if event is not None else None
        if model is not None and (ref is None or model['meta']['generation']!=ref['meta']['generation'] or model['meta']['source_frame']>ref['meta']['source_frame']):
            try:ref=reference(model)
            except ValueError as e:ref=None;log(logger,event='NEAR_REFERENCE_REJECTED',reason=str(e))
        meta=inbox.take()
        if meta is None:time.sleep(.001);continue
        src=meta['source_frame']
        if src>last+1:log(logger,event='NEAR_INPUT_REPLACED',frames=list(range(last+1,src)))
        last=src;begin=time.perf_counter();entry_age=age(meta)
        if entry_age>cfg['near_result_deadline_ms']:
            log(logger,event='NEAR_RESULT',source_frame=src,status='UNKNOWN_EXPIRED_INPUT',age_ms=entry_age,count=None);continue
        item=ring.read(src)
        if item is None:log(logger,event='NEAR_RESULT',source_frame=src,status='UNKNOWN_RING_EXPIRED',count=None);continue
        _,points=item;copied=time.perf_counter()
        updated=refresh(points)
        refreshed=time.perf_counter();accepted=updated['status']=='ACCEPTED'
        if not accepted and (ref is None or ref['meta']['generation']!=meta['generation'] or age(ref['meta'])>cfg['near_calibration_hold_ms']):
            log(logger,event='NEAR_RESULT',source_frame=src,status='UNKNOWN_RAIL_HEIGHT',age_ms=age(meta),count=None,copy_ms=(copied-begin)*1000);continue
        if accepted:
            model_local=dict(source_pair=updated['near_pair'],meta=meta,published=time.perf_counter());ref=reference(model_local)
            env=_envelope.Envelope(updated['near_pair'],cfg['width_m'],cfg['height_m'],cfg['penetration_m']+.000001,len(updated['near_pair']))
            built=time.perf_counter();hits=env.query(points,np.eye(4))
        else:
            built=time.perf_counter();hits=_envelope.near(points,ref['up'],ref['forward'],ref['right'],ref['floor'],cfg['width_m'],cfg['height_m'],cfg['penetration_m']+.000001,cfg['near_m'])
        finished=time.perf_counter();late=age(meta)>cfg['near_result_deadline_ms']
        status='EXPIRED_INTRUSION' if late and hits['count'] else 'UNKNOWN_EXPIRED_RESULT' if late else 'INTRUSION' if hits['count'] else 'NO_HITS_IN_NEAR_PRISM'
        if not hits['eligible_points']:status='UNKNOWN_NO_RETURNS'
        if not late and hits['eligible_points']:
            if accepted:near_pair=updated['near_pair']
            else:
                c=ref['up']*ref['floor']+np.linspace(0,8,17)[:,None]*ref['forward']
                near_pair=np.stack((c+ref['right']*.792,c-ref['right']*.792),1)
            recent.append(dict(meta=meta,pair=near_pair,held=not accepted,reference_source_frame=ref['meta']['source_frame']))
            recent=recent[-4:];nearbox.put(recent)
        log(logger,event='NEAR_RESULT',source_frame=src,status=status,age_ms=age(meta),copy_ms=(copied-begin)*1000,
            compute_ms=(finished-copied)*1000,total_ms=(finished-begin)*1000,reference_source_frame=ref['meta']['source_frame'],reference_age_ms=age(ref['meta']),
            up=ref['up'],forward=ref['forward'],right=ref['right'],floor=ref['floor'],clock=finished,
            rail_refresh_status=updated['status'],rail_refresh_ms=updated['timing_ms'],held_reference=not accepted,
            envelope_build_ms=(built-refreshed)*1000,query_ms=(finished-built)*1000,
            near_pair=updated['near_pair'] if accepted else None,**hits)
    logger.close()

def far_worker(ring,poses,models,nearbox,retrybox,stop,ready,out,cfg):
    logger=log_open(out,'far');history=[];decision=None;state=None;near_history=[];requested=None
    ready.set()
    log(logger,event='FAR_CONTROL_UNAVAILABLE',reason='STARTUP_NO_COMPLETE_CHAIN',clock=time.perf_counter())
    while not stop.is_set():
        event=models.take()
        if event is not None:
            decision=event['decision'];candidate=event.get('model')
            if candidate is not None:
                assert decision['full_chain'] and decision['meta']['source_frame']==candidate['meta']['source_frame']
                history=[m for m in history if m['meta']['generation']==candidate['meta']['generation'] and m['meta']['source_frame']<candidate['meta']['source_frame']]
                history.append(candidate);history=history[-cfg['primary_history_versions']:]
                log(logger,event='COMPLETE_CHAIN_RECEIVED',source_frame=candidate['meta']['source_frame'],contact_status=decision['contact_status'])
        job=poses.take()
        if job is None:time.sleep(.001);continue
        meta=job['meta'];src=meta['source_frame'];begin=time.perf_counter()
        if age(meta)>cfg['pose_result_deadline_ms']:
            log(logger,event='FAR_RESULT',source_frame=src,status='UNKNOWN_EXPIRED_POSE',age_ms=age(meta),count=None);continue
        if decision is None or decision['meta']['generation']!=meta['generation']:
            log(logger,event='FAR_RESULT',source_frame=src,status='UNKNOWN_WAITING_FULL_CHAIN',age_ms=age(meta),count=None);continue
        assert decision['meta']['source_frame']<=src
        item=ring.read(src)
        if item is None:
            log(logger,event='FAR_RESULT',source_frame=src,status='UNKNOWN_RING_EXPIRED',count=None);continue
        P=np.asarray(job['pose'],float)
        if P.shape!=(4,4) or not np.isfinite(P).all():
            log(logger,event='FAR_RESULT',source_frame=src,status='UNKNOWN_INVALID_POSE',count=None);continue
        _,points=item;copied=time.perf_counter();updated=refresh(points);accepted=updated['status']=='ACCEPTED'
        near_records=nearbox.take()
        if near_records is not None:near_history=near_records
        current_near=next((n for n in near_history if n['meta']['source_frame']==src and n['meta']['generation']==meta['generation']),None)
        if not accepted and current_near is None:
            log(logger,event='FAR_RESULT',source_frame=src,status='UNKNOWN_NEAR_REFERENCE',age_ms=age(meta),count=None);continue
        near=updated['near_pair'] if accepted else current_near['pair'];refreshed=time.perf_counter()
        # Only complete primary results enter this bounded version history.
        # Never choose a model by its obstacle count or by a standalone CR flag.
        candidates=[m for m in reversed(history) if m['meta']['generation']==meta['generation'] and m['meta']['source_frame']<=src and age(m['meta'])<=5000]
        selected=None;local=None;checks=[]
        for model in candidates:
            candidate_pair=(np.asarray(model['pair_world'])-P[:3,3])@P[:3,:3]
            check=agreement(candidate_pair,near,cfg['primary_vertical_agreement_m'],cfg['primary_lateral_agreement_m'])
            checks.append(dict(model_source=model['meta']['source_frame'],**check))
            if check['accepted']:
                selected=model;local=candidate_pair;break
        cached=valid_state(state,meta,time.perf_counter())
        fallback=(np.asarray(state['pair_world'])-P[:3,3])@P[:3,:3] if cached else local
        contact_status=decision['contact_status'] if age(decision['meta'])<=5000 else 'UNKNOWN_EXPIRED_DECISION'
        chosen=select(local,fallback,near,contact_status=contact_status,fresh_near=accepted,
                      vertical_m=cfg['primary_vertical_agreement_m'],lateral_m=cfg['primary_lateral_agreement_m'])
        active=state if chosen['fallback'] and cached else selected
        corrected=chosen['pair'];original_count=len(near)
        if not chosen['fallback'] and len(corrected)>len(near) and checks and not checks[0]['accepted']:
            chosen['scope']='HELD_PREVIOUS_VERIFIED_FULL_CHAIN'
        need_retry=chosen['retry'] or bool(checks and not checks[0]['accepted'])
        retry_key=(meta['generation'],decision['meta']['source_frame'])
        retry_requested=False
        if need_retry and requested!=retry_key:
            retrybox.put(dict(source_frame=src,generation=meta['generation'],reason='RECOMPUTE_COMPLETE_CHAIN'))
            requested=retry_key;retry_requested=True
        policy_done=time.perf_counter()
        env=_envelope.Envelope(corrected,cfg['width_m'],cfg['height_m'],cfg['penetration_m']+.000001,original_count)
        built=time.perf_counter();hits=env.query(points,np.eye(4));queried=time.perf_counter()
        if accepted and active is not None and len(corrected)>len(near):state=remember(corrected,P,meta,active)
        else:state=state if cached else None
        directory=Path(out)/'rail_refresh';directory.mkdir(exist_ok=True)
        contact=(np.asarray(active['contact_world'])-P[:3,3])@P[:3,:3] if active is not None else np.empty((0,3))
        prior=fallback if chosen['fallback'] else local
        if prior is None:prior=near.copy()
        source=active['meta']['source_frame'] if active is not None else None
        np.savez_compressed(directory/f'{src:06d}.npz',pair=corrected,prior_pair=prior,contact=contact,near_pair=near,
            model_source=-1 if source is None else source,original_count=original_count,pose=P,
            support_indices=updated['support_indices'] if accepted else np.empty(0,dtype=np.int64))
        finished=time.perf_counter();late=age(meta)>cfg['pose_result_deadline_ms']
        far_available=len(corrected)>len(near)
        status=('EXPIRED_INTRUSION' if hits['count'] else 'UNKNOWN_EXPIRED_RESULT') if late else ('INTRUSION' if hits['count'] else 'NO_HITS_IN_MODELED_PRISMS') if far_available else ('NEAR_ONLY_INTRUSION' if hits['count'] else 'NEAR_ONLY_FAR_UNAVAILABLE')
        if not hits['eligible_points']:status='UNKNOWN_NO_RETURNS'
        log(logger,event='FAR_RESULT',source_frame=src,model_source_frame=source,status=status,far_available=far_available,
            age_ms=age(meta),model_age_ms=age(active['meta']) if active is not None else None,
            copy_ms=(copied-begin)*1000,compute_ms=(finished-copied)*1000,total_ms=(finished-begin)*1000,clock=finished,
            rail_refresh_status=updated['status'],rail_refresh_ms=updated['timing_ms'],geometry_scope=chosen['scope'],
            decision_source_frame=decision['meta']['source_frame'],full_step2_status=contact_status,full_chain_completed=True,
            model_agreement_checks=checks,selected_agreement=chosen.get('agreement'),retry_requested=retry_requested,
            fallback_applied=chosen['fallback'],path_policy_ms=(policy_done-refreshed)*1000,
            sparse_completion=updated.get('sparse_completion',False),endpoint_stability_m=updated.get('endpoint_stability_m'),
            used_persisted_path=chosen['fallback'] and cached,prior_correction_source_frame=active.get('correction_source_frame') if active is not None else None,
            near_reference_source_frame=src if accepted else current_near['reference_source_frame'],
            reanchor=chosen.get('reanchor'),envelope_build_ms=(built-policy_done)*1000,query_ms=(queried-built)*1000,export_ms=(finished-queried)*1000,
            contact_refreshed=False,contact_detection_current=False,contact_curve_refreshed=False,**hits)
    logger.close()
