from fusion_common import *
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from fusion import valid_edge

def source_hash(task):
    run,i=task;row=frames(run)[i];path=dataset()/run/row['file']
    with laspy.open(path) as f:h=f.header
    raw=np.memmap(path,dtype=np.uint8,mode='r',offset=h.offset_to_point_data,shape=(h.point_count*h.point_format.size,));digest=hashlib.sha256(raw).hexdigest()
    assert digest==row['point_data_sha256'],str(path)
    return dict(run=run,frame=i,points=h.point_count,point_data_sha256=digest,verified=True)

def main():
    rows=load(OUT/'analysis_rows.json');names=load(OUT/'research_lock.json')['heldout_variants'];baseline={(r['run'],r['start_frame']):r for r in rows if r['variant']=='F0'}
    sparse=[];speed=[];seedcheck=[];source_files=set();next_step={};diagnoses=[]
    for name in names:
        common=[]
        for r in [r for r in rows if r['variant']==name and r.get('evaluation_eligible')]:
            if name=='FUSED_BOOTSTRAP':continue
            reason=r.get('terminal_evaluation_reason')
            if name=='F0':
                if r.get('wrong_structure_review'):reason='CR_GT_GAP'
                elif r.get('wrong_structure_suspected'):reason='WRONG_STRUCTURE_SUSPECTED'
                elif (r.get('terminal_future_points') or 0)<3:
                    info=load(BASE_OUT/'future_gt'/(key(r['run'],r['start_frame'])+'_reference_v2.json'))
                    reason='RUN_END' if info['horizon_reason']=='RUN_END' else 'CR_GT_GAP'
                elif (r.get('oracle_frame_reach') or 0)>r['continuous_reach_5cm']+4:reason='ORIENTATION_FAILURE'
                elif (r.get('terminal_unique') or 0)<3:reason='SENSOR_SPARSITY'
                else:reason='OBSERVATION_TEMPLATE_LIMIT'
            common.append(reason)
        diagnoses.append(dict(variant=name,n=len(common),**dict(Counter(common))))
    for r in rows:
        if not r.get('evaluation_eligible') or r['variant']=='FUSED_BOOTSTRAP':continue
        bp=load(OUT/'heldout/F0'/key(r['run'],r['start_frame'])/'prediction.json')
        fp=load(OUT/'heldout'/r['variant']/key(r['run'],r['start_frame'])/'prediction.json')
        terminal_index=bp['steps'][-1]['step_index'] if bp['steps'] else None
        next_step[r['variant'],r['run'],r['start_frame']]=any(s['step_index']==terminal_index and s['status']=='ACCEPTED' for s in fp['steps'])
    for name in names:
        good=[r for r in rows if r['variant']==name and r.get('evaluation_eligible')]
        for label,lo,hi in [('0',0,0),('1',1,1),('2',2,2),('3-4',3,4),('5-9',5,9),('10+',10,1e12)]:
            rr=[r for r in good if r.get('baseline_terminal_returns') is not None and (r.get('baseline_terminal_future_points') or 0)>=3 and lo<=r['baseline_terminal_returns']<=hi]
            sparse.append(dict(variant=name,current_returns_bin=label,starts=len(rr),current_returns_median=percentiles([r['baseline_terminal_returns'] for r in rr]).get('p50'),fused_returns_median=percentiles([r['baseline_terminal_fused_returns'] for r in rr]).get('p50'),fused_unique_median=percentiles([r['baseline_terminal_fused_unique'] for r in rr]).get('p50'),rescued4=sum((r.get('delta_reach') or 0)>=4 for r in rr),fused_at_least3=sum((r.get('baseline_terminal_fused_unique') or 0)>=3 for r in rr),baseline_failed_step_ordinal_accepted=sum(next_step.get((name,r['run'],r['start_frame']),False) for r in rr)))
        records=[]
        for r in good:
            i=r['start_frame'];mm=frames(r['run']);v=None
            if i:
                dt=(int(mm[i]['header_time_ns'])-int(mm[i-1]['header_time_ns']))/1e9
                if valid_edge(mm[i-1],mm[i]):v=float(np.linalg.norm(np.array(mm[i]['lidar_pose_in_folder'])[:3,3]-np.array(mm[i-1]['lidar_pose_in_folder'])[:3,3])/dt)
            records.append((r,v))
        for label,lo,hi in [('slow_<2',0,2),('2_to_5',2,5),('5_to_10',5,10),('10_plus',10,1e6)]:
            rr=[r for r,v in records if v is not None and lo<=v<hi];d=[r['continuous_reach_5cm']-baseline[r['run'],r['start_frame']]['continuous_reach_5cm'] for r in rr]
            speed.append(dict(variant=name,past_speed_mps_bin=label,n=len(rr),median_delta=percentiles(d).get('p50'),wins=sum(x>4 for x in d),losses=sum(x< -4 for x in d)))
    for group in ('development','heldout'):
        for path in (OUT/group).glob('*/*/prediction.json'):
            p=load(path);source_files.update((p['run'],int(k)) for k in p['fusion']['source_frames'])
    with ProcessPoolExecutor(max_workers=2) as pool:
        verified=[f.result() for f in as_completed([pool.submit(source_hash,t) for t in sorted(source_files)])]
    for p in (OUT/'heldout/FUSED_BOOTSTRAP').glob('*/prediction.json'):
        f=load(p)
        if f['fusion']['actual_frames']!=1:continue
        b=load(OUT/'heldout/F0'/p.parent.name/'prediction.json')
        # Fresh current-only bootstrap should agree where history contributes none.
        same=f['seed']['status']==b['seed']['status'] and f['seed']['indices']==b['seed']['indices']
        seedcheck.append(dict(run=f['run'],frame=f['start_frame'],identical=same))
    assert all(r['identical'] for r in seedcheck)
    csv_write(OUT/'sparsity_rescue.csv',sparse);csv_write(OUT/'speed_strata.csv',speed);csv_write(OUT/'audit/source_integrity.csv',verified)
    csv_write(OUT/'diagnosis_comparison_common_rules.csv',diagnoses)
    save(OUT/'audit/source_integrity.json',dict(files=len(verified),all_source_point_hashes_match=True));save(OUT/'audit/fused_bootstrap_current_only_identity.json',dict(starts=len(seedcheck),all_identical=True))
    print('SUPPLEMENT PASS',len(verified),'sources;',len(seedcheck),'single-frame bootstrap identities',flush=True)
if __name__=='__main__':main()
