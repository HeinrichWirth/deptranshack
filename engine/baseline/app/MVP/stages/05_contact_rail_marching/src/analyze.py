"""Post-prediction analysis and complete tabular outputs. No parameter selection."""
from common import *
from evaluation import load_prediction,reference,curve_match
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import gzip

def block_ci(rows,metric,block=20,reps=1000):
    rng=np.random.default_rng(270927);groups=[]
    for run in sorted(set(r['run'] for r in rows)):
        rr=sorted([r for r in rows if r['run']==run],key=lambda r:r['start_frame'])
        groups.append([[r for r in rr if r['start_frame']//block==b] for b in sorted(set(r['start_frame']//block for r in rr))])
    samples=[]
    for _ in range(reps):
        x=[r[metric] for blocks in groups for j in rng.integers(0,len(blocks),len(blocks)) for r in blocks[j] if r.get(metric) is not None]
        if x:samples.append(np.median(x))
    return dict(block_frames=block,replicates=reps,median_interval_95=np.percentile(samples,[2.5,97.5]).tolist() if samples else None,
                note='Blocks defined by floor(original frame index / block length), independently within each run; no bridging missing frame intervals.')

def observations(run,i,pred,ref,xyz):
    row=frames(run)[i];cloud=laspy.read(dataset()/run/row['file']);rings=np.asarray(cloud.ring) if 'ring' in cloud.point_format.dimension_names else np.zeros(len(xyz),dtype=int)
    cache=OUT/'heldout'/key(run,i)/'observability.json'
    if cache.exists():return load(cache),cloud
    ranges=np.linalg.norm(xyz,axis=1);nearest=cKDTree(ref['future']).query(xyz)[0] if len(ref['future']) else np.full(len(xyz),np.inf)
    # Evaluation-only observability: returns close to future labelled surface.
    rawaz=np.rint(np.arctan2(xyz[:,1],xyz[:,0])/1e-5).astype(np.int64);out=[]
    from evaluation import BINS
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        take=(ranges>=lo)&(ranges<hi)&(nearest<=.05)
        out.append(dict(run=run,start_frame=i,lo=lo,hi=hi,returns=int(take.sum()),unique_xyz=len(np.unique(xyz[take],axis=0)),
                        distinct_rings=len(np.unique(rings[take])),distinct_azimuth_samples=len(np.unique(rawaz[take])),
                        definition='current T geometry within 5cm of fused future labelled surface; evaluation only'))
    temporary=cache.with_name(f'observability_{os.getpid()}.json')
    save(temporary,out);temporary.replace(cache);return out,cloud

def main():
    rows=[];steps=[];bins=[];align=[];orientation=[];runtime=[];packets=[];observability=[]
    oracle_rows=[]
    for p in (OUT/'oracles').glob('*/*/evaluation.json'):
        e=load(p);oracle_rows.append(dict(oracle=p.parent.parent.name,**e['summary']))
    oracle={(r['run'],r['start_frame'],r['oracle']):r for r in oracle_rows}
    for folder in sorted((OUT/'heldout').iterdir()):
        if not folder.is_dir() or not (folder/'evaluation.json').exists():continue
        e=load(folder/'evaluation.json');r=e['summary'];pred=load_prediction(folder);run=r['run'];i=r['start_frame']
        if not r['seed_available']:
            r.update(evaluation_eligible=False,terminal_evaluation_reason='START_UNAVAILABLE');rows.append(r);continue
        xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,info=reference(run,i,'heldout')
        with np.load(folder/'point_evaluation.npz') as point_eval:
            seed_errors=point_eval['distance'][pred['point_step']==0]
        r['seed_current_GT_p95']=percentiles(seed_errors).get('p95')
        r['seed_current_GT_conflict']=r['seed_current_GT_p95'] is not None and r['seed_current_GT_p95']>.20
        r['future_component_not_linked']=bool(info.get('seed_gap_m',0)>.50)
        pp=xyz[pred['indices']];gg=ref['future'];maxr=r.get('max_accepted_range') or 0
        if len(gg) and len(pp):
            active=np.linalg.norm(gg,axis=1)<=maxr
            r['future_surface_coverage_5cm_while_active']=float(np.mean(cKDTree(pp).query(gg[active])[0]<=.05)) if active.any() else None
        else:r['future_surface_coverage_5cm_while_active']=None
        obs,cloud=observations(run,i,pred,ref,xyz);observability.extend(obs)
        last=pred['steps'][-1] if pred['steps'] else {};failed=last.get('status')=='STOP'
        future_n=current_n=returns=0;maxgap=None;sidechange=False
        if last.get('plane_origin') is not None:
            C=np.asarray(last['plane_origin']);B=np.asarray(last['basis']);W=pred['config']['window'];uv=(xyz-C)@B
            f=(ref['future']-C)@B;cc=(ref['current']-C)@B
            near=np.array(last.get('overlap_anchor',[0.,0.]))
            future_n=int(np.sum((f[:,0]>=W/2)&(f[:,0]<=W)&(np.linalg.norm(f[:,1:]-near,axis=1)<.4)))
            current_n=int(np.sum((cc[:,0]>=W/2)&(cc[:,0]<=W)&(np.linalg.norm(cc[:,1:]-near,axis=1)<.4)))
            ids=np.flatnonzero((uv[:,0]>=W/2)&(uv[:,0]<=W))
            if len(ref['future']) and len(ids):returns=int(np.sum(cKDTree(ref['future']).query(xyz[ids])[0]<=.05))
            ff=(ref['allfuture']-C)@B
            sidechange=bool(np.any((ff[:,0]>=W/2)&(ff[:,0]<=W)&(abs(ff[:,1]-near[0])>1.5)&(abs(ff[:,2]-near[1])<.3))) and future_n<3
        of=oracle.get((run,i,'ORACLE_FRAME'));os=oracle.get((run,i,'ORACLE_SEED_8'))
        reach=r.get('continuous_reach_5cm') or 0
        if not r.get('evaluation_eligible'):diagnosis='GT_UNAVAILABLE_AHEAD'
        elif r.get('wrong_structure_suspected'):diagnosis='WRONG_STRUCTURE_SUSPECTED'
        elif future_n<3:diagnosis='CR_SIDE_CHANGE_SUSPECTED' if sidechange else ('RUN_END' if info['horizon_reason']=='RUN_END' else 'CR_GT_GAP')
        elif of and (of.get('continuous_reach_5cm') or 0)>reach+4:diagnosis='ORIENTATION_FAILURE'
        elif os and (os.get('continuous_reach_5cm') or 0)>reach+4:diagnosis='SEED_OR_INITIAL_DRIFT'
        elif returns<3:diagnosis='SENSOR_SPARSITY'
        else:diagnosis='OBSERVATION_TEMPLATE_LIMIT'
        # Registration uncertainty is recorded, not used to alter acceptances/tolerance.
        r.update(terminal_evaluation_reason=diagnosis,terminal_future_points=future_n,terminal_current_gt_points=current_n,terminal_observable_returns=returns)
        for st in e['steps']:
            source_step=next((x for x in pred['steps'] if x['step_index']==st['step_index']),None)
            if source_step and source_step.get('template_candidates'):
                st['template_region_coverage']=source_step['template_candidates'][0]['coverage']
            for j,axis in enumerate(('x','y','z')):
                st['plane_origin_'+axis]=st.get('plane_origin_'+str(j))
                st['tangent_'+axis]=st.get('basis_'+str(3*j))
        rows.append(r);steps.extend(e['steps']);bins.extend(e['bins'])
        alignment=dict(run=run,start_frame=i,**info.get('alignment',{}));align.append(alignment)
        orientation.extend([x for x in e['steps'] if x.get('tangent_angle_error_deg') is not None])
        runtime.append(dict(run=run,start_frame=i,seed_ms=r['seed_ms'],march_ms=r['march_ms'],total_ms=r['total_ms'],io_ms=pred['seed'].get('io_ms')))
        packet=dict(run=run,start_frame=i,summary=r,last_successful_step=next((s for s in reversed(pred['steps']) if s['status']=='ACCEPTED'),None),
                    failed_step=last,reference=info,terminal_current_GT_count=current_n,terminal_future_GT_count=future_n,
                    real_current_cloud_returns_near_future_GT=returns,diagnosis=diagnosis,
                    oracle_frame=of,oracle_seed=os,explanation='Diagnosis uses post-prediction counterfactuals and raw support counts; GT-based wrong structure remains suspected, not manually proven.')
        if last.get('plane_origin') is not None:
            candidate_ids=np.flatnonzero((uv[:,0]>=0)&(uv[:,0]<=W)&(np.linalg.norm(uv[:,1:]-near,axis=1)<.4))
            packet['candidate_source_rows']=candidate_ids
            packet['candidate_source_point_indices']=np.asarray(cloud.point_index)[candidate_ids]
            packet['candidate_GT_nearest_distances']=cKDTree(ref['future']).query(xyz[candidate_ids])[0] if len(ref['future']) else None
            packet['candidate_range_from_sensor_T']=np.linalg.norm(xyz[candidate_ids],axis=1)
        save(OUT/'failures'/(key(run,i)+'.json'),packet);packets.append(packet)
    csv_write(OUT/'per_start_frame.csv',rows);csv_write(OUT/'per_step.csv',steps);csv_write(OUT/'distance_bins.csv',bins)
    csv_write(OUT/'future_gt_alignment.csv',align);csv_write(OUT/'orientation_metrics.csv',orientation);csv_write(OUT/'runtime.csv',runtime)
    csv_write(OUT/'observability.csv',observability);csv_write(OUT/'oracle_comparison.csv',oracle_rows)
    good=[r for r in rows if r.get('evaluation_eligible')];accepted=[r for r in rows if r['seed_available']]
    perrun=[]
    for run in SPLIT['validation']:
        rr=[r for r in rows if r['run']==run];gg=[r for r in good if r['run']==run]
        perrun.append(dict(run=run,starts=len(rr),seed_available=sum(r['seed_available'] for r in rr),evaluated=len(gg),
                          **{'reach_'+str(t):percentiles([r[f'continuous_reach_{t}cm'] for r in gg]) for t in (2,5,10)}))
    summary=dict(total_starts=len(rows),seed_available=len(accepted),seed_availability=len(accepted)/len(rows),meaningful_evaluated=len(good),per_run=perrun,
                 reach={str(t):percentiles([r[f'continuous_reach_{t}cm'] for r in good]) for t in (2,5,10)},
                 block_bootstrap={str(t):block_ci(good,f'continuous_reach_{t}cm') for t in (2,5,10)},
                 survival={str(d):float(np.mean([r['continuous_reach_5cm']>=d for r in good])) for d in (10,20,30,40,50,60,75,100,125)},
                 median_anchor_p95=percentiles([r['anchor_error_p95'] for r in good if r.get('anchor_error_p95') is not None]).get('p50'),
                 steps=percentiles([r['march_steps'] for r in good]),alignment_p95=percentiles([a['p95'] for a in align if a.get('p95') is not None]),
                 runtime_per_start=percentiles([r['total_ms'] for r in good]),runtime_per_step=percentiles([s['total_ms'] for s in steps if s.get('total_ms') is not None]))
    from collections import Counter
    summary['stops']=dict(Counter(r['stop_reason'] for r in accepted));summary['diagnoses']=dict(Counter(r['terminal_evaluation_reason'] for r in good))
    summary['start_unavailable']=dict(Counter(r['stop_reason'] for r in rows if not r['seed_available']))
    summary['seed_current_GT_conflicts']=sum(r.get('seed_current_GT_conflict',False) for r in accepted)
    summary['future_component_not_linked']=sum(r.get('future_component_not_linked',False) for r in accepted)
    summary['demonstrated_survival_all_seed_available']={str(d):sum((r.get('continuous_reach_5cm') or 0)>=d for r in accepted)/len(accepted) for d in (10,20,30,40,50,60,75,100,125)}
    save(OUT/'SUMMARY.json',summary);save(OUT/'analysis_rows.json',rows)
    csv_write(OUT/'reach_summary.csv',[dict(tolerance_cm=t,**summary['reach'][str(t)]) for t in (2,5,10)])
    csv_write(OUT/'failure_summary.csv',[dict(type='detector_stop',reason=k,count=v) for k,v in summary['stops'].items()]+[dict(type='offline_diagnosis',reason=k,count=v) for k,v in summary['diagnoses'].items()])
    # Compactness association is stratified by objective/method, not mixed scales.
    compact=[]
    for name in sorted(p.name for p in (OUT/'ablation').iterdir() if p.is_dir()):
        aa=[]
        for p in (OUT/'ablation'/name).glob('*/evaluation.json'):
            ev=load(p);aa.extend([s for s in ev['steps'] if s.get('transverse_spread') is not None and s.get('tangent_angle_error_deg') is not None])
        rho=float(spearmanr([s['transverse_spread'] for s in aa],[s['tangent_angle_error_deg'] for s in aa]).statistic) if len(aa)>3 else None
        compact.append(dict(variant=name,steps=len(aa),spearman_spread_tangent_error=rho,angle_p95=percentiles([s['tangent_angle_error_deg'] for s in aa]).get('p95')))
    csv_write(OUT/'compactness_metrics.csv',compact)
    template_bins=[]
    for lo,hi in zip((0,10,20,30,40,50,60,75,100,125),(10,20,30,40,50,60,75,100,125,150)):
        rr=[st for st in steps if st.get('range_far') is not None and lo<=st['range_far']<hi and st.get('template_region_coverage') is not None]
        template_bins.append(dict(lo=lo,hi=hi,steps=len(rr),**percentiles([st['template_region_coverage'] for st in rr])))
    csv_write(OUT/'template_coverage_bins.csv',template_bins)
    print('SUMMARY',clean(summary),flush=True)
if __name__=='__main__':main()
