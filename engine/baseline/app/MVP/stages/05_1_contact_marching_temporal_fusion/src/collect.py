"""Tables and paired conclusions; no inference or parameter changes."""
from fusion_common import *
from collections import Counter

def rows_for(name):
    return [load(p)['summary'] for p in (OUT/'heldout'/name).glob('*/evaluation.json')]

def main():
    lock=load(OUT/'research_lock.json');names=lock['heldout_variants'];allrows=[];steps=[];bins=[];source=[];density=[];blur=[];quality=[];overlap=[]
    diag={}
    for p in (OUT/'diagnostics').glob('*/results.json'):
        for r in load(p):diag[r['variant'],r['run'],r['start_frame']]=r
        source+=csv_read(p.parent/'source_contribution.csv');density+=csv_read(p.parent/'fusion_density.csv');blur+=csv_read(p.parent/'registration_blur.csv')
    for name in names:
        for p in (OUT/'heldout'/name).glob('*/evaluation.json'):
            e=load(p);r=e['summary'];r=diag.get((name,r['run'],r['start_frame']),r);allrows.append(r)
            pred=load(p.parent/'prediction.json');ss={s['step_index']:s for s in pred['steps']}
            for s in e['steps']:
                raw=ss.get(s['step_index'],{});s['variant']=name
                if raw.get('template_candidates'):s['template_region_coverage']=raw['template_candidates'][0]['coverage']
                if raw.get('plane_origin') is not None:s['attempted_range']=float(np.linalg.norm(np.asarray(raw['plane_origin'])+np.asarray(raw['basis'])[:,0]*8))
                steps.append(s)
            bins+=e['bins']
    baseline={(r['run'],r['start_frame']):r for r in allrows if r['variant']=='F0'}
    summary=[];paired=[];rescue=[];orientation=[];range_summary=[];perrun=[];paired_angles=[]
    baseline_steps={(r['run'],r['start_frame'],r['step_index']):r for r in steps if r['variant']=='F0'}
    for name in names:
        rr=[r for r in allrows if r['variant']==name];good=[r for r in rr if r.get('evaluation_eligible') and baseline[r['run'],r['start_frame']].get('evaluation_eligible')]
        available=[r for r in rr if r.get('seed_available')];d=np.array([r['continuous_reach_5cm']-baseline[r['run'],r['start_frame']]['continuous_reach_5cm'] for r in good]);reach=[r['continuous_reach_5cm'] for r in good]
        rsum=dict(variant=name,starts=len(rr),available=len(available),evaluable=len(good),seed_availability=len(available)/len(rr),reach=percentiles(reach),paired_delta=percentiles(d),wins=int(np.sum(d>4)),ties=int(np.sum(abs(d)<=4)),losses=int(np.sum(d< -4)),
            survival={str(x):float(np.mean(np.array(reach)>=x)) for x in (20,30,40,50,60)},anchor_p95=percentiles([r['anchor_error_p95'] for r in good if r.get('anchor_error_p95') is not None]),wrong=sum(bool(r.get('wrong_structure_suspected')) for r in good),raw_wrong=sum(bool(r.get('raw_wrong_structure_trigger')) for r in good),
            stops=dict(Counter(r['stop_reason'] for r in available)),diagnoses=dict(Counter(r.get('terminal_evaluation_reason','NOT_DIAGNOSED') for r in good)),true_range_extensions=sum(r.get('true_range_extension',False) for r in good));summary.append(rsum)
        for run in SPLIT['validation']:
            g=[r for r in good if r['run']==run];perrun.append(dict(variant=name,run=run,**percentiles([r['continuous_reach_5cm'] for r in g])))
        for r in good:
            b=baseline[r['run'],r['start_frame']];delta=r['continuous_reach_5cm']-b['continuous_reach_5cm'];paired.append(dict(variant=name,run=r['run'],start_frame=r['start_frame'],baseline_reach=b['continuous_reach_5cm'],fused_reach=r['continuous_reach_5cm'],delta=delta,outcome='win' if delta>4 else 'loss' if delta< -4 else 'tie',baseline_stop=b['stop_reason'],baseline_diagnosis=b.get('terminal_evaluation_reason'),**{k:r.get(k) for k in ('baseline_terminal_returns','baseline_terminal_fused_returns','confirmed_past_extension_points','true_range_extension')}))
            if name=='F0':continue
            basepath=OUT/'heldout/F0'/key(r['run'],r['start_frame']);newpath=OUT/'heldout'/name/key(r['run'],r['start_frame']);limit=min(b.get('max_accepted_range') or 0,r.get('max_accepted_range') or 0)
            qr=dict(variant=name,run=r['run'],start_frame=r['start_frame'],common_range=limit)
            for folder,label in ((basepath,'baseline'),(newpath,'fusion')):
                with np.load(folder/'point_evaluation.npz') as z:distance=z['distance'];ranges=z['range'];valid=z['evaluable']
                with np.load(folder/'points.npz') as z:ps=z['point_step']
                take=valid&(ps>0)&(ranges<=limit);v=distance[take];qr[label+'_n']=len(v);qr[label+'_p95']=percentiles(v).get('p95');qr[label+'_matched5']=float(np.mean(v<=.05)) if len(v) else None
            quality.append(qr)
            if b['stop_reason']=='OVERLAP_INCONSISTENT':
                bp=load(basepath/'prediction.json');fp=load(newpath/'prediction.json');bs=bp['steps'][-1];fs=fp['steps'][-1]
                overlap.append(dict(variant=name,run=r['run'],start_frame=r['start_frame'],delta=delta,classification='RESCUED' if delta>4 else 'WORSE' if delta< -4 else 'UNCHANGED',baseline_overlap_p90=bs.get('overlap_residual_p90'),fused_overlap_p90=fs.get('overlap_residual_p90'),baseline_overlap_n=bs.get('confirmed_overlap_n'),fused_overlap_n=fs.get('confirmed_overlap_n'),fused_new_support_n=fs.get('new_support_n'),baseline_template_score=bs.get('template_score'),fused_template_score=fs.get('template_score'),baseline_basis=bs.get('basis'),fused_basis=fs.get('basis'),note='Terminal windows can differ; registration_blur compares the SAME baseline overlap window. Worse alone does not prove blur causality.'))
        for category in ('TOO_FEW_SUPPORT','NO_NEW_POINTS','OVERLAP_INCONSISTENT','ORIENTATION_FAILURE','OBSERVATION_TEMPLATE_LIMIT','SENSOR_SPARSITY'):
            gg=[r for r in good if category in (baseline[r['run'],r['start_frame']]['stop_reason'],baseline[r['run'],r['start_frame']].get('terminal_evaluation_reason'))]
            dd=np.array([r['continuous_reach_5cm']-baseline[r['run'],r['start_frame']]['continuous_reach_5cm'] for r in gg]);rescue.append(dict(variant=name,baseline_category=category,n=len(gg),**{f'rescued_{x}m':int(np.sum(dd>=x)) for x in (4,8,12)},median_delta=float(np.median(dd)) if len(dd) else None))
        for subset in ('all','baseline_orientation_failure'):
            ss=[s for s in steps if s['variant']==name and s.get('tangent_angle_error_deg') is not None and (subset=='all' or baseline[s['run'],s['start_frame']].get('terminal_evaluation_reason')=='ORIENTATION_FAILURE')]
            for metric in ('tangent_angle_error_deg','yaw_angle_error_deg','pitch_angle_error_deg'):
                values=[abs(s[metric]) for s in ss if s.get(metric) is not None];orientation.append(dict(variant=name,subset=subset,metric=metric,**percentiles(values)))
        # Common accepted step indices avoid letting newly reached difficult
        # windows alone drive an apparent change of orientation quality.
        if name!='F0':
            for st in steps:
                if st['variant']!=name or st['status']!='ACCEPTED':continue
                bs=baseline_steps.get((st['run'],st['start_frame'],st['step_index']))
                if not bs or bs['status']!='ACCEPTED':continue
                for metric in ('tangent_angle_error_deg','yaw_angle_error_deg','pitch_angle_error_deg'):
                    if st.get(metric) is None or bs.get(metric) is None:continue
                    paired_angles.append(dict(variant=name,run=st['run'],start_frame=st['start_frame'],step_index=st['step_index'],metric=metric,baseline_abs=abs(bs[metric]),fusion_abs=abs(st[metric]),delta_abs=abs(st[metric])-abs(bs[metric]),baseline_orientation_failure=baseline[st['run'],st['start_frame']].get('terminal_evaluation_reason')=='ORIENTATION_FAILURE',note='Matched accepted step ordinal; curved local planes can differ, same frozen oracle reference.'))
        for lo,hi in zip((0,10,20,30,40,50,60),(10,20,30,40,50,60,75)):
            bb=[b for b in bins if b.get('variant')==name and b['lo']==lo];n=sum(b['evaluated'] for b in bb);ss=[s for s in steps if s['variant']==name and s.get('attempted_range') is not None and lo<=s['attempted_range']<hi];a=[s for s in ss if s['status']=='ACCEPTED'];rr=[s for s in a if s.get('template_region_coverage') is not None]
            range_summary.append(dict(variant=name,lo=lo,hi=hi,active_starts=sum(b['predicted']>0 for b in bb),support_count=sum(b['predicted'] for b in bb),evaluated_points=n,point_matched5=sum((b.get('matched_5cm') or 0)*b['evaluated'] for b in bb)/n if n else None,attempts=len(ss),accepted=len(a),accepted_rate=len(a)/len(ss) if ss else None,median_template_region_coverage=float(np.median([s['template_region_coverage'] for s in rr])) if rr else None,anchor_p95=percentiles([s['GT_anchor_error'] for s in a if s.get('GT_anchor_error') is not None]).get('p95')))
    save(OUT/'analysis_rows.json',allrows);save(OUT/'SUMMARY.json',dict(selected=lock['selected'],best_distance=lock['best_distance'],variants=summary,complete=all(s['starts']==1422 for s in summary)))
    for name,rows in [('per_start_variant',allrows),('per_step_variant',steps),('paired_comparison',paired),('source_contribution',source),('fusion_density',density),('registration_blur',blur),('failure_rescue',rescue),('orientation_comparison',orientation),('range_bins_comparison',range_summary),('common_range_quality',quality),('overlap_comparison',overlap),('per_run_summary',perrun)]:csv_write(OUT/(name+'.csv'),rows)
    csv_write(OUT/'orientation_paired.csv',paired_angles)
    csv_write(OUT/'method_ablation.csv',csv_read(OUT/'calibration/method_ablation.csv'))
    print('SUMMARY',[(s['variant'],s['evaluable'],s['reach'].get('p50'),s['wins'],s['losses']) for s in summary],flush=True)
if __name__=='__main__':main()
