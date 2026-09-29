"""Offline evaluation only. Annotation-derived references never enter inference."""
from . import ROOT
import sys
sys.path.insert(0,str(ROOT/'MVP/stages/06_3_final_geometry_rescue/src'))
from rescue_evaluation import align_prediction
from evaluate_rails import measure
from gt_reference import future_pair_reference
import long_common as lc
import numpy as np
import argparse
import time

OUT=ROOT/'results_final_pipeline'
BINS=[(8,30),(30,50),(50,75),(75,100)]


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase');a=ap.parse_args()
    phase=OUT/a.phase;barrier=lc.load(phase/'INFERENCE_COMPLETE.json')
    summaries=[];values={(name,lo,hi):[] for name in ('DEPLOYABLE_PIPELINE','RESEARCH_REFERENCE') for lo,hi in BINS}
    records=[]
    dev=lc.load(ROOT/'results_latent_contact_reference/protocol.json')['split']['development']
    for n,r in enumerate(barrier['starts']):
        run,i=r['run'],r['frame'];key=lc.key(run,i);folder=phase/key
        lc.check_marker(folder)
        meta=lc.load(folder/'summary.json')
        entry=dict(run=run,frame=i,raw_status=meta['status'],raw_available=meta['final_geometry_available'],
                   reference_available=False,paired=False)
        oldfolder=ROOT/'results_latent_contact_reference'/('phase_d' if run in dev else 'phase_e')/'C4_SMOOTH'/key
        oldpath=oldfolder/'prediction.npz'
        refpath=ROOT/'results_running_rails_from_contact/audit/gt_availability'/key/'reference.npz'
        if not refpath.exists() and meta['final_geometry_available']:
            anchor_cache=ROOT/'results_running_rails_from_contact/audit/future_class1'/run/'anchors.npz'
            if anchor_cache.exists():
                refpath=OUT/'audit/future_reference'/key/'reference.npz'
                if not refpath.exists():
                    ref=future_pair_reference(run,i,folder)
                    refpath.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(refpath,**{k:ref[k] for k in ('s','pair','spread','support')})
        planespath=ROOT/'results_running_rails_from_contact'/('phase_b' if run in dev else 'phase_e')/'B4_SPLINE_50__O1'/key/'prediction.npz'
        old=npz(oldpath) if oldpath.exists() else {}
        entry['reference_available']=bool(old)
        if not meta['final_geometry_available'] or not old or not refpath.exists() or not planespath.exists():
            if meta['final_geometry_available'] and refpath.exists():
                raw=npz(folder/'prediction.npz');ref=npz(refpath)
                if len(ref.get('s',[])):
                    ev=measure(raw,ref,raw['B']);take=ev['valid']&(raw['s']>=8)
                    x=ev['error'][take,1]
                    entry.update(unpaired_evaluated_stations=int(take.sum()),
                        unpaired_far_p95=float(np.percentile(x,95)) if len(x) else None,
                        unpaired_any_gt20cm=bool(np.any(x>.2)),unpaired_any_gt50cm=bool(np.any(x>.5)),
                        unpaired_reference='same future reference, own RAW transverse planes; no old seed geometry')
                    np.savez_compressed(folder/'evaluation_unpaired.npz',valid=take,range=ev['range'],station=raw['s'],
                        raw_error=ev['error'],gt=ev['gt'],C=raw['C'],B=raw['B'])
            summaries.append(entry);continue
        planes=npz(planespath)
        if not planes:
            summaries.append(entry);continue
        raw=npz(folder/'prediction.npz');ref=npz(refpath)
        aligned,ok,_,_=align_prediction(raw,planes)
        aligned_old,okold,_,_=align_prediction(old,planes)
        e=measure(aligned,ref,planes['B']);old_e=measure(aligned_old,ref,planes['B'])
        valid=e['valid']&old_e['valid']&ok&okold&(planes['s']>=8)
        distance=np.linalg.norm(planes['C'],axis=1)
        entry.update(paired=True,matched_stations=int(valid.sum()),raw_horizon=float(raw['s'][-1]),
                     reference_horizon=float(old['s'][-1]),curve_max_abs_difference=float(np.max(abs(raw['C']-old['C']))) if raw['C'].shape==old['C'].shape else None)
        for name,ev in (('DEPLOYABLE_PIPELINE',e),('RESEARCH_REFERENCE',old_e)):
            x=ev['error'][valid,1]
            entry[name+'_far_p95']=float(np.percentile(x,95)) if len(x) else None
            for threshold in (.1,.2,.5,1.):
                entry[name+f'_any_gt{int(threshold*100)}cm']=bool(np.any(x>threshold))
            for lo,hi in BINS:
                take=valid&(distance>=lo)&(distance<hi)
                v=ev['error'][take,1];values[name,lo,hi].extend(v.tolist())
                records.append(dict(run=run,frame=i,method=name,lo=lo,hi=hi,stations=len(v),
                    far_p95=float(np.percentile(v,95)) if len(v) else None,
                    far_gt20cm_stations=int(np.sum(v>.2)),far_gt50cm_stations=int(np.sum(v>.5))))
        np.savez_compressed(folder/'evaluation.npz',valid=valid,range=distance,station=planes['s'],
            raw_error=e['error'],reference_error=old_e['error'],raw_pair=aligned['pair'],
            reference_pair=aligned_old['pair'],gt=e['gt'],C=planes['C'],B=planes['B'])
        summaries.append(entry)
        if (n+1)%50==0:print('EVALUATE',a.phase,n+1,len(barrier['starts']),flush=True)
    metrics=[]
    for (name,lo,hi),v in values.items():
        metrics.append(dict(method=name,lo=lo,hi=hi,stations=len(v),
            far_p95=float(np.percentile(v,95)) if v else None,
            far_gt20cm_stations=int(np.sum(np.asarray(v)>.2)),far_gt50cm_stations=int(np.sum(np.asarray(v)>.5))))
    paired=[r for r in summaries if r['paired'] and r.get('matched_stations',0)>0]
    deltas=[]
    for lo,hi in BINS[1:]:
        x=next(r for r in metrics if r['method']=='DEPLOYABLE_PIPELINE' and r['lo']==lo)
        y=next(r for r in metrics if r['method']=='RESEARCH_REFERENCE' and r['lo']==lo)
        deltas.append(dict(lo=lo,hi=hi,degradation_m=None if x['far_p95'] is None else x['far_p95']-y['far_p95']))
    raw_n=sum(r['raw_available'] for r in summaries);old_n=sum(r['reference_available'] for r in summaries)
    counts={name:{f'gt{cm}cm':sum(r.get(name+f'_any_gt{cm}cm',False) for r in paired) for cm in (10,20,50,100)} for name in ('DEPLOYABLE_PIPELINE','RESEARCH_REFERENCE')}
    rule=lc.load(ROOT/'MVP/final_pipeline/adapters/config.json')['gate_before_benchmark']
    additional20=sum(r['DEPLOYABLE_PIPELINE_any_gt20cm'] and not r['RESEARCH_REFERENCE_any_gt20cm'] for r in paired)
    additional50=sum(r['DEPLOYABLE_PIPELINE_any_gt50cm'] and not r['RESEARCH_REFERENCE_any_gt50cm'] for r in paired)
    extra50=sum(r.get('unpaired_any_gt50cm',False) for r in summaries)
    gate=all(r['degradation_m'] is None or r['degradation_m']<=rule['maximum_pooled_far_p95_degradation_each_bin_m'] for r in deltas)
    gate &= additional20<=len(paired)*rule['additional_far_gt20cm_starts_fraction_max']
    gate &= additional50+extra50<=rule['additional_far_gt50cm_starts_max']
    gate &= (old_n-raw_n)/len(summaries)*100<=rule['availability_loss_percentage_points_max']
    result=dict(phase=a.phase,starts=len(summaries),raw_available=raw_n,reference_available=old_n,
        paired_evaluated_starts=len(paired),metrics=metrics,deltas=deltas,catastrophic_counts=counts,
        newly_gt20cm_starts=additional20,newly_gt50cm_starts=additional50,acceptance_passed=bool(gate),
        unpaired_evaluated_starts=sum(r.get('unpaired_evaluated_stations',0)>0 for r in summaries),
        unpaired_gt50cm_starts=extra50,
        gate=rule,scope='matched common transverse planes, same independent approximate future reference; post-seed stations',
        inference_ns=barrier['time_ns'],evaluation_ns=time.time_ns())
    lc.save(phase/'QUALITY.json',result);lc.save(phase/'per_start_quality.json',summaries)
    lc.csv_write(phase/'range_metrics.csv',metrics);lc.csv_write(phase/'per_start_quality.csv',summaries)
    lc.csv_write(phase/'per_start_range.csv',records)
    print(result,flush=True)


if __name__=='__main__':main()
