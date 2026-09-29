"""Recompute frozen geometry and evaluation in NEW paths; stop on any mismatch."""
from lc_common import *
from seed_input import read_input,calibrate_seed
from track_geometry import curve_model
from rail_models import predict,INPUT_KEYS
from concurrent.futures import ProcessPoolExecutor
import argparse

def reproduce(task):
    run,i,group,name=task;folder=OLD_OUT/group/name/key(run,i);record=load(folder/'prediction.json');marker=load(folder/'PREDICTION_COMPLETE.json')
    for f,h in marker['files'].items():assert sha(folder/f)==h
    meta,a=read_input(OLD_OUT/record['input_folder']);c4,points=read_c4(run,i);out={};detail={};maxdiff=0.
    if meta['status']=='AVAILABLE':
        initial=np.asarray(c4['seed']['basis']);anchors=np.asarray(c4.get('curve',[c4['seed']['anchor']]))
        if len(anchors)<2:anchors=np.array([c4['seed']['anchor'],np.asarray(c4['seed']['anchor'])+8*initial[:,0]])
        model=curve_model(anchors,initial)
        for f in ('s','C','B'):np.testing.assert_array_equal(model[f],a[f],err_msg=f)
        # Re-run the unchanged seed estimator on its pinned current-8m input.
        seed=calibrate_seed(a['seed_points'],a['seed_labels'],model,initial,meta['q'],8)
        for f in ('x0','covariance','mad','slope'):np.testing.assert_allclose(seed[f],meta['seed'][f],rtol=0,atol=1e-12)
        priors=load(OLD_OUT/'models/global_prior.json');prior=priors['leave_run_out'].get(run,priors['default']);out=predict(seed,{k:a[k] for k in INPUT_KEYS},meta['profiles'],meta['q'],record['config'],prior)
        with np.load(folder/'prediction.npz') as z:
            assert set(out)==set(z.files)
            for f in out:
                np.testing.assert_allclose(out[f],z[f],rtol=1e-11,atol=1e-12,equal_nan=True,err_msg=str((run,i,f)))
                if np.asarray(out[f]).dtype.kind=='f':
                    difference=np.abs(out[f]-z[f]);maxdiff=max(maxdiff,float(np.nanmax(difference))) if np.isfinite(difference).any() else maxdiff
        linear=np.column_stack([np.interp(a['s'],a['anchor_knots'],a['anchor_xyz'][:,k]) for k in range(3)]);detail['max_C2_vs_linear_cm']=float(np.linalg.norm(a['C']-linear,axis=1).max()*100)
    else:assert record['status']==meta['status']
    dest=OUT/'reproduction'/group/key(run,i);dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'prediction.npz',**out);save(dest/'prediction.json',dict(run=run,i=i,method=name,status=meta['status'],old_folder=folder.relative_to(ROOT).as_posix(),config=record['config'],max_numerical_difference=maxdiff,**detail));save(dest/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={f:sha(dest/f) for f in ('prediction.json','prediction.npz')},GT_used=False))
    return dict(run=run,frame=i,group=group,status=meta['status'],max_difference=maxdiff,**detail)

def evaluate(task):
    # Imported only in this post-prediction phase.
    from evaluate_rails import measure,rows_for
    run,i,group,name=task;folder=OUT/'reproduction'/group/key(run,i);record=load(folder/'prediction.json');old=ROOT/record['old_folder'];rows=[];bad=False
    if record['status']=='AVAILABLE':
        with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
        with np.load(OLD_OUT/'audit/gt_availability'/key(run,i)/'reference.npz') as z:ref={k:z[k] for k in z.files}
        e=measure(p,ref)
        with np.load(old/'evaluation.npz') as z:
            for f in e:np.testing.assert_allclose(e[f],z[f],rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=str((run,i,f)))
        rows=rows_for(dict(record,cr_input='CR1_C4'),p,e);bad=bool(np.any(e['valid']&(p['s']>=8)&(e['error'][:,1]>.2)))
    return rows,key(run,i) if bad else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args();lock=check_baseline();name=lock['expected']['method'];protocol=load(OLD_OUT/'protocol.json');tasks=[(r['run'],r['frame'],group,name) for group,cohort in [('phase_b','development'),('phase_e','heldout')] for r in protocol['cohort'][cohort]];results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j,r in enumerate(pool.map(reproduce,tasks),1):
            results.append(r)
            if j%100==0 or j==len(tasks):print('REPRODUCTION INFERENCE',j,len(tasks),flush=True)
    save(OUT/'reproduction/INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),starts=len(tasks),all_arrays_match=True));write_csv(OUT/'reproduction/per_start.csv',results)
    rows={'phase_b':[],'phase_e':[]};bad=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task,(rr,case) in zip(tasks,pool.map(evaluate,tasks)):
            rows[task[2]]+=rr
            if case:bad.append(case)
    from evaluate_rails import aggregate
    for group in rows:
        metrics=aggregate(rows[group]);expected=[r for r in load(OLD_OUT/group/'range_metrics.json') if r['method']==name];assert len(metrics)==len(expected)
        for r in metrics:
            wanted=next(v for v in expected if v['lo']==r['lo'] and v['hi']==r['hi'])
            for f,v in r.items():
                if isinstance(v,(int,float,np.number)):np.testing.assert_allclose(v,wanted[f],rtol=1e-10,atol=1e-10,err_msg=(group,f))
                else:assert v==wanted[f]
        save(OUT/'reproduction'/(group+'_range_metrics.json'),metrics)
    assert sorted(bad)==lock['expected']['gt20cm_cases'];assert sum(r['group']=='phase_e' and r['status']=='AVAILABLE' for r in results)==865
    for expected in lock['expected']['overshoot']:
        actual=next(r for r in results if key(r['run'],r['frame'])==expected['id']);np.testing.assert_allclose(actual['max_C2_vs_linear_cm'],expected['max_C2_vs_linear_cm'],rtol=1e-10)
    check_baseline();save(OUT/'reproduction/REPRODUCTION_COMPLETE.json',dict(matched=True,time_ns=time.time_ns(),starts=len(tasks),benchmark_available=865,bad_cases=sorted(bad),max_range=lock['expected']['max_range'],curves_seed_prediction_covariance_metrics_recomputed=True,frozen_seed_measurements='Pinned current 0–8m XYZ and labels, unchanged estimator. Future references opened only after reproduced predictions were saved.',max_array_difference=max(r['max_difference'] for r in results),all_expected_overshoots_match=True))
    print('REPRODUCTION MATCHED',len(tasks),'starts; 865 available benchmark; 11 >20cm; identical metrics and overshoot',flush=True)

if __name__=='__main__':main()
