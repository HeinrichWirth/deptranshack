"""Compact one-factor development sweep. The held-out folders are never read."""
from common import *
from tracker import DEFAULT,predict
from run_study import seed_geometry,save_prediction
from evaluation import reference,evaluate_prediction,load_prediction
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def variants():
    out={}
    def add(name,**kwargs):out[name]=dict(DEFAULT,**kwargs)
    for m in ('M0','M1','M2','M3','M4'):add(m,method=m)
    for w in (6.,8.,10.,12.):
        for frac in (.25,.5):add(f'window_{w:g}_{frac:g}',window=w,advance=w*frac)
    for tail in (4.,6.,8.,10.):add(f'tail_{tail:g}',tail=tail)
    for bs in (.20,.30,.50):add(f'bin_{bs:g}',bin_size=bs)
    for obj in ('trace','logdet','ellipse90','ellipse95','mad'):add('objective_'+obj,method='M3',objective=obj)
    for bound in (1.,2.,3.,5.):add(f'refine_{bound:g}',method='M3',refine_bound=bound)
    for bound in (1.,2.,3.):add(f'roll_{bound:g}',method='M4',roll_bound=bound)
    for bound in (1.,2.,3.,5.):add(f'jump_{bound:g}',orientation_bound=bound)
    for gate in (.03,.05,.10,.15,.20):add(f'gate_{gate:g}',gate=gate)
    add('fixed_anchor',anchor_mode='fixed')
    for plane in ('start','midpoint','end'):add('plane_'+plane,plane=plane)
    add('reset_up',transport='reset_up')
    # Deduplicate configurations, retaining aliases in the saved protocol.
    unique={};aliases={}
    for name,cfg in out.items():
        existing=next((k for k,v in unique.items() if v==cfg),None)
        if existing:aliases[name]=existing
        else:unique[name]=cfg
    return unique,aliases

def one(task):
    run,i,configs=task;xyz,seed=seed_geometry(run,i);template=ContactRailDetector().template
    for name,cfg in configs.items():
        folder=OUT/'ablation'/name/key(run,i)
        if not (folder/'PREDICTION_COMPLETE.json').exists():save_prediction(folder,predict(xyz,seed,template,cfg),run,i)
    # Every prediction is complete before opening evaluation evidence.
    ref,info=reference(run,i,'development');rows=[]
    for name in configs:
        folder=OUT/'ablation'/name/key(run,i)
        if (folder/'evaluation.json').exists():ev=load(folder/'evaluation.json')
        else:
            pred=load_prediction(folder);r,s,b,p=evaluate_prediction(pred,xyz,ref,info)
            ev=dict(version=2,summary=r,steps=s,bins=b,reference=info);save(folder/'evaluation.json',ev)
        rows.append(dict(variant=name,**ev['summary']))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--select',action='store_true');args=ap.parse_args()
    configs,aliases=variants();co=load(OUT/'audit/cohort.json');chosen=[]
    for run in SPLIT['development']:
        rr=[r for r in co if r['run']==run];chosen += [rr[j] for j in np.unique(np.linspace(0,len(rr)-1,4).astype(int))]
    save(OUT/'calibration/sweep_protocol.json',dict(configs=configs,aliases=aliases,cohort=chosen,
         selection='Maximize equal-run mean continuous correct reach at 5cm; exclude reference unavailable; penalize >20cm wrong-structure suspicion; prefer simpler method within 5%. One-factor sweep, no held-out tuning.'))
    t=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(one,(r['run'],r['frame'],configs)) for r in chosen]
        for j,f in enumerate(as_completed(fs)):
            rows.extend(f.result());print('SWEEP',j+1,len(fs),round(time.perf_counter()-t,1),flush=True)
    csv_write(OUT/'calibration/sweep_per_start.csv',rows);summary=[]
    for name,cfg in configs.items():
        rr=[r for r in rows if r['variant']==name and r.get('evaluation_eligible')];reach=[r['continuous_reach_5cm'] for r in rr]
        runmeans=[np.mean([r['continuous_reach_5cm'] for r in rr if r['run']==run]) for run in SPLIT['development'] if any(r['run']==run for r in rr)]
        bad=np.mean([r['wrong_structure_suspected'] for r in rr]) if rr else 1.
        score=float(np.mean(runmeans)-100*bad) if runmeans else -1e9
        summary.append(dict(variant=name,method=cfg['method'],starts=len(rr),score=score,wrong_structure_rate=bad,
            median_reach=float(np.median(reach)) if reach else None,reach_p10=float(np.percentile(reach,10)) if reach else None,reach_p90=float(np.percentile(reach,90)) if reach else None,
            anchor_p95=percentiles([r['anchor_error_p95'] for r in rr if r.get('anchor_error_p95') is not None]).get('p95'),
            runtime_median_ms=percentiles([r['total_ms'] for r in rr]).get('p50')))
    summary.sort(key=lambda r:-r['score']);csv_write(OUT/'method_ablation.csv',summary);save(OUT/'calibration/summary.json',summary)
    best=summary[0];tied=[r for r in summary if r['score']>=best['score']*.95]
    order={'M0':0,'M1':1,'M2':2,'M3':3,'M4':4};best=min(tied,key=lambda r:(order[r['method']],r['runtime_median_ms']))
    cfg=configs[best['variant']];save(OUT/'calibration/selected.json',dict(selected=best,config=cfg,best_score=summary[0]))
    print('SELECTED',best,cfg,flush=True)
if __name__=='__main__':main()
