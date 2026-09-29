"""Paired method comparison at the development-selected tracking gate."""
from common import *
from calibrate import one
from concurrent.futures import ProcessPoolExecutor,as_completed

def main():
    prior=load(OUT/'calibration/selected.json');base=prior['config'];co=load(OUT/'calibration/sweep_protocol.json')['cohort']
    configs={f'selected_{m}':dict(base,method=m) for m in ('M0','M1','M2','M3','M4')}
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,(r['run'],r['frame'],configs)) for r in co]):rows.extend(f.result())
    csv_write(OUT/'calibration/paired_selected_per_start.csv',rows);summary=[]
    for name,cfg in configs.items():
        rr=[r for r in rows if r['variant']==name and r.get('evaluation_eligible')]
        rm=[np.mean([r['continuous_reach_5cm'] for r in rr if r['run']==run]) for run in SPLIT['development'] if any(r['run']==run for r in rr)]
        bad=float(np.mean([r['wrong_structure_suspected'] for r in rr]));reach=[r['continuous_reach_5cm'] for r in rr]
        summary.append(dict(variant=name,method=cfg['method'],starts=len(rr),score=float(np.mean(rm)-100*bad),wrong_structure_rate=bad,
             median_reach=float(np.median(reach)),reach_p10=float(np.percentile(reach,10)),reach_p90=float(np.percentile(reach,90)),
             anchor_p95=percentiles([r['anchor_error_p95'] for r in rr if r.get('anchor_error_p95') is not None]).get('p95'),runtime_median_ms=float(np.median([r['total_ms'] for r in rr]))))
    summary.sort(key=lambda r:-r['score']);csv_write(OUT/'calibration/paired_selected_summary.csv',summary)
    # Fixed-frame remains the explicitly labelled negative baseline; choose the
    # simplest actual self-aligned method within 5% of the best self-aligned score.
    candidates=[r for r in summary if r['method']!='M0'];best=max(r['score'] for r in candidates)
    tied=[r for r in candidates if r['score']>=best*.95];chosen=min(tied,key=lambda r:int(r['method'][1:]))
    save(OUT/'calibration/selected.json',dict(selected=chosen,config=configs[chosen['variant']],best_score=summary[0],first_sweep=prior))
    print('PAIRED',summary,'FINAL',chosen,flush=True)
if __name__=='__main__':main()
