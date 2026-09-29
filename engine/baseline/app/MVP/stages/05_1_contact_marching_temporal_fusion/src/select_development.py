"""Development-only selection. No heldout metric reader; distinct from stdlib select."""
from fusion_common import *
from collections import Counter
import argparse

def summaries():
    data={}
    for folder in (OUT/'development').iterdir():
        if folder.is_dir():data[folder.name]=[load(p)['summary'] for p in folder.glob('*/evaluation.json')]
    baseline={(r['run'],r['start_frame']):r for r in data.get('F0',[]) if r.get('evaluation_eligible')};summary=[]
    configs=load(OUT/'configs.json')
    for name,rr in data.items():
        good=[r for r in rr if r.get('evaluation_eligible') and (r['run'],r['start_frame']) in baseline]
        if not good:continue
        values=[r['continuous_reach_5cm'] for r in good];d=[r['continuous_reach_5cm']-baseline[r['run'],r['start_frame']]['continuous_reach_5cm'] for r in good]
        perrun=[np.mean([r['continuous_reach_5cm'] for r in good if r['run']==run]) for run in sorted(set(r['run'] for r in good))]
        wrong=np.mean([r.get('raw_wrong_structure_trigger',False) for r in good]);counts=[]
        for r in good:counts.append(load(OUT/'development'/name/key(r['run'],r['start_frame'])/'prediction.json')['fusion']['actual_frames'])
        summary.append(dict(variant=name,n=len(good),starts=len(rr),median=float(np.median(values)),p90=float(np.percentile(values,90)),max=max(values),score=float(np.mean(perrun)-100*wrong),median_delta=float(np.median(d)),mean_delta=float(np.mean(d)),wins=int(np.sum(np.array(d)>4)),losses=int(np.sum(np.array(d)<-4)),wrong_fraction=float(wrong),mean_history=float(np.mean(counts)),stops=dict(Counter(r['stop_reason'] for r in rr if r.get('seed_available')))))
    return sorted(summary,key=lambda r:-r['score'])

def choose(rows):
    configs=load(OUT/'configs.json');main=[r for r in rows if r['variant']!='F0' and not configs[r['variant']].get('diagnostic') and not configs[r['variant']].get('tau')]
    best=max(r['score'] for r in main);eligible=[r for r in main if r['score']>=best-abs(best)*.05]
    return min(eligible,key=lambda r:(r['mean_history'],configs[r['variant']].get('representation','RAW_CONCAT')!='RAW_CONCAT',r['variant']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',action='store_true');a=ap.parse_args();rows=summaries();csv_write(OUT/'calibration/method_ablation.csv',rows);save(OUT/'calibration/summary.json',rows)
    print(json.dumps(rows,ensure_ascii=False,indent=1),flush=True)
    if a.lock:
        assert not (OUT/'research_lock.json').exists();selected=choose(rows);cfg=load(OUT/'configs.json');dist=[r for r in rows if r['variant'].startswith('H') and '_' not in r['variant']];best_distance=max(dist,key=lambda r:r['score'])['variant']
        active=list(dict.fromkeys(['F0','F2','F3','F4',best_distance,selected['variant'],'M2_F0','M2_BEST','FUSED_BOOTSTRAP']))
        f4=next(r for r in rows if r['variant']=='F4')
        if f4['wins']>f4['losses'] and f4['mean_delta']>0:active+=['F6','F8']
        cfg['M2_F0']=dict(frames=1,method='M2',diagnostic=True);cfg['M2_BEST']=dict(cfg[selected['variant']],method='M2',diagnostic=True)
        cfg['FUSED_BOOTSTRAP']=dict(cfg[selected['variant']],fused_bootstrap=True,diagnostic=True)
        save(OUT/'configs.json',cfg)
        inference=['fusion_common.py','fusion.py','study.py','weighting.py','weighted_tracker.py']
        save(OUT/'research_lock.json',dict(time_ns=time.time_ns(),production_release=False,selected=selected['variant'],best_distance=best_distance,heldout_variants=active,
            config_sha256=sha(OUT/'configs.json'),inference_sha256={'src/'+p:sha(STAGE/'src'/p) for p in inference},baseline_dependencies=load(OUT/'audit/previous_stage_sha.json'),selection_summary=selected))
        print('LOCK',selected['variant'],active,flush=True)
if __name__=='__main__':main()
