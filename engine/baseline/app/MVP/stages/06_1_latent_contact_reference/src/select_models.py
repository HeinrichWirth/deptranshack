from lc_common import *
from latent_inference import producer_hash
import argparse

def ranking(starts,stations,exclude_run=None):
    allowed=[r for r in starts if r.get('selection_allowed',True) and r['status']=='AVAILABLE' and r['run']!=exclude_run];methods=sorted(set(r['method'] for r in allowed));runs=sorted(set(r['run'] for r in allowed));out=[]
    # Equal run weighting avoids long/dense recordings deciding the ranking.
    # Coverage/width are shown but mean-geometry ranking is kept separate from
    # subsequent development-only uncertainty calibration.
    for name in methods:
        scores=[];rr=[r for r in allowed if r['method']==name]
        for run in runs:
            rows=[r for r in rr if r['run']==run];base=[r for r in allowed if r['run']==run and r['method']=='C0_CURRENT'];pairs=[(r,b) for r in rows for b in base if r['frame']==b['frame'] and r.get('far_p95') is not None and b.get('far_p95') is not None]
            if not pairs:continue
            far=float(np.mean([r['far_p95']/max(.02,b['far_p95']) for r,b in pairs]));cp=[r['cr_p95']/max(.02,b['cr_p95']) for r,b in pairs if r.get('cr_p95') is not None and b.get('cr_p95') is not None];cr=float(np.mean(cp)) if cp else 1.;bad=np.mean([r['far_gt20cm']>0 for r,b in pairs]);phys=np.mean([r['physical_violation_fraction'] for r,b in pairs]);lost=np.mean([1-r['alignment_count']/r['old_station_count'] for r,b in pairs]);score=far+.5*cr+.5*bad+.25*phys+2*lost;scores.append(dict(run=run,score=score,far_ratio=far,cr_ratio=cr,failed_start_fraction=bad,physical_fraction=phys,lost_fraction=lost))
        if scores:out.append(dict(method=name,score=float(np.mean([s['score'] for s in scores])),worst_run_score=float(max(s['score'] for s in scores)),run_scores=scores,far_bad_starts=sum(r.get('far_gt20cm',0)>0 for r in rr),starts=len(rr),far_mean_p95=float(np.mean([r['far_p95'] for r in rr if r.get('far_p95') is not None])),cr_mean_p95=float(np.mean([r['cr_p95'] for r in rr if r.get('cr_p95') is not None]))))
    return sorted(out,key=lambda r:(r['score'],r['method']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['phase_a','phase_b']);a=ap.parse_args();barrier=load(OUT/a.phase/'EVALUATION_COMPLETE.json');starts=load(OUT/a.phase/'per_start.json');stations=load(OUT/a.phase/'per_station.json');rank=ranking(starts,stations);protocol=load(OUT/'protocol.json')
    if a.phase=='phase_a':
        selected=[r['method'] for r in rank if r['method']!='C0_CURRENT'][:10]
        for control in ('C0_CURRENT','C1_POLYLINE','C2_PCHIP'):
            if control not in selected:selected.append(control)
        save(OUT/'selection/phase_a.json',dict(time_ns=time.time_ns(),selected=selected,ranking=rank,excluded_benchmark_cases=[r for r in protocol['phase_a'] if not r['selection_allowed']],criterion='Equal-run mean of far relative p95 + 0.5 CR relative p95 + 0.5 failure fraction + 0.25 physical violation + 2 lost common planes. Fixed denominator floors 2cm; benchmark diagnostics excluded.',evaluation_barrier=barrier))
        write_csv(OUT/'selection/phase_a_ranking.csv',rank);print('PHASE B SELECTED',selected)
    else:
        runs=protocol['split']['development'];loro=[];votes={r['method']:[] for r in rank}
        for run in runs:
            order=ranking(starts,stations,exclude_run=run)
            for pos,r in enumerate(order):votes[r['method']].append(pos);loro.append(dict(excluded_run=run,rank=pos+1,method=r['method'],training_score=r['score'],heldout_score=next((s['score'] for s in next(x for x in rank if x['method']==r['method'])['run_scores'] if s['run']==run),None)))
        for r in rank:r['mean_loro_rank']=float(np.mean(votes[r['method']]));r['max_loro_rank']=int(max(votes[r['method']])+1)
        order=sorted([r for r in rank if r['method']!='C0_CURRENT'],key=lambda r:(r['mean_loro_rank'],r['score']));top=[r['method'] for r in order[:3]]
        # Freeze geometry choices NOW; conditional covariance calibration runs on
        # development residuals next and must be locked before benchmark starts.
        save(OUT/'selection/phase_b.json',dict(time_ns=time.time_ns(),selected=top,best=top[0],ranking=rank,loro=loro));write_csv(OUT/'selection/loro.csv',loro);write_csv(OUT/'selection/phase_b_ranking.csv',rank);print('TOP3',top,'BEST',top[0])

if __name__=='__main__':main()
