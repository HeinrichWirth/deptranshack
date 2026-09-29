"""Development-only staged selection and cross-run robustness. No benchmark reader."""
from long_common import *
from configurations import combinations,configurations
import argparse

def read_group(group):
    assert group in ('phase_a_screen','phase_b_full_dev','phase_d_combinations')
    data={}
    for folder in (OUT/group).iterdir():
        if folder.is_dir():data[folder.name]=[load(p)['summary'] for p in folder.glob('*/evaluation.json')]
    return data

def summarize(data,exclude=None):
    baseline={(r['run'],r['start_frame']):r for r in data['B0'] if r.get('evaluation_eligible') and r['run']!=exclude};rows=[]
    for name,allrows in data.items():
        rr=[r for r in allrows if (r['run'],r['start_frame']) in baseline and r.get('evaluation_eligible')]
        if not rr:continue
        reach=np.array([r.get('continuous_reach_5cm') or 0 for r in rr]);delta=reach-[baseline[r['run'],r['start_frame']]['continuous_reach_5cm'] for r in rr]
        runrows=[]
        for run in sorted(set(r['run'] for r in rr)):
            ids=np.array([r['run']==run for r in rr]);r=reach[ids];d=delta[ids];bad=np.array([(x.get('confirmed_transverse_segments_gt20cm') or 0)>0 for x in rr])[ids]
            utility=2*np.mean(r>=50)+3*np.mean(r>=75)+5*np.mean(r>=100)+np.mean(np.minimum(r,120))/120-10*np.mean(bad)-2*np.mean(d<-4)
            runrows.append(dict(run=run,n=int(ids.sum()),utility=float(utility),p50=float(np.mean(r>=50)),p75=float(np.mean(r>=75)),p100=float(np.mean(r>=100)),loss4=float(np.mean(d<-4)),catastrophic=int(bad.sum())))
        gt100=np.array([(r.get('max_GT_available_range') or 0)>=100 for r in rr])
        row=dict(variant=name,n=len(rr),runs=len(runrows),starts=len(allrows),median=float(np.median(reach)),p90=float(np.quantile(reach,.9)),p95=float(np.quantile(reach,.95)),max=float(reach.max()),
          score=float(np.mean([r['utility'] for r in runrows])),median_run_utility=float(np.median([r['utility'] for r in runrows])),worst_run_p50=min(r['p50'] for r in runrows),worst_run_p75=min(r['p75'] for r in runrows),
          catastrophic_starts=sum(r['catastrophic'] for r in runrows),catastrophic_segments=sum(r.get('confirmed_transverse_segments_gt20cm') or 0 for r in rr),
          gt100_n=int(gt100.sum()),p100_conditional_gt=float(np.mean(reach[gt100]>=100)) if gt100.any() else None,
          confirmed_observed100=int(sum((r.get('max_confirmed_observed_range') or 0)>=100 for r in rr)),
          strict100=int(sum((r.get('max_continuous_correct_range_5cm') or 0)>=100 for r in rr)),
          transverse100=int(sum((r.get('transverse_continuous_reach_5cm') or 0)>=100 for r in rr)),
          run_metrics=runrows)
        row.update({f'p{h}':float(np.mean(reach>=h)) for h in (30,40,50,60,75,80,100,120)})
        row.update({f'wins{h}':int(np.sum(delta>h)) for h in (4,8,20)})
        row.update({f'losses{h}':int(np.sum(delta<-h)) for h in (4,8,20)})
        rows.append(row)
    return sorted(rows,key=lambda r:(-r['score'],-r['median_run_utility'],-r['worst_run_p50'],r['catastrophic_starts'],r['variant']))

def fingerprint(rows):
    return tuple((r['run'],r['start_frame'],r.get('continuous_reach_5cm'),r.get('max_confirmed_observed_range'),r.get('confirmed_transverse_segments_gt20cm')) for r in sorted(rows,key=lambda r:(r['run'],r['start_frame'])))

def rank_names(summary,n,data=None,exclude=None):
    names=[];seen=set()
    for r in summary:
        name=r['variant']
        if name in ('B0','B2','B4','B8','A0','TENTATIVE1'):continue
        if data is not None:
            sig=fingerprint([x for x in data[name] if x['run']!=exclude])
            if sig in seen:continue
            seen.add(sig)
        names.append(name)
        if len(names)>=n:break
    return names

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['a','b','freeze']);a=ap.parse_args()
    if a.phase=='a':
        data=read_group('phase_a_screen');expected=len(load(OUT/'audit/screen.json'))
        assert all(len(data.get(n,[]))==expected for n in configurations()),'Incomplete screen'
        rows=summarize(data);save(OUT/'phase_a_screen/summary.json',rows);csv_write(OUT/'phase_a.csv',[{k:v for k,v in r.items() if k!='run_metrics'} for r in rows])
        primary=rank_names(rows,15,data);folds={run:rank_names(summarize(data,run),15,data,run) for run in SPLIT['development']}
        # Union prevents screening on the left-out run from contaminating its cross-run selection.
        chosen=list(dict.fromkeys(['B0','B2','B4','B8','A0']+primary+[n for v in folds.values() for n in v]))
        save(OUT/'phase_b_selection.json',dict(variants=chosen,primary15=primary,fold_shortlists=folds,reason='Top15 distinct outcome fingerprints plus union of development-run-excluded Phase A shortlists; fingerprint deduplication also excludes the left-out run',time_ns=time.time_ns()))
        print('PHASE A TOP',[(r['variant'],round(r['score'],4),r['p50'],r['p75'],r['max'],r['catastrophic_starts']) for r in rows[:20]])
        print('PHASE B',len(chosen),chosen)
        return
    data=read_group('phase_b_full_dev');expected=len(load(OUT/'audit/cohort.json')['development'])
    selected=load(OUT/'phase_b_selection.json');assert all(len(data.get(n,[]))==expected for n in selected['variants']),'Incomplete Phase B'
    data={k:v for k,v in data.items() if k in selected['variants']}
    rows=summarize(data);save(OUT/'phase_b_full_dev/summary.json',rows);csv_write(OUT/'phase_b.csv',[{k:v for k,v in r.items() if k!='run_metrics'} for r in rows])
    top8=rank_names(rows,8,data);cross=[]
    for row in rows:
        if row['variant'] in top8:
            cross.extend(dict(variant=row['variant'],**r) for r in row['run_metrics'])
    csv_write(OUT/'cross_run.csv',cross);save(OUT/'phase_c_cross_run/top8.json',top8)
    if a.phase=='b':
        config=load(OUT/'configs.json');config.update(combinations());save(OUT/'configs.json',config)
        save(OUT/'phase_d_selection.json',dict(variants=list(combinations()),time_ns=time.time_ns()))
        print('PHASE B TOP',[(r['variant'],round(r['score'],4),r['p50'],r['p75'],r['max'],r['catastrophic_starts']) for r in rows[:15]])
        print('PHASE D',list(combinations()));return
    combos=read_group('phase_d_combinations');assert all(len(combos.get(n,[]))==expected for n in combinations()),'Incomplete combinations'
    data.update(combos);combo_summary=summarize(dict(combos,B0=data['B0']));save(OUT/'phase_d_combinations/summary.json',combo_summary);csv_write(OUT/'combinations.csv',[{k:v for k,v in r.items() if k!='run_metrics'} for r in combo_summary])
    rows=summarize(data);folds=[]
    for run in SPLIT['development']:
        allowed=set(selected['fold_shortlists'][run])|set(combinations())|{'B0','B2','B4','B8','A0'}
        training=summarize({k:v for k,v in data.items() if k in allowed},run);winner=rank_names(training,1,data,run)[0]
        held=next((r for r in next(x for x in rows if x['variant']==winner)['run_metrics'] if r['run']==run),None)
        folds.append(dict(left_out=run,selected_without_this_run=winner,training_score=next(r['score'] for r in training if r['variant']==winner),held_run_metrics=held))
    save(OUT/'phase_c_cross_run/leave_one_run_out.json',folds)
    unique=[];seen=set()
    for r in rows:
        name=r['variant']
        if name in ('B0','B2','B4','B8','A0'):continue
        sig=fingerprint(data[name])
        if sig in seen:continue
        seen.add(sig);unique.append(name)
        if len(unique)==3:break
    assert not (OUT/'research_freeze.json').exists()
    freeze=dict(time_ns=time.time_ns(),production_release=False,top3=unique,best_development=unique[0],code_sha256=code_hashes(),config_sha256=sha(OUT/'configs.json'),
      development_summary=[r for r in rows if r['variant'] in unique],all_ranking=rows,leave_one_run_out=folds,
      benchmark_warning='reused benchmark, not fresh external validation',selection='Prespecified equal-run long-range/safety/loss utility; median run and worst-run P50 tie breaks; three distinct development outcome fingerprints; no benchmark input')
    save(OUT/'research_freeze.json',freeze);print('FROZEN TOP3',unique)
    print([(r['variant'],round(r['score'],4),r['p50'],r['p75'],r['p100'],r['max']) for r in rows[:15]])

if __name__=='__main__':main()
