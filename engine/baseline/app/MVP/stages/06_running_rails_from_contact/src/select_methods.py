"""Development-only selection and LORO corridor calibration; no benchmark reads."""
from rr_common import *
from rail_models import configurations
from run_inference import code_hashes
from collections import Counter
import argparse

def station_rows(group):
    with (OUT/group/'per_station.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[]
        for r in csv.DictReader(f):
            if r['gt_available']!='True':continue
            out={k:(v if k in ('run','method','cr_input') else None if v=='' else v=='True' if v in ('True','False') else float(v)) for k,v in r.items()};rows.append(out)
        return rows

def score_rows(rows):
    blocks=[]
    for lo,hi in ((30,50),(50,75),(75,100)):
        rr=[r for r in rows if lo<=r['range_m']<hi]
        if len(rr)<3:continue
        far=np.array([r['far_error'] for r in rr]);center=np.array([r['center_error'] for r in rr]);coverage=np.mean([r['both_inside95'] for r in rr]);width=np.median([4.895*r['far_sigma_lateral']+4.895*r['far_sigma_vertical'] for r in rr])/2
        value=np.quantile(far,.95)+.5*np.quantile(center,.95)+.2*np.mean(far>.2)+.05*abs(.95-coverage)+.02*width
        blocks.append(float(value))
    return float(np.mean(blocks)) if blocks else None

def rankings(rows,exclude_run=None):
    result=[]
    for name in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==name and r['run']!=exclude_run];runs=sorted(set(r['run'] for r in rr));values=[]
        for run in runs:
            score=score_rows([r for r in rr if r['run']==run])
            if score is not None:values.append(score)
        if values:result.append(dict(method=name,score=.7*float(np.mean(values))+.3*max(values),mean_run_score=float(np.mean(values)),worst_run_score=max(values),scored_runs=len(values)))
    return sorted(result,key=lambda r:r['score'])

def weighted_quantile(values,weights,q):
    values=np.array(values,float);weights=np.array(weights,float);order=np.argsort(values);v=values[order];w=weights[order];return float(np.interp(q,(np.cumsum(w)-.5*w)/w.sum(),v))

def calibration(rows):
    rr=[r for r in rows if 30<=r['range_m']<100 and np.isfinite(r['far_mahal2']) and np.isfinite(r['near_mahal2'])]
    if not rr:return 1.
    by_start=Counter((r['run'],r['frame']) for r in rr);run_starts=Counter(run for run,frame in by_start)
    z=[np.sqrt(max(r['far_mahal2'],r['near_mahal2'])/5.991464547) for r in rr]
    w=[1/(by_start[(r['run'],r['frame'])]*run_starts[r['run']]) for r in rr]
    return max(1.,weighted_quantile(z,w,.95))

def phase_a():
    rows=station_rows('phase_a');ranks=rankings(rows);cfg=load(OUT/'configs.json');gates=load(OUT/'models/phase_a_signal_gates.json')
    selected=['B0_LOCAL','B1_BISHOP'];families=Counter(cfg[n]['family'] for n in selected)
    for row in ranks:
        n=row['method'];family=cfg[n]['family']
        if n in selected or family=='frenet':continue
        if family.startswith('profile') and not gates['profile_supported']:continue
        if families[family]>=2:continue
        selected.append(n);families[family]+=1
        if len(selected)>=8:break
    base=selected.copy();controls=[]
    # Staged offset ablations only for the selected alpha hypotheses.
    for n in base:
        for offset in ('O0','O1','O2'):
            if cfg[n]['offset_model']==offset:continue
            new=n+'__'+offset;cfg[new]=dict(cfg[n],offset_model=offset,offset_q_m_sqrt_m=.001 if offset=='O1' else 0.,parent=n,offset_ablation=True);controls.append(new)
    selected+=controls
    save(OUT/'configs.json',cfg);save(OUT/'phase_b_selection.json',dict(methods=selected,alpha_methods=base,offset_controls=controls,ranking=ranks,signal_gates=gates,benchmark_used=False))
    write_csv(OUT/'phase_a/ranking.csv',ranks);print('PHASE B',base,'+',len(controls),'offset controls',flush=True)

def phase_b():
    rows=station_rows('phase_b');oracle_rows=station_rows('oracle_cr/phase_b');cfg=load(OUT/'configs.json');runs=sorted(set(r['run'] for r in rows));folds=[];calrows=[]
    def combined(exclude=None):
        primary=rankings(rows,exclude);secondary={r['method']:r for r in rankings(oracle_rows,exclude)};out=[]
        for r in primary:
            o=secondary.get(r['method'])
            if o:out.append(dict(r,score=.7*r['score']+.3*o['score'],c4_score=r['score'],oracle_cr_score=o['score'],selection_objective='0.7 causal C4 + 0.3 oracle-CR geometry diagnostic; equal run weights and worst-run penalty.'))
        return sorted(out,key=lambda r:r['score'])
    ranks=combined()
    for held in runs:
        ranking=combined(held);winner=ranking[0]['method'];test=[r for r in rows if r['run']==held and r['method']==winner]
        folds.append(dict(held_run=held,chosen_without_run=winner,test_score=score_rows(test),test_rows=len(test),training_runs=[r for r in runs if r!=held]))
    for n in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==n]
        for held in runs:
            scale=calibration([r for r in rr if r['run']!=held]);test=[r for r in rr if r['run']==held and 30<=r['range_m']<100]
            if not test:continue
            calrows.append(dict(method=n,held_run=held,scale=scale,n=len(test),starts=len(set(r['frame'] for r in test)),coverage95=float(np.mean([max(r['near_mahal2'],r['far_mahal2'])<=5.991464547*scale**2 for r in test])),median_fullwidth95_far_lateral=float(np.median([4.895*r['far_sigma_lateral']*scale for r in test])),median_fullwidth95_far_vertical=float(np.median([4.895*r['far_sigma_vertical']*scale for r in test]))))
    save(OUT/'phase_b/ranking.json',ranks);write_csv(OUT/'phase_b/ranking.csv',ranks);save(OUT/'models/cross_run_selection.json',folds);write_csv(OUT/'cross_run.csv',folds);write_csv(OUT/'uncertainty/corridor_calibration_loro.csv',calrows)
    # Preserve different alpha families in the final short list; variants of the same mean are not independent evidence.
    top=[];families=set()
    for r in ranks:
        n=r['method'];family=cfg[n]['family'];family='constant' if family in ('bishop','legacy','random_walk') else family
        if family in families:continue
        top.append(n);families.add(family)
        if len(top)==3:break
    save(OUT/'models/proposed_final.json',dict(top3=top,ranking=ranks,benchmark_used=False,calibration_scales={n:calibration([r for r in rows if r['method']==n]) for n in ['B0_LOCAL','B1_BISHOP']+top},loro_selection=folds))
    print('PROPOSED FINAL',top,flush=True)

def freeze():
    target=OUT/'research_freeze.json'
    if target.exists():raise RuntimeError('Research freeze already exists; do not overwrite.')
    check_dependencies();proposal=load(OUT/'models/proposed_final.json');cfg=load(OUT/'configs.json');methods=list(dict.fromkeys(['B0_LOCAL','B1_BISHOP']+proposal['top3']))
    assert (OUT/'oracle_cr/phase_b/COMPLETE.json').exists(),'Oracle CR development evaluation is required before freeze.'
    for path in ('stress/COMPLETE.json','oracle_decomposition/extended_phase_b/COMPLETE.json','audit/phase_b/geometry_audit.json'):
        assert (OUT/path).exists(),path
    tests=(OUT/'audit/tests.txt').read_text(encoding='utf-8-sig');assert 'Ran 13 tests' in tests and '\nOK' in tests
    audit=load(OUT/'audit/phase_b/geometry_audit.json');assert audit['nonforward_curves']==0 and audit['nonlocal_close_curves']==0
    for n in methods:cfg[n]['corridor_scale']=proposal['calibration_scales'][n]
    save(OUT/'configs.json',cfg)
    hashes=code_hashes();evaluation={n:sha(STAGE/'src'/n) for n in ('gt_reference.py','evaluate_rails.py','oracle_study.py','extended_oracles.py','geometry_audit.py','roll_diagnostics.py','select_methods.py')}
    save(target,dict(time_ns=time.time_ns(),production_release=False,methods=methods,top3=proposal['top3'],best_development=proposal['top3'][0],code_sha256=hashes,evaluation_sha256=evaluation,config_sha256=sha(OUT/'configs.json'),dependency_lock_sha256=sha(STAGE/'dependency_lock.json'),global_prior_sha256=sha(OUT/'models/global_prior.json'),split=load(OUT/'protocol.json')['split'],selection=proposal,validation='reused research benchmark, not fresh blind validation'))
    print('RESEARCH FREEZE',methods,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['a','b','freeze']);args=ap.parse_args();{'a':phase_a,'b':phase_b,'freeze':freeze}[args.phase]()
