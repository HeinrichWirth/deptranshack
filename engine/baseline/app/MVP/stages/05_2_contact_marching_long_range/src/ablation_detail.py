"""Counters showing which proposed mechanism was actually exercised."""
from long_common import *
from concurrent.futures import ProcessPoolExecutor

def one(path):
    p=load(path);r=dict(variant=p['variant'],run=p['run'],start_frame=p['start_frame'],steps=len(p['steps']),
      support1_confirmed=0,support2_confirmed=0,support1_tentative=0,support2_tentative=0,promoted=0,gaps=0,bridged=0,history_steps=0,
      mild_second_overlap=0,first_overlap_fail=0,veto_candidates=0,beam_hypotheses=len(p.get('hypotheses',[])),long_windows=0,
      past_points=0,points=0,reg_rejected_sources=0,aligned_sources=0,coverage_dropped_sources=0)
    for s in p['steps']:
        r['gaps']+=int(s.get('state')=='GAP');r['bridged']+=int(s.get('state')=='GAP' and s.get('bridged',False));r['promoted']+=int(s.get('promoted_at') is not None)
        r['history_steps']+=int(s.get('history',1)>1);r['long_windows']+=int(s.get('window',8)>8)
        c=s.get('candidate',{});n=c.get('support_unique')
        if n in (1,2):r[f'support{n}_'+('confirmed' if s.get('state')=='CONFIRMED' else 'tentative')]+=1
        for t in s.get('trials',[]):
            for c in t.get('candidates',[]):
                q=c.get('second_overlap',{}).get('p90');r['mild_second_overlap']+=int(q is not None and .03<q<=.05)
                r['veto_candidates']+=int(c.get('current_disagrees',False))
        if s.get('failure_reason')=='OVERLAP_FIRST':r['first_overlap_fail']+=1
    meta=p['input']
    for q in meta.get('roi_audit',[]):
        for s in q['sources']:
            r['reg_rejected_sources']+=int(not s['accepted']);r['aligned_sources']+=int(np.linalg.norm(s['correction_vw'])>1e-8);r['coverage_dropped_sources']+=int(s.get('coverage_selected') is False)
    path=Path(path).parent/'provenance.npz'
    with np.load(path) as z:r['points']=len(z['xyz']);r['past_points']=int(np.sum(z['age_frames']>0))
    return r

def main():
    paths=[str(p) for p in (OUT/'phase_a_screen').glob('*/*/prediction.json')]
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,paths))
    csv_write(OUT/'audit/ablation_mechanisms.csv',rows)
    sums=[]
    for name in sorted(set(r['variant'] for r in rows)):
        rr=[r for r in rows if r['variant']==name];sums.append(dict(variant=name,starts=len(rr),**{k:sum(r[k] for r in rr) for k in rr[0] if k not in ('variant','run','start_frame')}))
    save(OUT/'audit/ablation_mechanisms.json',sums);print('MECHANISMS',len(rows))

if __name__=='__main__':main()
