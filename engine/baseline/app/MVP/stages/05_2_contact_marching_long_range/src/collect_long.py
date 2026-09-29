"""Post-inference tables, failure decomposition and evaluation-only switch bounds."""
from long_common import *
from scipy.spatial import cKDTree
from contact_rail_step2 import ContactRailDetector
from tracker import TemplateTracker
import argparse

def first_failure_step(p,folder):
    if not (folder/'point_evaluation.npz').exists():return None
    with np.load(folder/'point_evaluation.npz') as z:d=z['distance'];valid=z['evaluable']
    for s in p['steps']:
        if s.get('state') in ('GAP','TENTATIVE'):return s
        mask=(p['point_step']==s['step_index'])&(p['point_state']==2);v=mask&valid
        if mask.any() and not (v.sum()>=3 and v.sum()/mask.sum()>=.8 and np.mean(d[v]<=.05)>=.95):return s
        if s.get('status')=='STOP':return s
    return p['steps'][-1] if p['steps'] else None

def terminal_diagnosis(p,ref,obs):
    st=next((s for s in reversed(p['steps']) if s.get('plane_origin') is not None),{})
    if not st:return dict(category='OTHER',details='No marching frame')
    C=np.asarray(st['plane_origin']);B=np.asarray(st['basis']);W=st.get('window',p['config'].get('window',8));
    q=(obs['xyz']-C)@B;near=(obs['distance']<=.05)&(q[:,0]>=max(0,W-4))&(q[:,0]<=W)
    xyz=obs['xyz'][near];n=len(np.unique(xyz,axis=0));ranges=np.linalg.norm(ref['curve']-C,axis=1)
    j=int(np.argmin(ranges)) if len(ranges) else None
    angle=float(np.rad2deg(np.arccos(np.clip(B[:,0]@ref['curve_basis'][j,:,0],-1,1)))) if j is not None else None
    reason=p['reason'];state=st.get('state');trials=st.get('trials',[]);candidates=[c for t in trials for c in t.get('candidates',[])]
    best=candidates[0] if candidates else {};q90=best.get('second_overlap',{}).get('p90')
    if q90 is None:q90=st.get('first_overlap',{}).get('p90')
    future=(ref['future']-C)@B;gt_here=int(np.sum((future[:,0]>=max(0,W-4))&(future[:,0]<=W)))
    if gt_here<3:category='GT_GAP_OR_END'
    elif n==0:category='NO_POINTS'
    elif n<3:category='WEAK_POINTS'
    elif 'OVERLAP' in reason:category='OVERLAP'
    elif angle is not None and angle>1:category='FRAME'
    elif any(s.get('state')=='GAP' for s in p['steps']):category='GAP'
    elif reason=='COMPETITOR':category='COMPETITOR'
    elif len(np.unique(obs['node'][near]//4))<=1:category='PARTIAL_PROFILE'
    elif candidates and best.get('support_unique',0)>=3:category='ALGORITHMIC_CONSERVATISM'
    else:category='CANDIDATE_MISSING'
    return dict(category=category,terminal_gt_near_returns=int(near.sum()),terminal_gt_near_unique=n,terminal_future_points=gt_here,
      terminal_tangent_error_deg=angle,terminal_overlap_p90=q90,terminal_candidate=best,terminal_step=st['step_index'],
      analysis_scope='All causal history up to F16 near GT, evaluation only; exact selected history may be shallower',plane_origin=C,basis=B,window=W)

def collect(group):
    rows=[];steps=[];hyp=[];failure=[];overlap=[];gaps=[];tentative=[];partial=[]
    paths=sorted((OUT/group).glob('*/*/evaluation.json'))
    for path in paths:
        e=load(path);r=e['summary'];folder=path.parent;name=folder.parent.name;r=dict(r,group=group,folder=folder.relative_to(OUT).as_posix());rows.append(r)
        p=read_prediction(folder)
        for s in e.get('steps',[]):steps.append(dict(s,variant=name,group=group))
        for h in p.get('hypotheses',[]):hyp.append(dict(run=r['run'],start_frame=r['start_frame'],variant=name,group=group,**h))
        for s in p['steps']:
            common=dict(run=r['run'],start_frame=r['start_frame'],variant=name,group=group,step=s['step_index'])
            if s.get('state')=='GAP':gaps.append(dict(common,bridged=s.get('bridged'),reconnected_at=s.get('reconnected_at'),length=s.get('unobserved_length'),search_range=float(np.linalg.norm(s['plane_origin']))))
            if s.get('original_state')=='TENTATIVE':tentative.append(dict(common,promoted_at=s.get('promoted_at'),final_state=s['state'],range=s.get('range_far'),promotion_test=s.get('promotion_test'),original_support=s.get('new_support_n')))
            for trial in s.get('trials',[]):
                for c in trial.get('candidates',[]):
                    overlap.append(dict(common,history=trial['history'],window=trial['window'],rank=c['rank'],state=c['state'],reason=c['reason'],
                      first_p90=c['first_overlap']['p90'],second_p90=c['second_overlap']['p90'],anchor_shift=c['anchor_shift'],unique=c['support_unique'],coverage=c['template_coverage'],
                      source_count=c['source_count'],cells=c['spatial_cells'],template_regions=c['template_regions'],margin=c['competitor_margin'],track_length=c['track_length'],current_disagrees=c['current_disagrees']))
        op=OUT/'observability'/key(r['run'],r['start_frame'])
        if r.get('seed_available') and (op/'near_gt_points.npz').exists():
            with np.load(op/'near_gt_points.npz') as z:obs={k:z[k] for k in z.files}
            ref,info=reference_readonly(r['run'],r['start_frame']);diag=terminal_diagnosis(p,ref,obs)
            first=first_failure_step(p,folder)
            first_diag=terminal_diagnosis(dict(p,steps=[first],reason=first.get('failure_reason') or 'EVALUATION_OR_CONTINUITY_FAILURE'),ref,obs) if first else None
            save(OUT/'failures'/name/(key(r['run'],r['start_frame'])+'.json'),dict(summary=r,diagnosis=diag,first_failure=first,first_failure_diagnosis=first_diag,terminal=p['steps'][-1] if p['steps'] else {},reference=info))
            failure.append(dict(run=r['run'],start_frame=r['start_frame'],variant=name,group=group,**{k:v for k,v in diag.items() if k not in ('terminal_candidate','plane_origin','basis')}))
            ranges=load(op/'ranges.json');r['observable_T']=ranges[0]['max_observable_range_5cm'];r['observable_F16']=ranges[-1]['max_observable_range_5cm'];r['observable_minus_confirmed']=r['observable_F16']-(r.get('max_confirmed_observed_range') or 0)
    base={(r['run'],r['start_frame']):r for r in rows if r['variant']=='B0'}
    for r in rows:
        b=base.get((r['run'],r['start_frame']))
        if b and r.get('evaluation_eligible') and b.get('evaluation_eligible'):r['delta_reach']=r['continuous_reach_5cm']-b['continuous_reach_5cm']
    return rows,steps,hyp,failure,overlap,gaps,tentative

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',default='phase_e_benchmark',nargs='?');a=ap.parse_args()
    rows,steps,hyp,failure,overlap,gaps,tentative=collect(a.group)
    save(OUT/a.group/'analysis_rows.json',rows);csv_write(OUT/'per_start.csv',rows);csv_write(OUT/'per_step.csv',steps);csv_write(OUT/'per_hypothesis.csv',hyp)
    csv_write(OUT/'failure_reasons.csv',failure);csv_write(OUT/'overlap_analysis.csv',overlap);csv_write(OUT/'gap_analysis.csv',gaps);csv_write(OUT/'tentative_analysis.csv',tentative)
    csv_write(OUT/'partial_template_analysis.csv',[r for r in overlap if r['template_regions']<=2])
    base=[r for r in rows if r['variant']=='B0' and r.get('evaluation_eligible')];oracle=[];loss=[]
    for b in base:
        paired=[r for r in rows if r['run']==b['run'] and r['start_frame']==b['start_frame'] and r['variant'] in ('B0','B2','B4','B8')]
        best=max(paired,key=lambda r:r.get('continuous_reach_5cm') or 0)
        oracle.append(dict(run=b['run'],start_frame=b['start_frame'],scope='evaluation-only best complete start; not a deployable switch',best=best['variant'],baseline=b['continuous_reach_5cm'],oracle_reach=best['continuous_reach_5cm'],gain=best['continuous_reach_5cm']-b['continuous_reach_5cm']))
    for name in sorted(set(r['variant'] for r in rows)):
        rr=[r for r in rows if r['variant']==name and 'delta_reach' in r]
        loss.append(dict(variant=name,n=len(rr),**{f'wins_{n}m':sum(r['delta_reach']>n for r in rr) for n in (4,8,20)},**{f'losses_{n}m':sum(r['delta_reach']<-n for r in rr) for n in (4,8,20)}))
    csv_write(OUT/'fusion_switch_oracle.csv',oracle);csv_write(OUT/'loss_analysis.csv',loss)
    # Observed candidates from different methods cannot be spliced into an actual track.
    # This optimistic slab envelope is therefore explicitly kept as a separate bound.
    slab=[]
    for b in base:
        rr=[s for s in steps if s.get('run')==b['run'] and s.get('start_frame')==b['start_frame'] and s['variant'] in ('B0','B2','B4','B8') and s.get('status')=='ACCEPTED']
        for lo in range(8,128,4):
            candidates=[s for s in rr if s.get('range_near') is not None and s['range_near']<lo+4 and s['range_far']>=lo and (s.get('evaluated_fraction') or 0)>=.8 and (s.get('point_p95') or 999)<=.05]
            slab.append(dict(run=b['run'],start_frame=b['start_frame'],lo=lo,hi=lo+4,has_correct_method=bool(candidates),methods=';'.join(sorted(set(s['variant'] for s in candidates))),scope='optimistic independent-slab envelope, not a coherent predicted track'))
    csv_write(OUT/'fusion/oracle_slab_envelope.csv',slab)
    print('COLLECTED',len(rows),'starts',len(steps),'steps',len(hyp),'hypotheses')

if __name__=='__main__':main()
