"""Offline evaluation, entered only after verified prediction completion."""
from long_common import *
from evaluation import evaluate_prediction,curve_match
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def transverse(points,steps,ref,seed):
    n=len(points);dist=np.full(n,np.nan);axial=np.full(n,np.nan);valid=np.zeros(n,dtype=bool)
    if len(ref['future']) and len(ref['curve']):
        _,j=cKDTree(ref['future']).query(points);nearest=ref['future'][j]
        _,b=cKDTree(ref['curve']).query(nearest);delta=points-nearest
        local=np.einsum('ij,ijk->ik',delta,ref['curve_basis'][b])
        dist=np.linalg.norm(local[:,1:],axis=1);axial=abs(local[:,0]);_,_,valid=curve_match(points,ref)
        valid &= axial<=.30
    m=steps==0
    if m.any() and len(ref['current']):
        _,j=cKDTree(ref['current']).query(points[m]);d=(points[m]-ref['current'][j])@np.asarray(seed['basis'])
        dist[m]=np.linalg.norm(d[:,1:],axis=1);axial[m]=abs(d[:,0]);valid[m]=True
    dist[~valid]=np.nan
    return dist,axial,valid

def one(path):
    folder=Path(path)
    if (folder/'evaluation.json').exists():return folder.parent.name,'cached'
    marker=check_marker(folder);p=read_prediction(folder);run=p['run'];i=p['start_frame'];name=p['variant']
    if p['seed']['status']!='AVAILABLE':
        save(folder/'evaluation.json',dict(summary=dict(run=run,start_frame=i,variant=name,seed_available=False,evaluation_eligible=False,stop_reason=p['reason']),steps=[],bins=[],evaluated_ns=time.time_ns(),prediction_ns=marker['time_ns']))
        return name,run,i,'UNAVAILABLE'
    ref,info=reference_readonly(run,i)
    with np.load(folder/'provenance.npz') as z:xyz=z['xyz']
    state=p['point_state'];confirmed=state==2;cp=dict(p)
    for f in ARRAYS:
        if f in p and f not in ('curve','indices'):cp[f]=p[f][confirmed]
    cp['indices']=np.arange(confirmed.sum())
    row,sr,bins,point=evaluate_prediction(cp,xyz[confirmed],ref,info)
    row['variant']=name;row['legacy_continuous_reach_5cm']=row.get('continuous_reach_5cm');row['legacy_continuous_reach_10cm']=row.get('continuous_reach_10cm')
    tdist,axial,tvalid=transverse(xyz,p['point_step'],ref,p['seed'])
    ranges=np.linalg.norm(xyz,axis=1);stagepoint=p['point_step'];strict={5:0.,10:0.};alive={5:True,10:True};tx={5:0.,10:0.};txalive={5:True,10:True}
    exact=np.full(len(xyz),np.nan);ev=np.zeros(len(xyz),dtype=bool);exact[confirmed]=point['distance'];ev[confirmed]=point['evaluable']
    steps=[dict(step_index=0,state='CONFIRMED',status='SEED')]+p['steps'];cat={cm:0 for cm in (10,20,50,100)};tcat={cm:0 for cm in cat}
    for step in steps:
        k=step['step_index'];m=(stagepoint==k)&confirmed;v=m&ev;t=m&tvalid
        sstate=step.get('state','CONFIRMED' if step['status'] in ('ACCEPTED','SEED') else 'STOP')
        if sstate in ('GAP','TENTATIVE'):
            alive={a:False for a in alive};txalive={a:False for a in txalive}
        if m.any():
            for cm in strict:
                if alive[cm] and v.sum()>=3 and v.sum()/m.sum()>=.8 and np.mean(exact[v]<=cm/100)>=.95:strict[cm]=float(ranges[m].max())
                else:alive[cm]=False
                if txalive[cm] and t.sum()>=3 and t.sum()/m.sum()>=.8 and np.mean(tdist[t]<=cm/100)>=.95:tx[cm]=float(ranges[m].max())
                else:txalive[cm]=False
            if t.sum()>=3 and t.sum()/m.sum()>=.8:
                for cm in cat:cat[cm]+=int(np.quantile(tdist[t],.95)>cm/100)
        tm=(stagepoint==k)&~confirmed&tvalid
        if tm.sum()>=1:
            for cm in tcat:tcat[cm]+=int(np.quantile(tdist[tm],.95)>cm/100)
        found=next((s for s in sr if s['step_index']==k),None)
        if found is not None:
            found.update(state=sstate,history=step.get('history'),window=step.get('window'),promoted_at=step.get('promoted_at'),original_state=step.get('original_state'),
              transverse_p95=float(np.quantile(tdist[t],.95)) if t.any() else None,transverse_evaluated_points=int(t.sum()),
              axial_p95=float(np.quantile(axial[m&np.isfinite(axial)],.95)) if np.any(m&np.isfinite(axial)) else None)
    row.update({f'max_continuous_correct_range_{cm}cm':strict[cm] if row['evaluation_eligible'] else None for cm in strict})
    row.update({f'transverse_continuous_reach_{cm}cm':tx[cm] if row['evaluation_eligible'] else None for cm in tx})
    row.update({f'confirmed_transverse_segments_gt{cm}cm':n for cm,n in cat.items()})
    row.update({f'tentative_transverse_segments_gt{cm}cm':n for cm,n in tcat.items()})
    for cm in cat:
        row[f'confirmed_transverse_points_gt{cm}cm']=int(np.sum(confirmed&tvalid&(tdist>cm/100)))
        row[f'tentative_transverse_points_gt{cm}cm']=int(np.sum(~confirmed&tvalid&(tdist>cm/100)))
    row.update(max_search_hypothesis_range=p.get('max_search_hypothesis_range',row.get('max_attempted_range')),
      max_confirmed_observed_range=float(ranges[confirmed].max()) if confirmed.any() else 0.,
      max_tentative_range=float(ranges[~confirmed].max()) if np.any(~confirmed) else 0.,
      confirmed_points=int(confirmed.sum()),tentative_points=int((~confirmed).sum()),confirmed_transverse_evaluated=int(np.sum(confirmed&tvalid)),
      confirmed_transverse_p95=float(np.quantile(tdist[confirmed&tvalid],.95)) if np.any(confirmed&tvalid) else None,
      gap_steps=sum(s.get('state')=='GAP' for s in p['steps']),promoted_steps=sum('promoted_at' in s for s in p['steps']),
      wrong_structure_confirmed_suspected=cat[20]>0,readable_file=frames(run)[i]['file'],baseline_identical=p.get('baseline_identical'))
    np.savez_compressed(folder/'point_evaluation.npz',distance=exact,evaluable=ev,range=ranges,transverse=tdist,axial=axial,transverse_valid=tvalid,state=state)
    save(folder/'evaluation.json',dict(summary=row,steps=sr,bins=bins,reference=info,evaluated_ns=time.time_ns(),prediction_ns=marker['time_ns'],legacy_evaluator_sha=sha(ROOT/'MVP/stages/05_contact_rail_marching/src/evaluation.py')))
    if p.get('baseline_identical'):
        old=load(Path(p['baseline_source'])/'evaluation.json')['summary']
        fields=[k for k in old if k.startswith(('continuous_reach','point_matched','anchor_error','max_accepted','point_error','farthest_correct'))]
        for f in fields:assert row.get(f)==old.get(f),(name,run,i,f,row.get(f),old.get(f))
        save(folder/'BASELINE_IDENTICAL.json',dict(prediction=True,metrics=True,fields=fields))
    return name,run,i,row.get('continuous_reach_5cm'),row['max_confirmed_observed_range']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--variants',nargs='+');ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    names=a.variants or [p.name for p in (OUT/a.group).iterdir() if p.is_dir()]
    folders=[str(p.parent) for n in names for p in (OUT/a.group/n).glob('*/PREDICTION_COMPLETE.json')];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for j,result in enumerate(pool.map(one,folders),1):
            if j%50==0 or j==len(folders):print('EVALUATE',a.group,j,len(folders),result,'elapsed',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
