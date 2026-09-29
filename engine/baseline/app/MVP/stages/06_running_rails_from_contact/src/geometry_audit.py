"""Post-prediction audits; none of these data enter model selection or inference."""
from rr_common import *
from seed_input import read_input
from oracle_study import reference_case
from gt_reference import intersect_sections
from track_geometry import unit,offsets
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
import argparse

def case(task):
    row,group,name=task;run=row['run'];i=row['frame'];folder=OUT/group/name/key(run,i);record=load(folder/'prediction.json')
    if record['status']!='AVAILABLE':return dict(availability=[dict(run=run,frame=i,status=record['status'])],offsets=[],overlap=[],curves=[],side=[])
    with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
    with np.load(folder/'evaluation.npz') as z:e={k:z[k] for k in z.files}
    meta,a=read_input(OUT/record['input_folder']);ref=reference_case(run,i);q=int(p['q']);valid=e['valid'];ss=p['s'];rows=[];overlap=[];curves=[];side=[]
    cc=OUT/'oracle_decomposition'/group/name/key(run,i)/'curves.npz'
    if cc.exists():
        with np.load(cc) as z:
            for k in np.flatnonzero(z['valid']):
                x=z['oracle_state'][k];pred=p['state'][k]
                rows.append(dict(run=run,frame=i,station=ss[k],range_m=e['range'][k],alpha_gt=x[0],d_gt=x[1],h_near_gt=x[2],g_gt=x[3],h_far_gt=x[4],d_pred=pred[1],h_near_pred=pred[2],g_pred=pred[3],h_far_pred=pred[4]))
    obs=meta['seed']['observations'];C=np.array([o['cr'] for o in obs]);B=np.array([a['B'][np.argmin(abs(a['s']-o['s']))] for o in obs]);gt,unc,rs=intersect_sections(C,B,ref);gt=gt[:,[1,0]] if q>0 else gt
    for k in range(len(obs)):
        if not np.isfinite(gt[k]).all():continue
        delta=np.array(obs[k]['heads'])-gt[k];delta-=np.outer(delta@B[k,:,0],B[k,:,0]);errors=np.linalg.norm(delta,axis=1)
        overlap.append(dict(run=run,frame=i,station=obs[k]['s'],near_difference=errors[0],far_difference=errors[1],future_fusion_spread_near=unc[k,0],future_fusion_spread_far=unc[k,1],interpretation='Combined current/future anchor + registration consistency, not isolated sensor accuracy.'))
    for j,label in enumerate(('near','far','center')):
        curve=p['pair'][:,j] if j<2 else p['center'];d=np.gradient(curve,ss,axis=0);dd=np.gradient(d,ss,axis=0);norm=np.linalg.norm(d,axis=1);curvature=np.linalg.norm(np.cross(d,dd),axis=1)/np.maximum(norm,1e-8)**3;progress=np.einsum('ni,ni->n',d,a['B'][:,:,0])
        close=cKDTree(curve).query_pairs(.02);nonlocal_pairs=[(u,v) for u,v in close if abs(ss[u]-ss[v])>2]
        curves.append(dict(run=run,frame=i,method=name,curve=label,curvature_p95=stats(curvature).get('p95'),max_curvature=float(curvature.max()),min_forward_progress=float(progress.min()),nonforward_stations=int((progress<=0).sum()),nonlocal_sample_pairs_within_2cm=len(nonlocal_pairs),gauge_ptp=float(np.ptp(e['gauge_pred'])),max_range=float(e['range'].max()),inspection='1 m numerical samples; C2 interpolation and roll join tested separately, not a proof against sub-sample intersections.'))
    # Future class2 is evaluation only. Inspect both sides, including points rejected
    # from the old single-side oracle curve; never feeds a side event into STEP6.
    path=BASE/'future_gt'/(key(run,i)+'_reference_v2.npz')
    if len(ref['s'])>=3 and path.exists():
        with np.load(path) as z:future=z['allfuture']
        middle=ref['pair'].mean(axis=1);dist,j=cKDTree(middle).query(future);b=unit(ref['pair'][:,1]-ref['pair'][:,0]);lateral=np.einsum('ni,ni->n',future-middle[j],b[j]);station=ref['s'][j]
        keep=(dist<3.5)&(abs(lateral)>.95)&(abs(lateral)<2.)
        for lo in range(0,150,10):
            take=keep&(station>=lo)&(station<lo+10);n=int(take.sum())
            if n<10:continue
            opp=float(np.mean(np.sign(lateral[take])!=q));side.append(dict(run=run,frame=i,lo=lo,hi=lo+10,reference_points=n,opposite_side_fraction=opp,evaluation_switch_candidate=opp>.7,c4_prediction_reaches_bin=bool(e['range'].max()>lo),c4_confirmed_side_event=False,policy='No confirmed switch event in frozen C4; future side is evaluation only.'))
    availability=[dict(run=run,frame=i,status='AVAILABLE',predicted_stations=len(ss),evaluated_stations=int(valid.sum()),max_prediction_range=float(e['range'].max()),max_evaluated_range=float(e['range'][valid].max()) if valid.any() else None,max_future_reference_range=float(np.linalg.norm(ref['pair'].mean(axis=1),axis=1).max()) if len(ref['s']) else None,GT_spread_p50=stats(e['gt_uncertainty']).get('p50'),GT_spread_p95=stats(e['gt_uncertainty']).get('p95'),gap_stations=int((p['cr_state']!=2).sum()))]
    return dict(availability=availability,offsets=rows,overlap=overlap,curves=curves,side=side)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args();barrier=load(OUT/args.group/'INFERENCE_COMPLETE.json');proposal=load(OUT/'research_freeze.json') if (OUT/'research_freeze.json').exists() else load(OUT/'models/proposed_final.json');name=proposal.get('best_development',proposal['top3'][0]);allrows={k:[] for k in ('availability','offsets','overlap','curves','side')}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j,out in enumerate(pool.map(case,[(r,args.group,name) for r in barrier['starts']]),1):
            for k in allrows:allrows[k]+=out[k]
            if j%50==0:print('GEOMETRY AUDIT',args.group,j,flush=True)
    root=OUT/'audit'/args.group
    for k,rows in allrows.items():write_csv(root/(k+'.csv'),rows)
    save(root/'geometry_audit.json',dict(time_ns=time.time_ns(),method=name,starts=len(barrier['starts']),row_counts={k:len(v) for k,v in allrows.items()},overlap_near=stats([r['near_difference'] for r in allrows['overlap']]),overlap_far=stats([r['far_difference'] for r in allrows['overlap']]),side_candidate_bins=sum(r['evaluation_switch_candidate'] for r in allrows['side']),nonforward_curves=sum(r['nonforward_stations']>0 for r in allrows['curves']),nonlocal_close_curves=sum(r['nonlocal_sample_pairs_within_2cm']>0 for r in allrows['curves'])))
    print('GEOMETRY AUDIT COMPLETE',args.group,flush=True)

if __name__=='__main__':main()
