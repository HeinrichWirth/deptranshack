"""Evaluation-only anchor-source swap in ONE fixed spline family.

Future labels intentionally enter these ORACLE diagnostics after production
development predictions exist. These outputs cannot enter model selection.
"""
from lc_common import *
from latent_inference import read_causal,profile_cache,seed_rebase
from curve_geometry import *
from rail_models import predict
from study_evaluation import align_prediction
from evaluate_rails import measure
from concurrent.futures import ProcessPoolExecutor

def one(r):
    run,i=r['run'],r['frame'];m,a,p,pr=read_causal(run,i)
    if m['status']!='AVAILABLE':return []
    cfg=next(c for c in load(OUT/'protocol.json')['candidates'] if c['name']=='T1_LATENT_PHYSICAL');F=plan_frame(a['initial_basis'],a['world_up_proxy']);s=grid(a['s'][-1]);pa,profiles,timing=profile_cache(run,i,a,p,pr,cfg['profile']);sources={'C4':(a['anchor_knots'],a['anchor_xyz']),'LATENT':(pa['s'],pa['anchors'])};path=ROOT/'results_contact_marching/future_gt'/(key(run,i)+'_reference_v2.npz')
    if path.exists():
        with np.load(path) as z:pts=z['future'];oc=z['curve'];os=z['curve_s']
        ss=[];aa=[];oo=[]
        from study_evaluation import intersect_polyline
        for station in grid(s[-1],2):
            j=int(np.argmin(abs(a['s']-station)));rel=(pts-a['C'][j])@a['B'][j];mask=(abs(rel[:,0])<.5)&(np.linalg.norm(rel[:,1:],axis=1)<.5)
            hit,_=intersect_polyline(a['C'][j],a['B'][j,:,0],oc,os) if len(oc)>1 else (None,None)
            if mask.sum()>=3 and hit is not None:ss.append(station);aa.append(np.median(pts[mask],axis=0));oo.append(hit)
        if len(ss)>=3:sources.update(MEDIAN_CLASS2=(np.array(ss),np.array(aa)),ORACLE_CLASS2=(np.array(ss),np.array(oo)))
    with np.load(OLD_OUT/'phase_b/B4_SPLINE_50__O1'/key(run,i)/'prediction.npz') as z:old={k:z[k] for k in z.files}
    with np.load(OLD_OUT/'audit/gt_availability'/key(run,i)/'reference.npz') as z:ref={k:z[k] for k in z.files}
    railcfg=load(OLD_OUT/'phase_b/B4_SPLINE_50__O1'/key(run,i)/'prediction.json')['config'];rows=[]
    for name,(ss,aa) in sources.items():
        # Equal observation covariance isolates source positions in the SAME
        # family, independent of the production latent-covariance experiment.
        cov=np.repeat(np.eye(3)[None]*.035**2,len(ss),axis=0);C,cc,opt=smooth_spline(ss,aa,cov,s,F,cfg);B=curve_frames(s,C,a['initial_basis']);seed=seed_rebase(m['seed'],s,C,B,m['q'],a['initial_basis']);data={k:a[k] for k in ('world_up_proxy',)};data.update(s=s,C=C,B=B,legacy_B=B,cr_state=np.array([a['cr_state'][np.argmin(abs(a['s']-x))] for x in s]),support=np.interp(s,a['s'],a['support']),position_sigma=np.full(len(s),.035),tangent_sigma=np.full(len(s),np.deg2rad(.1)));pred=predict(seed,data,[dict(informative=False) for _ in s],m['q'],railcfg);aligned,valid,newC,newB=align_prediction(pred,old);e=measure(aligned,ref,old['B']);take=e['valid']&valid&(old['s']>=8);folder=OUT/'oracle_anchor_source'/name/key(run,i);folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'curves.npz',s=s,C=C,pair=pred['pair'],far_error=e['error'][:,1],valid=take);save(folder/'ORACLE_ONLY.json',dict(future_GT_used=name in ('MEDIAN_CLASS2','ORACLE_CLASS2'),selection_allowed=False,family='same robust physically regularized plan/vertical spline, knots 4m, equal anchor covariance',time_ns=time.time_ns()))
        rows.append(dict(run=run,frame=i,anchor_source=name,anchors=len(ss),evaluated=int(sum(take)),far_p95=stats(e['error'][take,1]).get('p95'),far_gt20cm=int(sum(e['error'][take,1]>.2)),future_GT_used=name in ('MEDIAN_CLASS2','ORACLE_CLASS2'),selection_allowed=False))
    return rows

def main():
    assert (OUT/'phase_b/EVALUATION_COMPLETE.json').exists();protocol=load(OUT/'protocol.json');rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,rr in enumerate(pool.map(one,protocol['development']),1):rows+=rr
    write_csv(OUT/'profile_anchor_source_ablation.csv',rows);save(OUT/'oracle_anchor_source/COMPLETE.json',dict(rows=len(rows),time_ns=time.time_ns(),selection_allowed=False));print('ANCHOR SOURCE DIAGNOSTIC',len(rows))

if __name__=='__main__':main()
