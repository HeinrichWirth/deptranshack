"""Explicitly non-deployable CR/roll/offset oracle controls, after CR1 inference."""
from rr_common import *
from seed_input import read_input
from track_geometry import *
from rail_models import predict,INPUT_KEYS
from evaluate_rails import measure,rows_for,aggregate
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,copy

def reference_case(run,i):
    path=OUT/'audit/gt_availability'/key(run,i)/'reference.npz'
    with np.load(path) as z:return {k:z[k] for k in z.files}

def oracle_input(meta,a,ref):
    curve=ref['curve'];ss=ref['curve_s'];initial=a['initial_basis'];seed_end=meta['seed']['seed_length']
    if len(curve)<5:return None,None
    # Denoise the GT proxy only within contiguous components; do not smooth across gaps.
    indices=np.r_[0,np.flatnonzero(np.diff(ss)>.8)+1,len(ss)];anchors=[]
    for lo,hi in zip(indices[:-1],indices[1:]):
        segment=curve[lo:hi]
        if len(segment)>=5:segment=gaussian_filter1d(segment,2,axis=0,mode='nearest')
        anchors.extend(segment[np.unique(np.r_[np.arange(0,len(segment),4),len(segment)-1])])
    anchors=np.array(anchors);anchors=anchors[(anchors@initial[:,0])>seed_end+.5]
    if not len(anchors):return None,None
    prefix=sample_curve(dict(knots=a['anchor_knots'],anchors=a['anchor_xyz']),np.array([0.,min(4.,a['s'][-1]),min(seed_end,a['s'][-1])]))
    model=curve_model(np.vstack((prefix,anchors)),initial)
    if model is None:return None,None
    C,B,s=model['C'],model['B'],model['s'];dist,j=cKDTree(a['C']).query(C);observed_distance=cKDTree(curve).query(C)[0]
    states=np.where((s<=seed_end)|(observed_distance<=.8),2,3);profiles=[]
    for k in range(len(s)):
        old=dict(meta['profiles'][int(j[k])]);oldbeta=old.get('beta')
        if oldbeta is not None:
            b,_=cross_axes(a['B'][j[k]],oldbeta);old['beta']=float(np.arctan2(b@B[k,:,2],b@B[k,:,1]))
        old['informative']=bool(old.get('informative',False) and dist[k]<2 and s[k]<=a['s'][-1]);profiles.append(old)
    data=dict(s=s,C=C,B=B,legacy_B=B.copy(),cr_state=states,support=np.where(states==2,10,0),position_sigma=np.full(len(s),.005),tangent_sigma=np.full(len(s),np.deg2rad(.1)),world_up_proxy=a['world_up_proxy'])
    return data,profiles

def one(task):
    row,group,names=task;run=row['run'];i=row['frame'];base=OUT/group/'B1_BISHOP'/key(run,i)
    assert (base/'PREDICTION_COMPLETE.json').exists();record=load(base/'prediction.json');meta,a=read_input(OUT/record['input_folder'])
    if meta['status']!='AVAILABLE':return [],[],[]
    ref=reference_case(run,i)
    with np.load(BASE/'future_gt'/(key(run,i)+'_reference_v2.npz')) as z:crref={k:z[k] for k in z.files}
    oracle,profiles=oracle_input(meta,a,crref);station_rows=[];summaries=[];decomposition=[]
    cfgs=load(OUT/'configs.json');prior_file=load(OUT/'models/global_prior.json');prior=prior_file['leave_run_out'].get(run,prior_file['default'])
    if oracle is not None:
        for name in names:
            cfg=dict(cfgs[name]);cfg['family']='bishop' if cfg['family']=='legacy' else cfg['family']
            p=predict(meta['seed'],oracle,profiles,meta['q'],cfg,prior);dest=OUT/'oracle_cr'/group/name/key(run,i);dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'prediction.npz',**p)
            rec=dict(run=run,i=i,method=name,cr_input='CR0_ORACLE_PROXY',production=False,common_initial_seed='C4 first 8m',future_class1_used_in_prediction=False,future_class2_used=True,gt_proxy_smoothing='1 m Gaussian, within contiguous class2 reference components; knots every ~2m; no smoothing across gaps',B0_note='No frozen marching frame beyond C4: oracle B0 is Bishop and is not used to rank production baseline.')
            save(dest/'prediction.json',rec);e=measure(p,ref,oracle['B']);np.savez_compressed(dest/'evaluation.npz',**e);station_rows.extend(rows_for(rec,p,e))
            summaries.append(dict(run=run,frame=i,method=name,cr_input=rec['cr_input'],max_range=float(np.linalg.norm(p['C'],axis=1).max()),evaluated=int(e['valid'].sum()),near_p95=stats(e['error'][:,0]).get('p95'),far_p95=stats(e['error'][:,1]).get('p95'),center_p95=stats(e['center_error']).get('p95')))
    # Six-way decomposition at identical C4 section planes and C4 horizon.
    for name in names:
        folder=OUT/group/name/key(run,i)
        if not (folder/'prediction.npz').exists():continue
        with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
        if not p:continue
        ev=measure(p,ref,a['B']);GT=ev['gt'];valid=ev['valid'];C=p['C'];BB=a['B'];q=meta['q'];N=len(C)
        # Intersect the oracle contact polyline with the SAME section planes.
        pseudo=dict(s=crref['curve_s'],pair=np.repeat(crref['curve'][:,None,:],2,axis=1),spread=np.zeros((len(crref['curve']),2,3)))
        from gt_reference import intersect_sections
        cc,_,_=intersect_sections(C,BB,pseudo);CC=cc[:,0];valid=valid&np.isfinite(CC).all(axis=1)
        xtruth=np.full((N,5),np.nan)
        for k in np.flatnonzero(valid):xtruth[k]=offsets(CC[k],BB[k],GT[k],q)
        seedx=np.repeat(np.array(meta['seed']['x0'])[None],N,axis=0)
        prodx=p['state'].copy()
        # Express production lateral direction in common C4 section basis.
        prodx[:,0]=ev['alpha_pred_common']
        alternatives={'production_CR_production_alpha':p['pair']}
        rollx=prodx.copy();rollx[:,0]=xtruth[:,0]
        alternatives['production_CR_oracle_alpha']=rails(C,BB,rollx,q)
        offsetsx=prodx.copy();offsetsx[:,1:]=xtruth[:,1:]
        alternatives['production_CR_oracle_offsets']=rails(C,BB,offsetsx,q)
        doublex=seedx.copy();doublex[:,0]=xtruth[:,0]
        alternatives['oracle_CR_oracle_alpha']=rails(CC,BB,doublex,q)
        alternatives['all_oracle']=rails(CC,BB,xtruth,q)
        if oracle is not None:
            of=OUT/'oracle_cr'/group/name/key(run,i)/'prediction.npz'
            with np.load(of) as z:op={k:z[k] for k in z.files}
            proxy=dict(s=op['s'],pair=op['pair'],spread=np.zeros_like(op['pair']))
            match,_,_=intersect_sections(C,BB,proxy);alternatives['oracle_CR_production_alpha']=match
        alternatives['oracle_CR_position_only_production_alpha']=rails(CC,BB,prodx,q)
        arrays={}
        for label,pp in alternatives.items():
            d=pp-GT;trans=d-np.einsum('nri,ni->nr',d,BB[:,:,0])[:,:,None]*BB[:,None,:,0]
            error=np.linalg.norm(trans,axis=2);center=np.linalg.norm(trans.mean(axis=1),axis=1);usable=valid&np.isfinite(error).all(axis=1)
            arrays[label]=pp
            for lo,hi in ((8,150),(30,50),(50,75),(75,100),(100,150)):
                take=usable&(ev['range']>=lo)&(ev['range']<hi)
                if not take.any():continue
                decomposition.append(dict(run=run,frame=i,method=name,variant=label,lo=lo,hi=hi,n=int(take.sum()),near_p95=float(np.quantile(error[take,0],.95)),far_p95=float(np.quantile(error[take,1],.95)),center_p95=float(np.quantile(center[take],.95)),far_gt20cm=int(np.sum(error[take,1]>.2))))
        if valid.any():assert np.nanmax(np.linalg.norm(alternatives['all_oracle'][valid]-GT[valid],axis=2))<1e-7
        dest=OUT/'oracle_decomposition'/group/name/key(run,i);dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'curves.npz',C=C,common_B=BB,gt=GT,valid=valid,oracle_C=CC,oracle_state=xtruth,**arrays)
    return station_rows,summaries,decomposition

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--methods',nargs='+');a=ap.parse_args()
    barrier=load(OUT/a.group/'INFERENCE_COMPLETE.json');load(OUT/a.group/'EVALUATION_COMPLETE.json');names=a.methods or barrier['methods'];tasks=[(r,a.group,names) for r in barrier['starts']];rows=[];summaries=[];decomp=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,t) for t in tasks]
        for j,f in enumerate(as_completed(ff),1):
            r,s,d=f.result();rows.extend(r);summaries.extend(s);decomp.extend(d)
            if j%10==0 or j==len(ff):print('ORACLES',a.group,j,len(ff),flush=True)
    write_csv(OUT/'oracle_cr'/a.group/'per_station.csv',rows);write_csv(OUT/'oracle_cr'/a.group/'per_start.csv',summaries);write_csv(OUT/'oracle_cr'/a.group/'range_metrics.csv',aggregate(rows));save(OUT/'oracle_cr'/a.group/'range_metrics.json',aggregate(rows));save(OUT/'oracle_cr'/a.group/'per_start.json',summaries)
    write_csv(OUT/'oracle_decomposition'/a.group/'metrics.csv',decomp);save(OUT/'oracle_decomposition'/a.group/'metrics.json',decomp)
    save(OUT/'oracle_cr'/a.group/'COMPLETE.json',dict(time_ns=time.time_ns(),starts=len(tasks),methods=names,production=False))

if __name__=='__main__':main()
