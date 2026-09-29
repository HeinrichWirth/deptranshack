"""Post-prediction metrics. Never imported by the production rail predictor."""
from rr_common import *
from gt_reference import prepare_run,future_pair_reference,intersect_sections
from seed_input import read_input
from track_geometry import unit,wrapped,cross_axes
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

BINS=[(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,75),(75,100),(100,125),(125,150),(30,50),(50,75)]

def measure(pred,reference,common_B=None):
    C=pred['C'];B=pred['B'] if common_B is None else common_B;q=int(pred['q']);gt,unc,rs=intersect_sections(C,B,reference)
    gt=gt[:,[1,0]] if q>0 else gt
    valid=np.isfinite(gt).all(axis=(1,2));bb=np.full_like(C,np.nan);nn=np.full_like(C,np.nan)
    if valid.any():bb[valid]=unit(q*(gt[valid,0]-gt[valid,1]));nn[valid]=np.cross(B[valid,:,0],bb[valid])
    delta=pred['pair']-gt;lat=np.einsum('nri,ni->nr',delta,bb);vert=np.einsum('nri,ni->nr',delta,nn);err=np.hypot(lat,vert)
    center_err=np.hypot(lat.mean(axis=1),vert.mean(axis=1));alpha_gt=np.arctan2(np.einsum('ni,ni->n',bb,B[:,:,2]),np.einsum('ni,ni->n',bb,B[:,:,1]))
    pred_b,_=cross_axes(pred['B'],pred['state'][:,0]);alpha_pred=np.arctan2(np.einsum('ni,ni->n',pred_b,B[:,:,2]),np.einsum('ni,ni->n',pred_b,B[:,:,1]));alpha_error=wrapped(alpha_pred-alpha_gt)
    mahal=np.full((len(C),2),np.nan);sigma=np.full((len(C),2,2),np.nan)
    for k in np.flatnonzero(valid):
        T=np.column_stack((bb[k],nn[k]))
        for j in (0,1):
            cov=T.T@pred['rail_cov'][k,j]@T;d=np.array([lat[k,j],vert[k,j]])
            mahal[k,j]=d@np.linalg.pinv(cov)@d;sigma[k,j]=np.sqrt(np.maximum(np.diag(cov),0))
    return dict(gt=gt,gt_uncertainty=unc,gt_station=rs,valid=valid,lat=lat,vert=vert,error=err,center_error=center_err,alpha_gt=alpha_gt,alpha_pred_common=alpha_pred,alpha_error=alpha_error,mahal=mahal,sigma_gt_axes=sigma,range=np.linalg.norm(C,axis=1),gauge_gt=np.linalg.norm(gt[:,1]-gt[:,0],axis=1),gauge_pred=np.linalg.norm(pred['pair'][:,1]-pred['pair'][:,0],axis=1))

def rows_for(rec,p,e):
    rows=[]
    for k in range(len(p['s'])):
        row=dict(run=rec['run'],frame=rec['i'],method=rec['method'],cr_input=rec.get('cr_input','CR1_C4'),station=float(p['s'][k]),range_m=float(e['range'][k]),gt_available=bool(e['valid'][k]),cr_state=int(p['cr_state'][k]),support=int(p['support'][k]),alpha_pred=float(e['alpha_pred_common'][k]),alpha_sigma=float(p['alpha_sigma'][k]),alpha_gt=e['alpha_gt'][k],alpha_error_deg=np.rad2deg(e['alpha_error'][k]),center_error=e['center_error'][k],gauge_error=e['gauge_pred'][k]-e['gauge_gt'][k])
        for j,name in enumerate(('near','far')):
            row.update({name+'_lateral':e['lat'][k,j],name+'_vertical':e['vert'][k,j],name+'_error':e['error'][k,j],name+'_mahal2':e['mahal'][k,j],name+'_inside95':bool(e['mahal'][k,j]<=5.991464547) if e['valid'][k] else None,name+'_inside99':bool(e['mahal'][k,j]<=9.210340372) if e['valid'][k] else None,name+'_sigma_lateral':p['sigma_lateral'][k,j],name+'_sigma_vertical':p['sigma_vertical'][k,j]})
        row['both_inside95']=bool(np.all(e['mahal'][k]<=5.991464547)) if e['valid'][k] else None
        row['both_inside99']=bool(np.all(e['mahal'][k]<=9.210340372)) if e['valid'][k] else None
        rows.append(row)
    return rows

def case(task):
    r,group,names=task;run=r['run'];i=r['frame'];root=OUT/group
    source=root/names[0]/key(run,i);ref=future_pair_reference(run,i,source);results=[];rows=[]
    first_record=load(source/'prediction.json');_,common_input=read_input(OUT/first_record['input_folder'])
    save(OUT/'audit/gt_availability'/key(run,i)/'reference.json',dict(future_end=ref['future_end'],future_frames=ref['future_frames'],stations=len(ref['s']),max_range=float(np.linalg.norm(ref['pair'].mean(axis=1),axis=1).max()) if len(ref['s']) else None))
    np.savez_compressed(OUT/'audit/gt_availability'/key(run,i)/'reference.npz',**{k:ref[k] for k in ('s','pair','spread','support')})
    for name in names:
        folder=root/name/key(run,i);marker=load(folder/'PREDICTION_COMPLETE.json')
        for n,h in marker['files'].items():assert sha(folder/n)==h
        rec=load(folder/'prediction.json');summary=dict(run=run,frame=i,method=name,status=rec['status'],cr_input=rec['cr_input'],prediction_ms=rec['prediction_ms'])
        if rec['status']=='AVAILABLE':
            with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
            e=measure(p,ref,common_input['B']);np.savez_compressed(folder/'evaluation.npz',**e);rows.extend(rows_for(rec,p,e))
            for j,n in enumerate(('near','far')):
                summary[n+'_p95']=stats(e['error'][:,j]).get('p95');summary[n+'_gt20cm']=int(np.sum(e['error'][:,j]>.20))
            summary.update(stations=len(p['s']),evaluated=int(e['valid'].sum()),max_predicted_range=float(e['range'].max()),max_evaluated_range=float(e['range'][e['valid']].max()) if e['valid'].any() else None,center_p95=stats(e['center_error']).get('p95'))
        save(folder/'evaluation.json',dict(summary=summary,prediction_ns=marker['time_ns'],evaluated_ns=time.time_ns(),gt_reference='Independent future class1 head components; same CR transverse plane.'))
        results.append(summary)
    return results,rows

def aggregate(rows):
    result=[]
    for name in sorted(set(r['method'] for r in rows)):
        for lo,hi in BINS:
            rr=[r for r in rows if r['method']==name and r['gt_available'] and lo<=r['range_m']<hi]
            if not rr:continue
            row=dict(method=name,lo=lo,hi=hi,starts=len(set((r['run'],r['frame']) for r in rr)),stations=len(rr),both_coverage95=np.mean([r['both_inside95'] for r in rr]),both_coverage99=np.mean([r['both_inside99'] for r in rr]))
            for f in ('near_error','far_error','center_error','alpha_error_deg','gauge_error','near_lateral','near_vertical','far_lateral','far_vertical'):
                for k,v in stats([abs(r[f]) for r in rr]).items():row[f+'_'+k]=v
            for side in ('near','far'):
                for cm in (5,10,20,50,100):row[side+f'_gt{cm}cm']=sum(r[side+'_error']>cm/100 for r in rr)
                for sd in (1,2,3):row[side+f'_coverage_{sd}sigma']=np.mean([r[side+'_mahal2']<=sd*sd for r in rr])
                for ax in ('lateral','vertical'):row[side+f'_median_fullwidth95_{ax}']=float(np.median([2*np.sqrt(5.991464547)*r[side+'_sigma_'+ax] for r in rr]))
            result.append(row)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();barrier=load(OUT/a.group/'INFERENCE_COMPLETE.json');rows=barrier['starts'];names=barrier['methods'];runs=sorted(set(r['run'] for r in rows))
    with ProcessPoolExecutor(max_workers=min(a.workers,4)) as pool:
        for value in pool.map(prepare_run,[(r,a.group) for r in runs]):print('GT PREPARED',value,flush=True)
    summaries=[];stations=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(case,(r,a.group,names)) for r in rows]
        for j,f in enumerate(as_completed(ff),1):
            s,r=f.result();summaries.extend(s);stations.extend(r)
            if j%10==0 or j==len(ff):print('EVALUATE RAILS',a.group,j,len(ff),round(time.perf_counter()-start,1),flush=True)
    write_csv(OUT/a.group/'per_start.csv',summaries);write_csv(OUT/a.group/'per_station.csv',stations);metrics=aggregate(stations);write_csv(OUT/a.group/'range_metrics.csv',metrics)
    save(OUT/a.group/'per_start.json',summaries);save(OUT/a.group/'range_metrics.json',metrics)
    save(OUT/a.group/'EVALUATION_COMPLETE.json',dict(time_ns=time.time_ns(),inference_ns=barrier['time_ns'],starts=len(rows),predictions=len(summaries),station_rows=len(stations)))

if __name__=='__main__':main()
