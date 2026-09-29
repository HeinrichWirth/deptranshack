"""Post-prediction only. Reads pinned future references after an inference barrier."""
from lc_common import *
from evaluate_rails import measure
from track_geometry import unit

BINS=[(0,8),(8,30),(30,50),(50,75),(75,100),(100,125),(125,150)]

def intersect_polyline(C,T,points,station):
    """Intersection with a specified common plane, never independent nearest XYZ."""
    d=(points-C)@T;ids=np.flatnonzero((d[:-1]<=0)&(d[1:]>=0)&(np.diff(d)>1e-9))
    if not len(ids):return None,None
    # If a curve folds, retain forward local candidate and explicitly flag physics.
    k=int(ids[np.argmin(abs(station[ids]-np.median(station[ids])))]) if len(ids)>1 else int(ids[0]);t=-d[k]/(d[k+1]-d[k]);return points[k]*(1-t)+points[k+1]*t,(k,t)

def align_prediction(p,old):
    N=len(old['s']);nearest=np.array([np.argmin(abs(p['s']-s)) for s in old['s']]);out={k:v[nearest].astype(float).copy() if np.ndim(v)>0 and len(v)==len(p['s']) else v for k,v in p.items()};out['s']=old['s'].copy();out['C']=old['C'].copy();out['B']=old['B'].copy();valid=np.zeros(N,bool);newC=np.full_like(old['C'],np.nan);newB=np.full_like(old['B'],np.nan)
    for j,(c,b) in enumerate(zip(old['C'],old['B'])):
        point,kt=intersect_polyline(c,b[:,0],p['C'],p['s'])
        # End plane roundoff tolerance only, not extrapolation into missing range.
        if kt is None:
            dist=(p['C']-c)@b[:,0];idx=int(np.argmin(abs(dist)))
            if abs(dist[idx])<.01:kt=(min(idx,len(p['s'])-2),1. if idx==len(p['s'])-1 else 0.);point=p['C'][idx]
            else:continue
        k,t=kt
        for field,val in p.items():
            if np.ndim(val)>0 and len(val)==len(p['s']) and field not in ('s','C','B'):out[field][j]=(1-t)*val[k]+t*val[k+1]
        pair=[]
        for rail in (0,1):
            hit,_=intersect_polyline(c,b[:,0],p['pair'][:,rail],p['s'])
            if hit is None:hit=out['pair'][j,rail]-b[:,0]*((out['pair'][j,rail]-c)@b[:,0])
            pair.append(hit)
        out['pair'][j]=pair;out['center'][j]=np.mean(pair,axis=0);newC[j]=point;tt=unit((1-t)*p['B'][k,:,0]+t*p['B'][k+1,:,0]);e=unit((1-t)*p['B'][k,:,1]+t*p['B'][k+1,:,1]);e=unit(e-tt*(tt@e));newB[j]=np.column_stack((tt,e,np.cross(tt,e)));valid[j]=True
    # The measurement function uses the common planes, but predicted roll is
    # interpreted in the actual new curve frame for a fair angular error.
    out['B'][valid]=newB[valid]
    return out,valid,newC,newB

def evaluate_case(task):
    r,group,names=task;run=r['run'];i=r['frame'];oldgroup='phase_b' if run in load(OUT/'protocol.json')['split']['development'] else 'phase_e';oldfolder=OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/key(run,i)
    with np.load(oldfolder/'prediction.npz') as z:old={k:z[k] for k in z.files}
    if old:
        with np.load(OLD_OUT/'audit/gt_availability'/key(run,i)/'reference.npz') as z:ref={k:z[k] for k in z.files}
        proxy_path=ROOT/'results_contact_marching/future_gt'/(key(run,i)+'_reference_v2.npz')
        proxy=None
        if proxy_path.exists():
            with np.load(proxy_path) as z:proxy={k:z[k] for k in ('curve','curve_s','curve_basis')}
    rows=[];summaries=[]
    for name in names:
        folder=OUT/group/name/key(run,i);marker=load(folder/'PREDICTION_COMPLETE.json')
        for f,h in marker['files'].items():assert sha(folder/f)==h
        rec=load(folder/'prediction.json');summary=dict(run=run,frame=i,method=name,status=rec['status'],selection_allowed=r.get('selection_allowed',True),**rec['physics'],**rec['timing'])
        if rec['status']!='AVAILABLE':summaries.append(summary);continue
        with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
        with np.load(folder/'curve.npz') as z:curve={k:z[k] for k in z.files}
        if name=='C0_CURRENT':aligned=old;ok=np.ones(len(old['s']),bool);newC=old['C'];newB=old['B']
        else:aligned,ok,newC,newB=align_prediction(p,old)
        ev=measure(aligned,ref,old['B']);ev['valid'] &=ok;N=len(old['s']);crerr=np.full((N,2),np.nan);crgt=np.full((N,3),np.nan);tangent=np.full(N,np.nan);crcov=np.full((N,2,2),np.nan);crmahal=np.full(N,np.nan)
        for j in range(N):
            if not ok[j]:continue
            cv=np.array([[np.interp(old['s'][j],curve['s'],curve['covariance'][:,a,b]) for b in range(3)] for a in range(3)]);crcov[j]=old['B'][j,:,1:].T@cv@old['B'][j,:,1:]
            if proxy is not None and len(proxy['curve'])>1:
                hit,kt=intersect_polyline(old['C'][j],old['B'][j,:,0],proxy['curve'],proxy['curve_s'])
                if hit is not None:
                    crgt[j]=hit;crerr[j]=(newC[j]-hit)@old['B'][j,:,1:];k,t=kt;gt_t=unit(proxy['curve'][k+1]-proxy['curve'][k]);tangent[j]=np.rad2deg(np.arccos(np.clip(gt_t@newB[j,:,0],-1,1)));crmahal[j]=crerr[j]@np.linalg.pinv(crcov[j])@crerr[j]
        ev.update(cr_error=crerr,cr_gt=crgt,cr_tangent_deg=tangent,cr_cov=crcov,cr_mahal2=crmahal,aligned_new_C=newC,aligned_new_B=newB,alignment_available=ok,station=old['s'])
        np.savez_compressed(folder/'evaluation.npz',**ev)
        for j in range(N):
            row=dict(run=run,frame=i,method=name,selection_allowed=r.get('selection_allowed',True),station=float(old['s'][j]),range_m=float(np.linalg.norm(old['C'][j])),alignment_available=bool(ok[j]),valid=bool(ev['valid'][j]),near_error=float(ev['error'][j,0]) if ev['valid'][j] else None,far_error=float(ev['error'][j,1]) if ev['valid'][j] else None,center_error=float(ev['center_error'][j]) if ev['valid'][j] else None,alpha_error_deg=float(np.rad2deg(abs(ev['alpha_error'][j]))) if ev['valid'][j] else None,cr_lateral=float(crerr[j,0]),cr_vertical=float(crerr[j,1]),cr_error=float(np.linalg.norm(crerr[j])),cr_tangent_deg=float(tangent[j]),cr_mahal2=float(crmahal[j]),joint95=bool(np.all(ev['mahal'][j]<=5.991464547)) if ev['valid'][j] else None,joint99=bool(np.all(ev['mahal'][j]<=9.210340372)) if ev['valid'][j] else None,far_fullwidth95_v=float(2*np.sqrt(5.991464547)*ev['sigma_gt_axes'][j,1,0]),far_fullwidth95_w=float(2*np.sqrt(5.991464547)*ev['sigma_gt_axes'][j,1,1]))
            rows.append(row)
        take=ev['valid']&(old['s']>=8);summary.update(evaluated=int(sum(take)),alignment_count=int(sum(ok)),old_station_count=N,max_range=float(np.linalg.norm(p['C'],axis=1).max()),far_p95=stats(ev['error'][take,1]).get('p95'),near_p95=stats(ev['error'][take,0]).get('p95'),center_p95=stats(ev['center_error'][take]).get('p95'),far_gt20cm=int(np.sum(ev['error'][take,1]>.2)),cr_p95=stats(np.linalg.norm(crerr,axis=1)).get('p95'),joint95=float(np.mean(np.all(ev['mahal'][take]<=5.991464547,axis=1))) if take.any() else None)
        save(folder/'evaluation.json',dict(summary=summary,prediction_time_ns=marker['time_ns'],evaluation_time_ns=time.time_ns(),planes='Frozen C0 common transverse planes. All models intersect those planes; no independent closest-point matching.',proxy_note='Future class2 empirical curve, imperfect annotation/registration reference, evaluation only.'))
        summaries.append(summary)
    return summaries,rows

def summarize(rows):
    out=[]
    for method in sorted(set(r['method'] for r in rows)):
        for lo,hi in BINS:
            rr=[r for r in rows if r['method']==method and lo<=r['range_m']<hi];valid=[r for r in rr if r['valid']]
            record=dict(method=method,lo=lo,hi=hi,starts=len(set((r['run'],r['frame']) for r in valid)),stations=len(valid),possible_stations=len(rr),alignment_fraction=np.mean([r['alignment_available'] for r in rr]) if rr else None)
            for f in ('near_error','far_error','center_error','cr_error','cr_lateral','cr_vertical','cr_tangent_deg','alpha_error_deg'):
                vals=[abs(r[f]) for r in rr if r.get(f) is not None];record.update({f+'_'+k:v for k,v in stats(vals).items()})
            for f in ('joint95','joint99','far_fullwidth95_v','far_fullwidth95_w'):record[f]=float(np.mean([r[f] for r in valid])) if valid else None
            crvalid=[r for r in rr if np.isfinite(r['cr_mahal2'])];record['cr_coverage95']=np.mean([r['cr_mahal2']<=5.991464547 for r in crvalid]) if crvalid else None;record['cr_coverage99']=np.mean([r['cr_mahal2']<=9.210340372 for r in crvalid]) if crvalid else None;record['far_gt20cm']=sum(r['far_error']>.2 for r in valid);out.append(record)
    return out
