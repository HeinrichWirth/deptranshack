"""Post-inference only. Match all predictions on reference transverse planes."""
from . import ROOT,OUT
import long_common as lc
import numpy as np
from gt_reference import future_pair_reference,intersect_sections
import argparse,time


def read(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def matched(a,b,gt):
    heads,unc,station=intersect_sections(a['C'],a['B'],gt)
    if int(a['q'])>0:heads=heads[:,[1,0]]
    # The same frozen section intersection routine, now applied to a numeric
    # prediction polyline, supplies predictions in these exact same planes.
    bp,_,_=intersect_sections(a['C'],a['B'],dict(s=b['s'],pair=b['pair'],spread=np.zeros_like(b['pair'])))
    bcr,_,_=intersect_sections(a['C'],a['B'],dict(s=b['s'],pair=np.repeat(b['C'][:,None],2,axis=1),spread=np.zeros_like(b['pair'])))
    valid=np.isfinite(heads).all((1,2))&np.isfinite(bp).all((1,2))
    delta_a=a['pair']-heads;delta_b=bp-heads
    normal=a['B'][:,:,0]
    def err(d):
        if d.ndim==3:d=d-np.einsum('nri,ni->nr',d,normal)[:,:,None]*normal[:,None]
        else:d=d-np.einsum('ni,ni->n',d,normal)[:,None]*normal
        return np.linalg.norm(d,axis=-1)
    return dict(valid=valid,reference=err(delta_a),variant=err(delta_b),center_reference=err(delta_a.mean(1)),center_variant=err(delta_b.mean(1)),
        cr_change=np.linalg.norm(bcr[:,0]-a['C'],axis=1),variant_cr=bcr[:,0],ranges=np.linalg.norm(a['C'],axis=1),gt=heads,variant_pair=bp)


def evaluate_folder(root):
    barrier=lc.load(root/'INFERENCE_COMPLETE.json');stations=[];cases=[]
    for row in barrier['rows']:
        run=row['run'];index=row['frame'];folder=root/run/f'{index:06d}'
        a=read(folder/'reference.npz') if (folder/'reference.npz').exists() else None
        b=read(folder/'prediction.npz') if (folder/'prediction.npz').exists() else None
        case=dict(run=run,frame=index,reference_available=a is not None,variant_available=b is not None,
                  reference_horizon=0 if a is None else float(np.linalg.norm(a['C'],axis=1).max()),variant_horizon=0 if b is None else float(np.linalg.norm(b['C'],axis=1).max()),wrong_side=a is not None and b is not None and int(a['q'])!=int(b['q']))
        cases.append(case)
        if a is None or b is None:continue
        ref=future_pair_reference(run,index,folder);e=matched(a,b,ref)
        case['new_crossing_stations']=int(np.sum(e['valid'] & (np.einsum('ij,ij->i',e['variant_pair'][:,0]-e['variant_pair'][:,1],a['pair'][:,0]-a['pair'][:,1])<=0)))
        crpath=ROOT/'results_contact_marching/future_gt'/f'{run}__{index:06d}_reference_v2.npz'
        e['contact_reference']=np.full(len(a['C']),np.nan);e['contact_variant']=np.full(len(a['C']),np.nan)
        if crpath.exists():
            cr=read(crpath);pair=np.repeat(cr['curve'][:,None],2,axis=1)
            # Preserve the old 0.8m maximum observed GT interval.
            ss=np.r_[0,np.cumsum(np.where(np.diff(cr['curve_s'])<=.8,np.diff(cr['curve_s']),2.))]
            gtcr,_,_=intersect_sections(a['C'],a['B'],dict(s=ss,pair=pair,spread=np.zeros_like(pair)))
            predcr,_,_=intersect_sections(a['C'],a['B'],dict(s=b['s'],pair=np.repeat(b['C'][:,None],2,axis=1),spread=np.zeros_like(b['pair'])))
            e['contact_reference']=np.linalg.norm(a['C']-gtcr[:,0],axis=1)
            e['contact_variant']=np.linalg.norm(predcr[:,0]-gtcr[:,0],axis=1)
            both=np.isfinite(e['contact_reference'])&np.isfinite(e['contact_variant'])
            e['contact_reference'][~both]=np.nan;e['contact_variant'][~both]=np.nan
        np.savez_compressed(folder/'evaluation.npz',**e)
        for k in np.flatnonzero(e['valid']):
            stations.append(dict(run=run,frame=index,range_m=e['ranges'][k],near_reference=e['reference'][k,0],far_reference=e['reference'][k,1],near_variant=e['variant'][k,0],far_variant=e['variant'][k,1],center_reference=e['center_reference'][k],center_variant=e['center_variant'][k],cr_change=e['cr_change'][k],contact_reference=e['contact_reference'][k],contact_variant=e['contact_variant'][k]))
    summary=[];gates=[]
    for lo,hi in ((30,50),(50,75),(75,100)):
        rr=[r for r in stations if lo<=r['range_m']<hi];line=dict(lo=lo,hi=hi,stations=len(rr),starts=len(set((r['run'],r['frame']) for r in rr)))
        for key in ('near_reference','far_reference','near_variant','far_variant','center_reference','center_variant','cr_change','contact_reference','contact_variant'):
            values=[r[key] for r in rr if np.isfinite(r[key])]
            line[key+'_p95']=None if not values else float(np.percentile(values,95))
            if key.startswith('contact'):line[key+'_stations']=len(values)
        line['far_p95_delta_m']=None if not rr else line['far_variant_p95']-line['far_reference_p95'];line['pass']=bool(rr) and line['far_p95_delta_m']<=.01;gates.append(line['pass']);summary.append(line)
    for threshold in (.1,.2,.5):
        aa={(r['run'],r['frame']) for r in stations if r['far_reference']>threshold};bb={(r['run'],r['frame']) for r in stations if r['far_variant']>threshold}
        gates.append(len(bb-aa)<=(1 if threshold==.2 else 0) if threshold!=.1 else True)
        summary.append(dict(metric=f'new_starts_over_{int(threshold*100)}cm',value=len(bb-aa),reference=len(aa),variant=len(bb)))
    for threshold in (30,50,75):
        aa=np.mean([r['reference_horizon']>=threshold for r in cases]);bb=np.mean([r['variant_horizon']>=threshold for r in cases]);gates.append(aa-bb<=.02000001);summary.append(dict(metric=f'coverage_{threshold}m',reference=aa,variant=bb,loss_pp=100*(aa-bb)))
    wrong=sum(r['wrong_side'] for r in cases);crossings=sum(r.get('new_crossing_stations',0) for r in cases)
    gates.extend([wrong==0,crossings==0]);summary.extend([dict(metric='new_wrong_side_starts',value=wrong),dict(metric='new_crossing_stations',value=crossings)])
    lc.csv_write(root/'matched_stations.csv',stations);lc.csv_write(root/'quality.csv',summary);lc.csv_write(root/'cases.csv',cases)
    lc.save(root/'QUALITY.json',dict(pass_primary_gates=all(gates),summary=summary,cases=cases,evaluation_ns=time.time_ns(),inference_ns=barrier['time_ns'],note='CR change is prediction disagreement, not CR accuracy. GT is approximate.'))
    print(root.name,'PASS',all(gates),summary,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();evaluate_folder(OUT/a.folder)

