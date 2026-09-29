"""Evaluation-only information ceiling and source/feature diagnostics."""
from long_common import *
from fusion import CausalSource
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

HISTORY=(1,2,4,8,12,16)
HORIZONS=(60,75,90,100,110,120)

def extra_fields(run,i):
    path=dataset()/run/frames(run)[i]['file']
    with laspy.open(path) as f:h=f.header
    raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
    cls=np.asarray(raw['raw_classification']&31).copy() if 'raw_classification' in raw.dtype.names else np.asarray(raw['classification']).copy()
    fields={k:np.asarray(raw[k]).copy() for k in ('intensity','intensity_raw','bit_fields','ring') if k in raw.dtype.names}
    schema=list(h.point_format.dimension_names);del raw
    return cls,fields,schema

def one(task):
    run,i,group=task;target=OUT/'observability'/key(run,i)
    if (target/'COMPLETE.json').exists():return run,i,'cached'
    pred=OUT/group/'B0'/key(run,i);check_marker(pred);seed=seed_from_cache(run,i)
    if seed['status']!='AVAILABLE':
        save(target/'COMPLETE.json',dict(unavailable=True,time_ns=time.time_ns()));return run,i,'unavailable'
    ref,info=reference_readonly(run,i)
    if not len(ref['curve']) or not len(ref['future']):
        save(target/'COMPLETE.json',dict(unavailable_reference=True,time_ns=time.time_ns()));return run,i,'no GT'
    source=CausalSource(run,i);ids,_=source.history(dict(frames=16));gtree=cKDTree(ref['future']);curve_tree=cKDTree(ref['curve'])
    template=ContactRailDetector().template*np.array([seed['side'],1]);tt=cKDTree(template)
    pieces=[];budgets=[];visibility=[];feature=[];near_parts=[];schema=None
    for k in ids:
        p,_=source.read(k);rr=p['sensor_distance_at_T'];qids=np.flatnonzero((rr>=8)&(rr<=130))
        dist,j=gtree.query(p['xyz'][qids],distance_upper_bound=np.nextafter(.10,np.inf));take=dist<=.10
        ix=qids[take];d=dist[take];near=p['xyz'][ix];_,b=curve_tree.query(near)
        local=np.einsum('ij,ijk->ik',near-ref['curve'][b],ref['curve_basis'][b]);node=tt.query(local[:,1:])[1]
        cls,fields,schema=extra_fields(run,k)
        near_parts.append(dict(xyz=near,source_frame=np.full(len(ix),k,dtype=np.int32),source_row=p['source_row'][ix],age_frames=p['age_frames'][ix],
          ring=p['ring'][ix],azimuth=p['azimuth'][ix],distance=d,range=rr[ix],node=node,local_vw=local[:,1:],class2=cls[ix]==2))
        if run in SPLIT['development'] and k==i:
            # Matched sampling region/range: CR labels vs adjacent clutter within 0.5m of GT curve.
            cd,_=curve_tree.query(p['xyz'][qids],distance_upper_bound=.5);roi=qids[np.isfinite(cd)]
            for lo,hi in ((0,20),(20,40),(40,60),(60,80),(80,100),(100,130)):
                ix2=roi[(rr[roi]>=lo)&(rr[roi]<hi)]
                for f,x in fields.items():
                    if f=='bit_fields':x=x&7;f='return_number'
                    pos=x[ix2[cls[ix2]==2]];neg=x[ix2[cls[ix2]!=2]]
                    auc=None
                    if len(pos) and len(neg):
                        rank=rankdata(np.r_[pos,neg]);auc=float((rank[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
                    feature.append(dict(run=run,start_frame=i,lo=lo,hi=hi,field=f,positive_n=len(pos),negative_n=len(neg),auc=auc,
                      positive_median=float(np.median(pos)) if len(pos) else None,negative_median=float(np.median(neg)) if len(neg) else None))
    c={f:np.concatenate([p[f] for p in near_parts]) for f in near_parts[0]}
    gtrange=np.linalg.norm(ref['future'],axis=1)
    for n in HISTORY:
        allowed=set(ids[:n]);base=np.isin(c['source_frame'],list(allowed));m=base&(c['distance']<=.05)
        unique=np.unique(c['xyz'][m],axis=0);rangeu=np.linalg.norm(unique,axis=1);counts=np.histogram(rangeu,np.arange(8,133,4))[0]
        max3=float(8+4*(np.flatnonzero(counts>=3)[-1]+1)) if np.any(counts>=3) else 0.
        row=dict(run=run,start_frame=i,history=n,actual_frames=len(allowed),max_GT_available_range=info.get('max_GT_available_range'),
          max_observable_range_5cm=float(c['range'][m].max()) if m.any() else 0.,max_observable_range_10cm=float(c['range'][base].max()) if base.any() else 0.,
          max_observable_3unique_block_end=max3,near_gt_returns=int(m.sum()),unique_xyz=len(unique),
          labelled_class2_near_returns=int(np.sum(m&c['class2'])),has_returns_ge100=bool(np.any(m&(c['range']>=100))),has_returns_ge120=bool(np.any(m&(c['range']>=120))))
        pieces.append(row)
        for h in HORIZONS:
            z=m&(c['range']>=h-2)&(c['range']<h+2)
            budgets.append(dict(run=run,start_frame=i,history=n,center_m=h,lo=h-2,hi=h+2,returns=int(z.sum()),unique_xyz=len(np.unique(c['xyz'][z],axis=0)),
              source_count=len(np.unique(c['source_frame'][z])),rings=len(np.unique(c['ring'][z])),azimuth_samples=len(np.unique(c['azimuth'][z])),
              template_regions=len(np.unique(c['node'][z]//4)),class2_returns=int(np.sum(z&c['class2'])),gt_points=int(np.sum((gtrange>=h-2)&(gtrange<h+2)))))
        for lo,hi in ((0,20),(20,40),(40,60),(60,80),(80,100),(100,130)):
            z=m&(c['range']>=lo)&(c['range']<hi)
            # Occupancy along actual canonical template, not bounding-box width of clutter.
            v=template[:,0];w=template[:,1];uv,vc=np.unique(np.round(v,3),return_counts=True);uw,wc=np.unique(np.round(w,3),return_counts=True)
            vertical=abs(v-uv[np.argmax(vc)])<.004;horizontal=abs(w-uw[np.argmax(wc)])<.004
            nodes=np.unique(c['node'][z]);visibility.append(dict(run=run,start_frame=i,history=n,lo=lo,hi=hi,returns=int(z.sum()),
              labelled_returns=int(np.sum(z&c['class2'])),template_samples=len(nodes),template_fraction=len(nodes)/len(template),
              vertical_samples=int(vertical[nodes].sum()),horizontal_samples=int(horizontal[nodes].sum()),
              distinct_sources=len(np.unique(c['source_frame'][z])),distinct_rings=len(np.unique(c['ring'][z]))))
    target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/'near_gt_points.npz',**c)
    save(target/'ranges.json',pieces);save(target/'budgets.json',budgets);save(target/'visibility.json',visibility);save(target/'features.json',feature)
    save(target/'COMPLETE.json',dict(time_ns=time.time_ns(),source_frames=ids,schema=schema,evaluation_only=True,near_gt_is_proxy_not_manual_truth=True))
    return run,i,[(r['history'],round(r['max_observable_range_5cm'],1)) for r in pieces]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    cohort=load(OUT/'audit/cohort.json')['heldout' if a.group=='phase_e_benchmark' else 'development']
    # Phase A includes only fixed subset; full development after Phase B.
    if a.group=='phase_a_screen':cohort=load(OUT/'audit/screen.json')
    tasks=[(r['run'],r['frame'],a.group) for r in cohort];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,r) for r in tasks]
        for j,f in enumerate(as_completed(ff),1):
            v=f.result()
            if j%20==0 or j==len(ff):print('OBSERVABILITY',j,len(ff),v,'elapsed',round(time.perf_counter()-t,1),flush=True)
    for field,name in [('ranges','observability_range.csv'),('budgets','point_budget_100m.csv'),('visibility','partial_template/profile_visibility.csv'),('features','audit/development_features.csv')]:
        rows=[r for p in (OUT/'observability').glob('*/'+field+'.json') for r in load(p)];csv_write(OUT/name,rows)

if __name__=='__main__':main()
