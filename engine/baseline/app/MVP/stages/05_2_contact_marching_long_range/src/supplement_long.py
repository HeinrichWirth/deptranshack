"""Post-inference development feature schema and labelled profile diagnostics."""
from long_common import *
from observability import extra_fields
from fusion import CausalSource
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from contact_rail_step2 import ContactRailDetector

def main():
    rows=[];visibility=[];schemas=[];distributions=[];template=ContactRailDetector().template
    uv,vc=np.unique(np.round(template[:,0],3),return_counts=True);uw,wc=np.unique(np.round(template[:,1],3),return_counts=True)
    vert=abs(template[:,0]-uv[np.argmax(vc)])<.004;horiz=abs(template[:,1]-uw[np.argmax(wc)])<.004
    for r in load(OUT/'audit/cohort.json')['development']:
        run=r['run'];i=r['frame'];target=OUT/'observability'/key(run,i)
        if not (target/'near_gt_points.npz').exists():continue
        check_marker(OUT/'phase_b_full_dev/B0'/key(run,i));p,_=CausalSource(run,i).read(i)
        cls,fields,schema=extra_fields(run,i);path=dataset()/run/frames(run)[i]['file']
        with laspy.open(path) as las:h=las.header
        raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
        if h.point_format.id<6:returns=((raw['bit_fields']>>3)&7).copy()
        else:returns=((raw['bit_fields']>>4)&15).copy()
        del raw
        schemas.append(dict(run=run,start_frame=i,point_format=h.point_format.id,dimensions=schema,number_of_returns_values=np.unique(returns).tolist()))
        feature_values={name:x for name,x in fields.items() if name!='bit_fields'}
        feature_values.update(number_of_returns=returns,return_number=fields['bit_fields']&(7 if h.point_format.id<6 else 15))
        ref,_=reference_readonly(run,i);ranges=p['sensor_distance_at_T'];near=cKDTree(ref['curve']).query(p['xyz'],distance_upper_bound=.5)[0]<=.5
        for lo,hi in ((0,20),(20,40),(40,60),(60,80),(80,100),(100,130)):
            mask=near&(ranges>=max(8,lo))&(ranges<hi);pos=returns[mask&(cls==2)];neg=returns[mask&(cls!=2)];auc=None
            if len(pos) and len(neg):auc=float((rankdata(np.r_[pos,neg])[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
            rows.append(dict(run=run,start_frame=i,lo=lo,hi=hi,field='number_of_returns',positive_n=len(pos),negative_n=len(neg),auc=auc,positive_median=float(np.median(pos)) if len(pos) else None,negative_median=float(np.median(neg)) if len(neg) else None))
            for name,values in feature_values.items():
                for label,condition in [('class2',cls==2),('clutter',cls!=2)]:
                    a=values[mask&condition]
                    distributions.append(dict(run=run,start_frame=i,lo=lo,hi=hi,field=name,label=label,n=len(a),**{f'q{q}':float(np.quantile(a,q/100)) if len(a) else None for q in (5,25,50,75,95)}))
        with np.load(target/'near_gt_points.npz') as z:points={k:z[k] for k in z.files}
        sources=load(target/'COMPLETE.json')['source_frames']
        for hist in (1,2,4,8,12,16):
            base=np.isin(points['source_frame'],sources[:hist])&points['class2']&(points['distance']<=.05)
            for lo,hi in ((0,20),(20,40),(40,60),(60,80),(80,100),(100,130)):
                mask=base&(points['range']>=lo)&(points['range']<hi);nodes=np.unique(points['node'][mask]);n=int(mask.sum())
                visibility.append(dict(run=run,start_frame=i,history=hist,lo=lo,hi=hi,class2_returns=n,template_fraction=len(nodes)/len(template),vertical_samples=int(vert[nodes].sum()),horizontal_samples=int(horiz[nodes].sum()),both_faces_visible=bool(vert[nodes].any() and horiz[nodes].any()),scope='class2 AND <=5cm future GT tube; imperfect labels/registration'))
    save(OUT/'audit/development_schema.json',schemas);csv_write(OUT/'audit/number_of_returns.csv',rows);csv_write(OUT/'partial_template/development_class2_visibility.csv',visibility)
    csv_write(OUT/'audit/development_conditional_distributions.csv',distributions)
    features=[r for p in (OUT/'observability').glob('*/features.json') for r in load(p)]+rows
    summary=[]
    for field in sorted(set(r['field'] for r in features)):
        for lo in (0,20,40,60,80,100):
            rr=[r for r in features if r['field']==field and r['lo']==lo and r.get('auc') is not None]
            if rr:summary.append(dict(field=field,lo=lo,starts=len(rr),median_auc=float(np.median([r['auc'] for r in rr])),positive_n=sum(r['positive_n'] for r in rr),negative_n=sum(r['negative_n'] for r in rr)))
    save(OUT/'audit/development_feature_summary.json',summary);csv_write(OUT/'audit/development_feature_summary.csv',summary)
    print('SUPPLEMENT',len(schemas),len(rows),len(visibility),flush=True)

if __name__=='__main__':main()
