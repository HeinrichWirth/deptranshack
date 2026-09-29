from common import *
from evaluation import load_prediction,reference
from scipy.spatial import cKDTree

def main():
    rows=load(OUT/'analysis_rows.json');out=[]
    for r in rows:
        if not r.get('wrong_structure_suspected'):continue
        run=r['run'];i=r['start_frame'];folder=OUT/'heldout'/key(run,i);pred=load_prediction(folder);ev=load(folder/'evaluation.json');xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,_=reference(run,i,'heldout')
        for step in ev['steps']:
            if step['status']!='ACCEPTED' or not step.get('point_p95') or step['point_p95']<=.20:continue
            raw=next(s for s in pred['steps'] if s['step_index']==step['step_index']);ids=np.array(raw['support_indices'],dtype=int);p=xyz[ids];B=np.array(raw['basis'])
            d,j=cKDTree(ref['future']).query(p);delta=(p-ref['future'][j])@B;cur=cKDTree(ref['current']).query(p)[0] if len(ref['current']) else np.full(len(p),np.nan)
            bad=d>.20;record=dict(run=run,start_frame=i,step_index=step['step_index'],points=len(p),points_over_20cm=int(bad.sum()),
                distance_p95=float(np.percentile(d,95)),longitudinal_abs_p95=float(np.percentile(abs(delta[:,0]),95)),transverse_p95=float(np.percentile(np.linalg.norm(delta[:,1:],axis=1),95)),
                current_annotation_p95=percentiles(cur).get('p95'),bad_longitudinal_median=float(np.median(abs(delta[bad,0]))) if bad.any() else None,
                bad_transverse_median=float(np.median(np.linalg.norm(delta[bad,1:],axis=1))) if bad.any() else None,
                evidence='3D nearest-reference vector resolved along actual slice tangent; projection alone hides missing longitudinal GT support.')
            out.append(record)
    save(OUT/'audit/wrong_structure_investigation.json',out);csv_write(OUT/'wrong_structure_investigation.csv',out);print(out,flush=True)
if __name__=='__main__':main()
