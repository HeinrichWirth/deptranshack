"""Long-horizon inverse-geometry controls on CR0, separately from C4-limited reach."""
from rr_common import *
from oracle_study import reference_case
from track_geometry import offsets,rails
from evaluate_rails import measure,rows_for,aggregate
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');a=ap.parse_args();rows=[];starts=[]
    for folder in (OUT/'oracle_cr'/a.group/'B1_BISHOP').iterdir():
        if not folder.is_dir():continue
        rec=load(folder/'prediction.json');ref=reference_case(rec['run'],rec['i'])
        with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
        e=measure(p,ref);truth=np.full_like(p['state'],np.nan);q=int(p['q'])
        for k in np.flatnonzero(e['valid']):truth[k]=offsets(p['C'][k],p['B'][k],e['gt'][k],q)
        for name in ('CR0_ORACLE_ALPHA','CR0_ORACLE_OFFSETS','CR0_ALL_ORACLE'):
            new={k:v.copy() for k,v in p.items()};x=p['state'].copy()
            if name!='CR0_ORACLE_OFFSETS':x[:,0]=truth[:,0]
            if name!='CR0_ORACLE_ALPHA':x[:,1:]=truth[:,1:]
            new['state']=x;new['pair']=rails(p['C'],p['B'],x,q);new['center']=new['pair'].mean(axis=1)
            ev=measure(new,ref);r=dict(rec,method=name,production=False,future_class1_used_in_prediction=True,oracle_scope='Diagnostic only; true reference state, never a deployable method.')
            dest=OUT/'oracle_decomposition'/('extended_'+a.group)/name/folder.name;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'prediction.npz',**new);np.savez_compressed(dest/'evaluation.npz',**ev);save(dest/'prediction.json',r)
            rows.extend(rows_for(r,new,ev));starts.append(dict(run=r['run'],frame=r['i'],method=name,far_p95=stats(ev['error'][:,1]).get('p95'),near_p95=stats(ev['error'][:,0]).get('p95'),max_evaluated_range=float(ev['range'][ev['valid']].max()) if ev['valid'].any() else None))
    path=OUT/'oracle_decomposition'/('extended_'+a.group);write_csv(path/'per_station.csv',rows);write_csv(path/'per_start.csv',starts);write_csv(path/'range_metrics.csv',aggregate(rows));save(path/'range_metrics.json',aggregate(rows));save(path/'COMPLETE.json',dict(time_ns=time.time_ns(),evaluations=len(starts),production=False))
    print('EXTENDED ORACLES',a.group,len(starts),flush=True)

if __name__=='__main__':main()
