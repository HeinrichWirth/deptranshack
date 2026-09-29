"""Explain frozen results without altering them or fitting new parameters."""
from rr_common import *
from seed_input import read_input
from track_geometry import unit

def main():
    name=load(OUT/'research_freeze.json')['best_development'];rows=[];centers=[];duplicates=[]
    for group in ('phase_b','phase_e'):
        for r in load(OUT/group/'per_start.json'):
            if r['method']!=name or r['status']!='AVAILABLE':continue
            folder=OUT/group/name/key(r['run'],r['frame']);record=load(folder/'prediction.json');meta,a=read_input(OUT/record['input_folder']);s=a['s'];linear=np.column_stack([np.interp(s,a['anchor_knots'],a['anchor_xyz'][:,j]) for j in range(3)]);difference=np.linalg.norm(a['C']-linear,axis=1);k=int(np.argmax(difference))
            rows.append(dict(run=r['run'],frame=r['frame'],cohort=group,anchor_count=len(a['anchor_xyz']),max_C2_vs_linear_m=float(difference[k]),station_at_max=s[k],smallest_anchor_interval=float(np.diff(a['anchor_knots']).min()),diagnostic_only=True))
            ds=np.diff(s)
            if np.any(ds<1e-9):duplicates.append(dict(run=r['run'],frame=r['frame'],cohort=group,min_ds=float(ds.min()),end_station=float(s[-1]),explanation='End station differs from integer grid by floating point epsilon. Frozen prediction retained; no geometric backward segment.'))
            if group!='phase_e':continue
            with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
            with np.load(folder/'evaluation.npz') as z:e={k:z[k] for k in z.files}
            for k in np.flatnonzero(e['valid']):
                b=unit(int(p['q'])*(e['gt'][k,0]-e['gt'][k,1]));n=np.cross(a['B'][k,:,0],b);T=np.column_stack((b,n));cov=T.T@p['center_cov'][k]@T;delta=(p['center'][k]-e['gt'][k].mean(axis=0))@T;mahal=float(delta@np.linalg.pinv(cov)@delta)
                centers.append(dict(run=r['run'],frame=r['frame'],station=s[k],range_m=e['range'][k],center_error=float(np.linalg.norm(delta)),sigma_lateral=float(np.sqrt(max(cov[0,0],0))),sigma_vertical=float(np.sqrt(max(cov[1,1],0))),inside95=mahal<=5.991464547,inside99=mahal<=9.210340372))
    write_csv(OUT/'audit/interpolation_diagnostic.csv',rows);write_csv(OUT/'audit/near_duplicate_stations.csv',duplicates);write_csv(OUT/'corridors/center_per_station.csv',centers)
    summary=[]
    for lo,hi in ((30,50),(50,75),(75,100),(100,125)):
        rr=[r for r in centers if lo<=r['range_m']<hi]
        if rr:summary.append(dict(lo=lo,hi=hi,n=len(rr),coverage95=float(np.mean([r['inside95'] for r in rr])),coverage99=float(np.mean([r['inside99'] for r in rr])),median_fullwidth95_lateral=float(4.895*np.median([r['sigma_lateral'] for r in rr])),median_fullwidth95_vertical=float(4.895*np.median([r['sigma_vertical'] for r in rr]))))
    write_csv(OUT/'corridors/center_range_metrics.csv',summary);save(OUT/'audit/posthoc_diagnostics.json',dict(time_ns=time.time_ns(),interpolation_maxima=stats([r['max_C2_vs_linear_m'] for r in rows]),cases_C2_more_10cm=sum(r['max_C2_vs_linear_m']>.1 for r in rows),near_duplicate_stations=duplicates,center_corridors=summary,no_prediction_modified=True,no_new_model_evaluated=True))
    print('POSTHOC DIAGNOSTICS',len(rows),len(duplicates),'duplicate endpoint cases',flush=True)

if __name__=='__main__':main()
