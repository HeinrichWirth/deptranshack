"""Select diagnostic runs from measured frozen C4_SMOOTH, never their names."""
from . import ROOT
import long_common as lc
import numpy as np
import json

OUT=ROOT/'results_final_pipeline'


def main():
    protocol=lc.load(ROOT/'results_latent_contact_reference/protocol.json')
    rows=[]
    for run in protocol['split']['development']+protocol['split']['validation']:
        records=lc.frames(run);directory=lc.dataset()/run
        group='phase_d' if run in protocol['split']['development'] else 'phase_e'
        folders=sorted((ROOT/'results_latent_contact_reference'/group/'C4_SMOOTH').glob(run+'__*'))
        curvature=[];horizon=[];angles=[];valid=0;changes=0;heading_change=0.
        for folder in folders:
            path=folder/'curve.npz'
            if not path.exists():continue
            with np.load(path) as z:
                if 'C' not in z.files:continue
                C=z['C'];s=z['s']
            if len(C)<3:continue
            i=int(folder.name.rsplit('__',1)[1]);R=np.asarray(records[i]['lidar_pose_in_folder'])[:3,:3]
            C=C@R.T
            d=np.gradient(C,s,axis=0);heading=np.unwrap(np.arctan2(d[:,1],d[:,0]));k=np.gradient(heading,s)
            curvature.extend(abs(k));horizon.append(float(s[-1]));angles.append((i,float(heading[0])))
            active=k[abs(k)>.001];changes+=int(np.sum(np.diff(np.sign(active))!=0));heading_change+=float(np.sum(abs(np.diff(heading))))
            valid+=1
        a=np.asarray(curvature)
        if not len(a):continue
        byte_estimate=sum((directory/r['file']).stat().st_size for r in records)
        rows.append(dict(run=run,frames=len(records),frozen_starts=len(folders),valid_frames=valid,
            availability=valid/max(1,len(folders)),coverage_basis='all frozen sampled starts; not yet full-run inference',
            median_horizon=float(np.median(horizon)),median_abs_curvature=float(np.median(a)),
            p90_abs_curvature=float(np.percentile(a,90)),p95_abs_curvature=float(np.percentile(a,95)),
            fraction_curved_stations=float(np.mean(a>.001)),curvature_sign_changes=changes,
            sum_forecast_heading_change_rad=heading_change,
            total_start_heading_change_rad=float(np.sum(abs(np.diff(np.unwrap([h for _,h in sorted(angles)]))))),
            raw_las_bytes=byte_estimate,diagnostic_disk_estimate_bytes=int(byte_estimate*1.6)))
    eligible=[r for r in rows if r['frames']>=200 and r['availability']>=.5 and r['median_horizon']>=15]
    straight=min(eligible,key=lambda r:(r['median_abs_curvature'],r['total_start_heading_change_rad']))
    curved=max([r for r in eligible if r['run']!=straight['run']],key=lambda r:(r['p90_abs_curvature'],r['fraction_curved_stations']))
    straight=dict(straight,role='straight_run',reason='lowest median horizontal curvature among sufficiently long and covered runs')
    curved=dict(curved,role='curved_run',reason='highest p90 horizontal curvature among remaining sufficiently long and covered runs')
    lc.save(OUT/'selected_las_runs.json',dict(rules=dict(minimum_frames=200,minimum_availability=.5,minimum_median_horizon=15,curved_threshold_per_m=.001),
        selection_source='frozen C4_SMOOTH world horizontal curvature, independent of seed and GT quality',runs=[straight,curved],candidates=rows))
    lc.csv_write(OUT/'run_curvature_selection.csv',rows)
    print(json.dumps([straight,curved],indent=2),flush=True)


if __name__=='__main__':main()
