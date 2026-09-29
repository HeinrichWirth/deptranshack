"""Offline current-pose evaluation after each complete inference run barrier."""
from . import ROOT,OUT
from .run_full import RUNS
from MVP.final_pipeline.spatial_scheduler import transform_prediction
import long_common as lc
import numpy as np
from gt_reference import future_pair_reference
from evaluate_rails import measure
import time,argparse


def read(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def evaluate(run):
    base=OUT/'runs/scheduler_spatial'/run;barrier=lc.load(base/'INFERENCE_COMPLETE.json');records=lc.frames(run)
    rows=[];sources={};errors=[]
    for r in barrier['rows']:
        if r['fresh'] or not r['valid_geometry_state']:continue
        i=r['frame'];j=r['source_geometry_frame'];Q=np.asarray(records[i]['lidar_pose_in_folder'])
        if j not in sources:
            P=np.asarray(records[j]['lidar_pose_in_folder']);sources[j]=transform_prediction(read(base/f'{j:06d}/prediction.npz'),P[:3,:3],P[:3,3],True)
        p=transform_prediction(sources[j],Q[:3,:3],Q[:3,3],False)
        folder=OUT/'evaluation'/run/f'{i:06d}';folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'prediction.npz',**p)
        lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=barrier['time_ns'],files={'prediction.npz':lc.sha(folder/'prediction.npz')},
            source_frame=j,geometry_reused=True))
        cache=ROOT/'results_running_rails_from_contact/audit/future_class1'/run/'anchors.npz'
        entry=dict(run=run,frame=i,source_frame=j,age_frames=r['geometry_age_frames'],age_seconds=r['geometry_age_seconds'],
            distance=r['distance_since_geometry_update'],stale=r['stale'],evaluated_stations=0,far_p95=None,far_max=None,near_p95=None,near_max=None)
        if cache.exists():
            ref=future_pair_reference(run,i,folder)
            if len(ref['s']):
                e=measure(p,ref,p['B']);valid=e['valid']&(e['range']>=8)&(e['range']<=100)
                # Sections behind the current sensor along inherited forward basis are excluded.
                forward=p['B'][0,:,0];valid&=(p['C']@forward)>=0
                vals=e['error'][valid,1];near=e['error'][valid,0]
                entry.update(evaluated_stations=len(vals),far_p95=float(np.percentile(vals,95)) if len(vals) else None,
                    far_max=float(np.max(vals)) if len(vals) else None,
                    near_p95=float(np.percentile(near,95)) if len(near) else None,near_max=float(np.max(near)) if len(near) else None)
                errors.extend(dict(run=run,frame=i,distance=entry['distance'],stale=entry['stale'],far_error=float(x),near_error=float(y),range_m=float(z))
                    for x,y,z in zip(vals,near,e['range'][valid]))
        rows.append(entry)
    lc.save(OUT/'evaluation'/run/'SUMMARY.json',dict(rows=rows,errors=errors,evaluation_time_ns=time.time_ns(),inference_barrier_ns=barrier['time_ns']))
    print('REUSE_EVAL',run,len(rows),sum(r['evaluated_stations'] for r in rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',choices=RUNS);a=ap.parse_args()
    for run in [a.run] if a.run else RUNS:evaluate(run)


if __name__=='__main__':main()
