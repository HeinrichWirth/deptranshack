"""Post-inference original-vs-copy audit. Original files are read-only."""
import json,sys,tarfile,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)

def verify(folder):
    if (folder/'results.tar.gz').exists() and not (folder/'payload/COMPLETE.json').exists():
        with tarfile.open(folder/'results.tar.gz','r:gz') as archive:archive.extractall(folder,filter='data')
    out=folder/'payload' if (folder/'payload').exists() else folder
    old=ROOT/'results_ros2_replay_v1/full';summary=load(out/'SUMMARY.json');poses=lines(out/'poses.jsonl');oldposes=lines(old/'poses.jsonl');arrivals=lines(out/'arrivals.jsonl');audit=load(out/'causality_audit.json')
    assert len(poses)==len(arrivals)==summary['frames']
    diffs=[i for i,p in enumerate(poses) if p['pose']!=oldposes[i]['pose']];assert not diffs,('POSE_DIFFERENCES',diffs)
    for i,p in enumerate(poses):assert p['available_after_frame']==i and not p['uses_future_pose'] and p['reference_index']<=i
    for row in audit['solves']:
        i=row['frame'];assert row['max_pose_frame']<=i+1 and row['dispatched_after_frame']>=i+1 and max(row['cloud_frames'])<=i and min(row['cloud_frames'])>=i-11
    checked=0;differences=[]
    for row in load(out/'solves.json'):
        i=row['frame'];directory=out/'solves'/f'{i:06d}';done=load(directory/'PREDICTION_COMPLETE.json')
        assert not done['uses_annotation_labels'] and not done['uses_future_clouds']
        for name,digest in done['files'].items():assert sha(directory/name)==digest
        with np.load(directory/'point_provenance.npz') as z:
            r=z['rows'];assert not len(r) or (r['source_frame'].min()>=i-11 and r['source_frame'].max()<=i)
        reference=old/'solves'/f'{i:06d}'
        if not reference.exists():continue
        checked+=1;a=load(directory/'summary.json');b=load(reference/'summary.json')
        for key in ('status','original_reason','final_geometry_available'):
            if a[key]!=b[key]:differences.append([i,key])
        for name in ('prediction.npz','curve.npz','point_provenance.npz'):
            left,right=directory/name,reference/name
            if left.exists()!=right.exists():differences.append([i,name,'presence']);continue
            if not left.exists():continue
            with np.load(left) as x,np.load(right) as y:
                for key in set(x.files)|set(y.files):
                    if key not in x or key not in y or not equal(x[key],y[key]):
                        # Structured provenance contains intentional NaN confidence values.
                        if key=='rows' and x[key].dtype.names==y[key].dtype.names and x[key].shape==y[key].shape and all(equal(x[key][n],y[key][n]) for n in x[key].dtype.names):continue
                        differences.append([i,name,key])
    result=dict(pass_=not differences,frames=len(poses),poses_exact=True,common_solves=checked,differences=differences,causal_jobs=len(audit['solves']),las_files=len(list(out.rglob('*.las'))),all_export_hashes_checked=True)
    (folder/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(folder.name,json.dumps(result));assert result['pass_']
    return result

if __name__=='__main__':
    for path in sys.argv[1:]:verify(Path(path))
