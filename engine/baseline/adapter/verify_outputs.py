"""Independent post-run checks of causality, frozen source and LAS provenance."""
from pathlib import Path
import sys,json,hashlib,sqlite3
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import bootstrap
import numpy as np
import laspy
from cdr_cloud import decode
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    out=ROOT/'results_ros2_replay_v1/full';summary=load(out/'SUMMARY.json');audit=load(out/'causality_audit.json')
    arrivals=[json.loads(s) for s in (out/'arrivals.jsonl').read_text().splitlines()];poses=[json.loads(s) for s in (out/'poses.jsonl').read_text().splitlines()]
    assert len(arrivals)==len(poses)==1510 and [r['frame'] for r in arrivals]==list(range(1510))
    assert all(p['available_after_frame']==i and not p['uses_future_pose'] and p['reference_index']<=i for i,p in enumerate(poses))
    for job in audit['solves']:
        i=job['frame'];assert i<1509 and job['max_pose_frame']<=i+1 and job['dispatched_after_frame']>=i+1
        assert max(job['cloud_frames'])==i and min(job['cloud_frames'])>=i-11
        assert all(poses[k]['pose'] is not None for k in job['cloud_frames'])
    for r in load(out/'solves.json'):
        folder=out/'solves'/f'{r["frame"]:06d}';done=load(folder/'PREDICTION_COMPLETE.json')
        for name,digest in done['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
        with np.load(folder/'point_provenance.npz') as z:
            rows=z['rows'];assert not len(rows) or rows['source_frame'].max()<=r['frame']
    file=out/'stitched_cloud.las';counts=np.array([r['eligible_points'] if poses[i]['pose'] is not None else 0 for i,r in enumerate(arrivals)],dtype=np.int64);starts=np.r_[0,np.cumsum(counts)]
    db=ROOT/'datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3';conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    candidates=sorted({0,1,218,232,370,755,1000,1509}|{c['frame'] for c in load(out/'EXPORTS.json')['examples']})
    checked=[]
    with laspy.open(file) as f:
        assert f.header.point_count==int(starts[-1])
        for i in candidates:
            if not counts[i]:continue
            meta,p=decode(conn.execute('SELECT data FROM messages WHERE id=?',(arrivals[i]['message_id'],)).fetchone()[0]);xyz=np.column_stack([p[k] for k in ('x','y','z')]);ids=np.flatnonzero(np.isfinite(xyz).all(axis=1)&np.any(xyz!=0,axis=1))
            P=np.asarray(poses[i]['pose']);expected=xyz[ids].astype(float)@P[:3,:3].T+P[:3,3]
            f.seek(int(starts[i]));rows=f.read_points(int(counts[i]));np.testing.assert_allclose(np.column_stack([rows.x,rows.y,rows.z]),expected,rtol=0,atol=.00005001)
            np.testing.assert_array_equal(rows.point_index,ids);np.testing.assert_array_equal(rows.intensity_raw,p['intensity'][ids]);assert np.all(rows.frame_index==i) and np.all(rows.message_id==arrivals[i]['message_id']);checked.append(i)
    for sample in load(out/'EXPORTS.json')['examples']:
        with laspy.open(out/sample['las']) as f:rows=f.read()
        generated=np.asarray(rows.geometry_generated)==1
        assert np.all(np.asarray(rows.point_index)[generated]==-1)
        assert set(np.asarray(rows.geometry_component)[generated])<= {1,2,3}
        assert int((~generated).sum())==arrivals[sample['frame']]['eligible_points']
    manifest=load(ROOT/'MVP/realtime_final/FINAL_GEOMETRY_RELEASE_MANIFEST.json');checked_frozen=0
    for group in ('runtime_files','build_files','release_artifacts'):
        for name,item in manifest[group].items():
            p=ROOT/name
            if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];checked_frozen+=1
    for name,digest in load(ROOT/'results_ros2_replay_v1/RUN_SOURCE_LOCK.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    result=dict(pass_=True,frames=1510,causal_solves_checked=len(audit['solves']),map_points=int(starts[-1]),map_frames_checked=checked,frozen_files_checked=checked_frozen,adapter_unchanged=True,all_solve_exports_hash_checked=True)
    (out/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print('VERIFICATION_PASS',result)
if __name__=='__main__':main()
