"""Compare a packaged native build with offline reference arrays on the same inputs.

External files are required; no recording or reference cloud is distributed.
Reference NPZ contains pair, contact, curve, indices and point_state per frame.
"""
import argparse,json,os,sqlite3,sys,tempfile
from pathlib import Path
from types import MappingProxyType
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['C4_BACKEND']='full_native_exact';os.environ['PERSISTENT_C4']='OFF'
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bag',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frames',nargs='+',type=int,required=True)
    args=parser.parse_args()
    engine=Path(__file__).resolve().parents[1]/'engine'
    sys.path[:0]=[str(engine/'runtime'),str(engine/'native'),str(engine/'baseline/adapter'),str(engine/'baseline/postprocess'),str(engine/'baseline/app')]
    from cdr_cloud import decode
    from MVP.realtime_final.memory_engine import MemoryEngine,SharedRing
    from MVP.final_pipeline.frame_data import FrameData
    from fusion import CausalSource
    from exact_math import install
    from rt_common import finite
    records=json.loads(args.records.read_text())['frames'];rows=[]
    db=sqlite3.connect(args.bag.resolve().as_uri()+'?mode=ro',uri=True)
    with tempfile.TemporaryDirectory() as directory:
        session=Path(directory);(session/'input.frames.json').write_text(json.dumps(dict(frames=records)))
        for src in args.frames:
            e=MemoryEngine(4);e.start_run(session);install(e);ring=SharedRing(records)
            history,_=CausalSource('',src,records).history({'frames':12})
            for k in sorted(history):
                assert k<=src
                blob=db.execute('SELECT data FROM messages WHERE id=?',(records[k]['message_id'],)).fetchone()[0]
                _,p=decode(blob);values=np.column_stack([p[n] for n in ['x','y','z','intensity']]);ids=np.flatnonzero(finite(values)).astype(np.uint32)
                pose=np.asarray(records[k]['lidar_pose_in_folder']);world=values[ids,:3].astype(float)@pose[:3,:3].T+pose[:3,3]
                ring.arrive(FrameData(world,np.full(len(ids),-1,np.int16),ids,MappingProxyType(dict(intensity_raw=values[ids,3])),f'RAW_{k}',k,len(world)*16))
            e.prepare(ring,src);r=e.process_frame(src)
            actual=dict(pair=r['prediction']['pair'],contact=r['prediction']['C'],curve=r['c4']['curve'],indices=r['c4']['indices'],point_state=r['c4']['point_state'])
            reference=np.load(args.reference/f'{src:06d}.npz');checks={}
            for name,value in actual.items():
                target=reference[name];same_shape=value.shape==target.shape
                integer=np.issubdtype(value.dtype,np.integer)
                equal=same_shape and (np.array_equal(value,target) if integer else np.allclose(value,target,rtol=0,atol=1e-8))
                checks[name]=dict(pass_=bool(equal),shape=list(value.shape),max_abs_error=float(np.max(np.abs(value.astype(float)-target.astype(float)))) if same_shape and value.size else None,exact=bool(same_shape and np.array_equal(value,target)))
            rows.append(dict(frame=src,checks=checks));e.close()
            print(json.dumps(rows[-1]),flush=True)
    report=dict(pass_=all(c['pass_'] for row in rows for c in row['checks'].values()),float_absolute_tolerance_m=1e-8,frames=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    assert report['pass_'],report

if __name__=='__main__':main()
