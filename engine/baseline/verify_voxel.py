import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,json,sqlite3,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/work/adapter')
from cdr_cloud import decode
from copy_voxel import voxel_sample as fast
from causal_registration import original_functions
old=original_functions(Path('/work/registration'))[0]['voxel_sample']
conn=sqlite3.connect('file:/tmp/input.db3?mode=ro',uri=True);ids=[r[0] for r in conn.execute('SELECT id FROM messages ORDER BY id')]
chosen=sorted(set(ids[::31]+ids[210:235]+ids[360:375]+ids[-2:]));rows=[]
for mid in chosen:
    _,p=decode(conn.execute('SELECT data FROM messages WHERE id=?',(mid,)).fetchone()[0]);xyz=np.column_stack([p[k] for k in ('x','y','z')]);r=np.linalg.norm(xyz,axis=1);xyz=xyz[np.isfinite(xyz).all(axis=1)&(r>1.2)&(r<140)]
    t=time.perf_counter();a=old(xyz,.22);old_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter();b=fast(xyz,.22);new_ms=(time.perf_counter()-t)*1000
    np.testing.assert_array_equal(a,b);rows.append(dict(message=mid,old_ms=old_ms,new_ms=new_ms,source_points=len(xyz),voxel_points=len(a)))
rng=np.random.default_rng(582)
for dtype in (np.float32,np.float64):
    a=rng.uniform(-140,140,(100000,3)).astype(dtype);a=np.vstack((a,a[:2000]))
    np.testing.assert_array_equal(old(a),fast(a))
    edges=np.arange(-650,651,dtype=dtype)*dtype(.22)
    a=np.column_stack((np.tile(edges,3),np.repeat([-1,0,1],len(edges)).astype(dtype),np.zeros(3*len(edges),dtype)))
    np.testing.assert_array_equal(old(a),fast(a))
np.testing.assert_array_equal(old(np.empty((0,3),np.float32)),fast(np.empty((0,3),np.float32)))
result=dict(pass_=True,real_frames=len(rows),exact_point_values_and_order=True,original_median_ms=float(np.median([r['old_ms'] for r in rows])),optimized_median_ms=float(np.median([r['new_ms'] for r in rows])),rows=rows)
Path('/work/results/VOXEL_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
