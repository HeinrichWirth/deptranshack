"""Real-cloud probe must not advance or mutate the retained registration state."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['COPY_VOXEL']='packed'
import sys,sqlite3,pickle,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT/'COPY_MAIN/adapter')]
import numpy as np
from causal_registration import CausalRegistration
from direction_probe import probe, fork, evaluate_fork
from cdr_cloud import decode
def digest(reg):
    return hashlib.sha256(pickle.dumps({k:v for k,v in reg.__dict__.items() if k not in ('fn','source_hashes')})).hexdigest()
db=ROOT/'datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3'
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
    source=con.execute('SELECT data FROM messages ORDER BY timestamp,id LIMIT 5').fetchall()
clouds=[]
for blob, in source:
    meta,rows=decode(blob);clouds.append((np.column_stack([rows[k] for k in ('x','y','z')]),meta['header_time_ns']))
a=CausalRegistration(ROOT/'COPY_MAIN/registration');b=CausalRegistration(ROOT/'COPY_MAIN/registration')
for i,k in enumerate((0,2)):
    xa,stamp=clouds[k]
    pa=a.step(xa,i,stamp);pb=b.step(xa,i,stamp)
    np.testing.assert_array_equal(pa['pose'],pb['pose'])
before=digest(a)
pose,details,ms=probe(a,*clouds[3]);assert digest(a)==before
assert pose['pose'] is not None
# Compare against the original path that also permits reference refresh.
import copy
saved=a.__dict__.copy()
a.__dict__.update({k:copy.deepcopy(v) for k,v in saved.items() if k not in ('fn','source_hashes')})
legacy=a.step(clouds[3][0],2,clouds[3][1],update_reference=True)
a.__dict__.clear();a.__dict__.update(saved)
assert legacy==pose
pa=a.step(clouds[4][0],2,clouds[4][1]);pb=b.step(clouds[4][0],2,clouds[4][1])
np.testing.assert_array_equal(pa['pose'],pb['pose'])
before=digest(a)
try:probe(a,np.zeros((100,3)),clouds[4][1]+500_000_000)
except ValueError:pass
else:raise AssertionError('gap accepted')
assert digest(a)==before
from concurrent.futures import ThreadPoolExecutor
c=CausalRegistration(ROOT/'COPY_MAIN/registration')
for i,k in enumerate((0,2)):c.step(clouds[k][0],i,clouds[k][1])
isolated=fork(c)
with ThreadPoolExecutor(max_workers=1) as pool:
    future=pool.submit(evaluate_fork,isolated,*clouds[3])
    next_pose=c.step(clouds[4][0],2,clouds[4][1])
    async_pose,_,_,_=future.result()
assert async_pose==pose
np.testing.assert_array_equal(next_pose['pose'],pb['pose'])
assert c.index==2 and isolated.index==2
print('PASS independent asynchronous direction while retained registration advances')
print(f'PASS real original T+1 pose, exact retained-state restoration, exact next retained pose, exception rollback; probe {ms:.2f} ms')
