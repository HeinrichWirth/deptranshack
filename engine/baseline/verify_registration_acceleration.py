"""Acceptance/fallback invariants of parallel recovery and exact NN pruning."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT/'COPY_MAIN/adapter')]
import numpy as np
from scipy.spatial import cKDTree
from causal_registration import original_functions
from registration_acceleration import install_parallel_recovery

# Query bounds never remove a correspondence accepted by the original strict gate.
rng=np.random.default_rng(618)
target=rng.normal(size=(3000,3));query=rng.normal(size=(6000,3))
target=np.r_[target,target[:10]];tree=cKDTree(target)
d,ids=tree.query(query)
for limit in (.35,.5,1.8):
    bd,bi=tree.query(query,distance_upper_bound=limit);keep=d<limit
    np.testing.assert_array_equal(keep,bd<limit)
    np.testing.assert_array_equal(d[keep],bd[keep]);np.testing.assert_array_equal(ids[keep],bi[keep])

# An invalid newest reference must not hide an older valid one; original seed
# order and tie-breaking must remain intact, even when completion order differs.
for failure in ('overlap','exception','spread','none'):
    scope,_=original_functions(ROOT/'COPY_MAIN/registration')
    source=np.zeros((110,3));older=np.zeros((110,3));newer=np.ones((110,3))
    references=[(older,older,np.eye(4),5),(newer,newer,np.eye(4),10)]
    def trial(source,target,normal,initial,prior,iterations):
        assert iterations==60
        bad=target[0,0]==1
        if bad and failure=='exception':raise ValueError('no overlap')
        pose=prior.copy()
        if bad and failure=='spread':pose=initial.copy()
        return pose,dict(overlap=.1 if bad and failure=='overlap' else .9,rmse_plane_m=.01)
    scope['recovery_icp']=trial
    expected=scope['recover_endpoint'](source,np.eye(4),references)
    install_parallel_recovery(scope)
    actual=scope['recover_endpoint'](source,np.eye(4),references)
    for a,b in zip(actual,expected):
        if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        else:assert a==b
print('PASS bounded NN accepted correspondences and parallel recovery fallback/acceptance/order')
