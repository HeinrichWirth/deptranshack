"""Isolated STEP5.2 paths and immutable dependency imports."""
import os, sys
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parents[2]
OLD_STAGE = ROOT / 'MVP/stages/05_1_contact_marching_temporal_fusion'
OLD = ROOT / 'results_contact_marching_temporal_fusion'
BASE = ROOT / 'results_contact_marching'
OUT = ROOT / 'results_contact_marching_long_range'
sys.path.insert(0, str(OLD_STAGE / 'src'))
from fusion_common import np, laspy, load, save, sha, clean, csv_write, csv_read, dataset, frames, key, percentiles, SPLIT, CFG, seed_from_cache
import time, json, copy, hashlib
from functools import lru_cache
ARRAYS = ('indices','point_step','local_uvw','template_residual','confidence','point_anchor','s_from_seed','curve','point_state','continuity_score','template_score','track_prior_score')
INFERENCE = ('long_common.py','roi_source.py','long_tracker.py','configurations.py','infer.py')

def initialize():
    names = ('phase_a_screen','phase_b_full_dev','phase_c_cross_run','phase_d_combinations','phase_e_benchmark','observability','beam','tentative','gaps','dynamic_windows','partial_template','fusion','failures','100m_cases','las','gallery','animations','runtime','audit')
    for name in names: (OUT/name).mkdir(parents=True, exist_ok=True)

def code_hashes(): return {p: sha(STAGE/'src'/p) for p in INFERENCE}

def read_prediction(folder):
    p = load(folder/'prediction.json')
    with np.load(folder/'points.npz') as z: p.update({k:z[k] for k in z.files})
    return p

def check_marker(folder):
    m=load(folder/'PREDICTION_COMPLETE.json')
    for f,h in m['files'].items(): assert sha(folder/f)==h, (folder,f)
    return m

@lru_cache(maxsize=12)
def reference_readonly(run,i):
    p=BASE/'future_gt'/(key(run,i)+'_reference_v2.npz')
    with np.load(p) as z: ref={k:z[k] for k in z.files}
    return ref,load(p.with_suffix('.json'))

def save_prediction(folder,p,cloud,run,i,name):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'points.npz',**{k:p[k] for k in ARRAYS if k in p})
    rec={k:v for k,v in p.items() if k not in ARRAYS}
    rec.update(run=run,start_frame=i,variant=name,input=cloud['meta'])
    save(folder/'prediction.json',rec)
    ids=p['indices']
    np.savez_compressed(folder/'provenance.npz',**{k:v[ids] for k,v in cloud.items() if isinstance(v,np.ndarray)})
    # All queried historical records are audit evidence, not predicted points.
    roi=cloud['age_frames']>0
    np.savez_compressed(folder/'evidence.npz',**{k:v[roi] for k,v in cloud.items() if isinstance(v,np.ndarray) and k in ('xyz','source_frame','source_row','source_point_index','age_frames','ring')})
    save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={f:sha(folder/f) for f in ('prediction.json','points.npz','provenance.npz','evidence.npz')},inference_sha256=code_hashes(),uses_future_clouds=False,uses_classification=False))
