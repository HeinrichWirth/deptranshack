"""Paths and read-only STEP5 imports. No monkeypatching of frozen modules."""
import os,sys
sys.dont_write_bytecode=True
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parents[2]
PREVIOUS=ROOT/'MVP/stages/05_contact_rail_marching'
BASE_OUT=ROOT/'results_contact_marching'
OUT=ROOT/'results_contact_marching_temporal_fusion'
sys.path.insert(0,str(PREVIOUS/'src'))
from common import np,laspy,load,save,sha,clean,csv_write,csv_read,dataset,frames,key,percentiles,SPLIT
import time,json,copy,hashlib
CFG=load(BASE_OUT/'config.json')
ARRAY_KEYS=('indices','point_step','local_uvw','template_residual','confidence','point_anchor','s_from_seed','curve')

def initialize():
    for name in ('audit','calibration','heldout','development','diagnostics','failures','gallery','animations','las','runtime'):(OUT/name).mkdir(parents=True,exist_ok=True)

def load_prediction(folder):
    p=load(folder/'prediction.json')
    with np.load(folder/'points.npz') as z:p.update({k:z[k] for k in z.files})
    return p

def seed_from_cache(run,i):
    s=load(BASE_OUT/'cache'/key(run,i)/'seed.json')
    for name in ('basis','anchor','indices'):
        if s.get(name) is not None:s[name]=np.asarray(s[name],dtype=np.int64 if name=='indices' else float)
    return s

def save_prediction(folder,p,run,i,variant,cloud):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'points.npz',**{k:p[k] for k in ARRAY_KEYS if k in p})
    record={k:v for k,v in p.items() if k not in ARRAY_KEYS};record.update(run=run,start_frame=i,variant=variant,fusion=cloud['meta'])
    save(folder/'prediction.json',record)
    ids=p['indices'];np.savez_compressed(folder/'provenance.npz',**{k:cloud[k][ids] for k in ('source_frame','source_row','source_point_index','age_frames','age_seconds','age_distance','sensor_distance_at_source','sensor_distance_at_T','ring','azimuth')},xyz=cloud['xyz'][ids])
    save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),prediction_sha256=sha(folder/'prediction.json'),points_sha256=sha(folder/'points.npz'),provenance_sha256=sha(folder/'provenance.npz'),uses_future_clouds=False,uses_classification=False))

def dependency_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in PREVIOUS.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
