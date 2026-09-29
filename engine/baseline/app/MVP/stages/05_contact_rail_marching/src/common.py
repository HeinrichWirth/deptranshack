"""Research I/O only. This module is deliberately not imported by the tracker."""
import os, sys
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parents[2]
OUT = ROOT / 'results_contact_marching'
sys.path[:0] = [str(ROOT / '.runtime'), str(ROOT / 'MVP/pipelines')]
from frozen_classification_common import np, laspy, load, save, sha, clean, csv_write, locks
import time, json, csv, hashlib
DEFAULT_DATA = ROOT / 'output/LAS_FRAMES_CLASSIFIED_20260926/ANNOTATED'
SPLIT = load(ROOT / 'MVP/stages/03_contact_generalization/results_contact_generalization/protocol.json')['outer_split']

def initialize():
    for name in ('audit','calibration','future_gt','ablation','heldout','steps','failures','gallery','animations','las','runtime','cache'):
        (OUT/name).mkdir(parents=True, exist_ok=True)

def dataset():
    p = STAGE/'dataset.json'
    return Path(load(p)['data_root']) if p.exists() else DEFAULT_DATA

def frames(run):
    return load(next((dataset()/run).glob('*.frames.json')))['frames']

def key(run, i): return f'{run}__{i:06d}'

def read_geometry(run, row):
    """Reads current XYZ only: never accesses classification or any other cloud."""
    path = dataset()/run/row['file']
    with laspy.open(path) as f: h = f.header
    raw = np.memmap(path, dtype=h.point_format.dtype(), mode='r', offset=h.offset_to_point_data, shape=(h.point_count,))
    xyz = np.column_stack([np.asarray(raw[a], dtype=np.float64)*h.scales[j]+h.offsets[j] for j,a in enumerate(('X','Y','Z'))])
    del raw
    pose = np.asarray(row['lidar_pose_in_folder'], dtype=float)
    return (xyz-pose[:3,3])@pose[:3,:3]

def percentiles(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {'n':0}
    return dict(n=len(x), mean=float(np.mean(x)), **{f'p{p}':float(np.percentile(x,p)) for p in (0,10,25,50,75,90,95,99,100)})

def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
