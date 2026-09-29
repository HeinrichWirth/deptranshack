"""STEP6 private paths and I/O. Frozen dependencies are read-only."""
import os, sys, json, hashlib, csv, time
from pathlib import Path
sys.dont_write_bytecode=True
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parents[2]
OUT=ROOT/'results_running_rails_from_contact'
C4ROOT=ROOT/'results_contact_marching_long_range'
C4STAGE=ROOT/'MVP/stages/05_2_contact_marching_long_range'
BASE=ROOT/'results_contact_marching'
sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np
import laspy
from functools import lru_cache

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:clean(v) for k,v in r.items()} for r in rows])
def key(run,i):return f'{run}__{i:06d}'
def stats(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),**({f'p{p}':float(np.percentile(a,p)) for p in (0,50,90,95,99,100)} if len(a) else {}))
def dataset():return Path(load(ROOT/'MVP/stages/05_contact_rail_marching/dataset.json')['data_root'])
@lru_cache(maxsize=16)
def frames(run):return load(next((dataset()/run).glob('*.frames.json')))['frames']
def check_dependencies():
    lock=load(STAGE/'dependency_lock.json')
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,('Frozen dependency changed',rel)
    return lock
def c4_path(run,i):
    split=load(OUT/'protocol.json')['split']
    group='phase_d_combinations' if run in split['development'] else 'phase_e_benchmark'
    return C4ROOT/group/'C4'/key(run,i)
def read_c4(run,i):
    """Only completed, pinned causal C4 predictions. Never recomputes C4."""
    folder=c4_path(run,i);mark=load(folder/'PREDICTION_COMPLETE.json')
    index=load(STAGE/'c4_input_index.json')[key(run,i)]
    assert sha(folder/'PREDICTION_COMPLETE.json')==index['marker_sha256']
    for name,h in mark['files'].items():assert sha(folder/name)==h,(folder,name)
    p=load(folder/'prediction.json')
    assert p['variant']=='C4' and p['start_frame']==i and p['run']==run
    with np.load(folder/'points.npz') as z:p.update({k:z[k] for k in z.files})
    with np.load(folder/'provenance.npz') as z:points={k:z[k] for k in z.files}
    assert np.all(points['source_frame']<=i)
    assert np.all(points['age_frames']==i-points['source_frame'])
    return p,points
