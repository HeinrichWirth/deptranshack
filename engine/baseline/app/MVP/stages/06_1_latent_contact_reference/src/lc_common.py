"""STEP6.1 paths. Frozen STEP6 modules are imported read-only, never repointed."""
import os,sys,time,json,csv,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
STAGE=Path(__file__).resolve().parents[1];ROOT=STAGE.parents[2];OUT=ROOT/'results_latent_contact_reference';OLD_STAGE=ROOT/'MVP/stages/06_running_rails_from_contact';OLD_OUT=ROOT/'results_running_rails_from_contact'
sys.path.insert(0,str(OLD_STAGE/'src'))
import rr_common as frozen
from rr_common import np,load,save,sha,write_csv,clean,key,stats,dataset,frames,read_c4,c4_path

def check_baseline():
    lock=load(OUT/'baseline_freeze.json')
    for path,digest in lock['files'].items():assert sha(ROOT/path)==digest,('Frozen baseline changed',path)
    frozen.check_dependencies()
    return lock

def require_reproduction():
    result=load(OUT/'reproduction/REPRODUCTION_COMPLETE.json');assert result['matched'] is True
    return result
