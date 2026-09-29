"""Frozen stages plus the separately authorized RAW STEP1-to-STEP6 adapter."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.runtime'),
               str(ROOT / 'MVP/stages/05_2_contact_marching_long_range/src'),
               str(ROOT / 'MVP/stages/06_running_rails_from_contact/src'),
               str(ROOT / 'MVP/stages/06_final_geometry/src')]


def process_run(run_dir, config=None):
    from .pipeline import process_run as run
    return run(run_dir, config)
