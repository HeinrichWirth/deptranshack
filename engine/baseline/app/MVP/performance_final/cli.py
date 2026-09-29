"""Streaming scheduler CLI; default verified Python backend, native opt-in."""
from . import ROOT
from .engine import Engine
from MVP.final_pipeline.spatial_scheduler import SpatialScheduler
import long_common as lc
import numpy as np
import argparse,os,json,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run');ap.add_argument('--output');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--backend',choices=['python','native'],default=os.environ.get('C4_BACKEND','python'));a=ap.parse_args()
    e=Engine('spatial' if a.backend=='python' else 'native_spatial')
    if a.self_test:
        print(json.dumps(dict(status='IMPORT_ABI_OK',backend=a.backend)));return
    if not a.run or not a.output:ap.error('--run and --output are required')
    run=Path(a.run).resolve();out=Path(a.output).resolve()
    if out==run or out.is_relative_to(run) or out.is_relative_to(ROOT/'MVP/stages'):raise ValueError('Output cannot overwrite input or frozen stages')
    s=SpatialScheduler(e);s.start_run(run);out.mkdir(parents=True,exist_ok=True)
    for i in range(len(e.records)):
        result=s.process_frame(i);row=result['summary'];directory=out/f'{i:06d}';directory.mkdir(exist_ok=True)
        lc.save(directory/'summary.json',row)
        # Reuse is an explicit source pointer. Keep one map state per update.
        if row['fresh']:
            np.savez_compressed(directory/'map_prediction.npz',**s.last_valid_geometry['prediction'])
            np.savez_compressed(directory/'map_curve.npz',**s.last_valid_geometry['curve'])
        print(json.dumps(lc.clean(row)),flush=True)
    lc.save(out/'RUN_COMPLETE.json',dict(backend=a.backend,rows=s.rows,time_ns=time.time_ns()))


if __name__=='__main__':main()
