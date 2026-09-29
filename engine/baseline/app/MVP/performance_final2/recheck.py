"""Final build against already frozen per-start original hashes; no retuning."""
from . import OUT
from .engine import Engine
from .native_api import startup
from MVP.performance_final.ablation import fingerprint
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import long_common as lc

def run(job):
    name,cases=job;e=Engine('native_batch_spatial');e.start_run(lc.dataset()/name);rows=[]
    for c in cases:
        expected=lc.load(OUT/'equivalence'/name/f'{c["frame"]:06d}'/'PREDICTION_COMPLETE.json')['original_hash']
        actual=fingerprint(e.process_frame(c['frame']));assert actual==expected,c
        rows.append(dict(**c,bitwise=True,expected=expected,actual=actual))
    e.close();return rows

def main():
    cases=lc.load(OUT/'benchmark_cohort.json')['final'];jobs=[(r,[c for c in cases if c['run']==r]) for r in dict.fromkeys(c['run'] for c in cases)];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for rr in pool.map(run,jobs):rows.extend(rr);print('FINAL_BUILD_RECHECK',len(rows),200,flush=True)
    lc.csv_write(OUT/'final_build_recheck.csv',rows);lc.save(OUT/'FINAL_BUILD_RECHECK.json',dict(cases=len(rows),bitwise=True,
        backend='native_batch_spatial',native_sha256=lc.sha(Path(startup().__file__))))

if __name__=='__main__':main()
