"""Linux native tests plus whole-chain equivalence, no dataset in the image."""
from . import ROOT,OUT
from .engine import Engine
from . import micro
from MVP.performance_final.ablation import fingerprint
import long_common as lc
import numpy as np
import subprocess,sys,shutil,time,platform,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def equivalence_run(job):
    run,cases,backend=job;engines=[Engine('python'),Engine(backend)];rows=[]
    for e in engines:e.start_run(Path('/data')/run)
    for case in cases:
        values=[e.process_frame(case['frame']) for e in engines];exact=fingerprint(values[0])==fingerprint(values[1])
        rows.append(dict(**case,bitwise=exact,original_hash=fingerprint(values[0]),best_hash=fingerprint(values[1])));assert exact,case
    for e in engines:e.close()
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic-host',action='store_true');ap.add_argument('--wait-for-timing',action='store_true');a=ap.parse_args()
    assert all(shutil.which(t) is None for t in ('gcc','g++','cmake','make'))
    pinned=lc.load(OUT/'micro_summary.json');micro.main();shutil.copy2(OUT/'micro_summary.json',OUT/'linux_micro_summary.json');lc.save(OUT/'micro_summary.json',pinned)
    from .extra_micro import main as extra_micro
    extra_micro()
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','MVP/performance_final2/tests','-v'],check=True)
    cohort=lc.load(OUT/'benchmark_cohort.json');backend=lc.load(OUT/'selection.json')['backend']
    selected=cohort['clean'] if a.diagnostic_host else cohort['final'];jobs=[]
    for run in dict.fromkeys(c['run'] for c in selected):jobs.append((run,[c for c in selected if c['run']==run],backend))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(equivalence_run,jobs):rows.extend(result);print('LINUX_EQ',len(rows),len(selected),flush=True)
    lc.csv_write(OUT/'linux_full_equivalence.csv',rows)
    if a.wait_for_timing:
        print('WAITING_TIMING_PERMISSION',flush=True)
        while not (OUT/'ALLOW_TIMING').exists():time.sleep(1)
    # Dedicated serial timing after all equivalence processes have exited.
    timings=[];run=None;engines={}
    for case in cohort['clean']:
        if case['run']!=run:
            for e in engines.values():e.close()
            run=case['run'];engines={m:Engine(m) for m in ('python','native_spatial',backend)}
            for e in engines.values():e.start_run(Path('/data')/run)
        hashes=[]
        for name,e in engines.items():
            r=e.process_frame(case['frame']);hashes.append(fingerprint(r));timings.append(dict(**case,backend=name,**r['summary']['timing']))
        assert len(set(hashes))==1
    for e in engines.values():e.close()
    lc.csv_write(OUT/'docker_timings.csv',timings)
    lc.save(OUT/'DOCKER_COMPLETE.json',dict(platform=platform.platform(),python=sys.version,cases=len(rows),bitwise=True,
        no_build_tools=True,host_optimized=a.diagnostic_host,compiler=Path('/app/compiler2.txt').read_text(),
        cmake=Path('/app/cmake2.txt').read_text(),time_ns=time.time_ns()))
    print('DOCKER_FINAL2_PASSED',len(rows),flush=True)

if __name__=='__main__':main()
