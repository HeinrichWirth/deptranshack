"""Serial fresh-process engineering ablation; no evaluator imports."""
from . import ROOT
from .pipeline import Pipeline
from .verify_raw import normalized
from .memory import rss
from fusion import CausalSource
import long_common as lc
import hashlib
import json
import subprocess
import sys
import os
import time
import argparse

OUT=ROOT/'results_final_pipeline'
COHORT=[('roundT_squareT_pressureGate_squareT',100,'straight'),
        ('roundT_pressureGate_roundT',72,'curved'),
        ('squareT_platform_squareT_switch',300,'long'),
        ('new_data_part_01d1',550,'sparse')]


def child(variant):
    ready=time.time_ns();records=[]
    for run,start,category in COHORT:
        init=time.perf_counter();p=Pipeline(dict(engine='baseline' if variant=='baseline' else ('raw_cache' if variant=='raw_cache' else 'cache'),preloaded=variant=='preloaded'))
        p.start_run(lc.dataset()/run);initialization=time.perf_counter()-init
        base_memory=rss()
        for i in range(start,start+8):
            preload=0.
            if variant=='preloaded':
                begin=time.perf_counter();src=CausalSource(run,i,p.records[:i+1])
                ids,_=src.history(dict(frames=12))
                for k in reversed(ids):p.cache.get(k,i)
                preload=time.perf_counter()-begin
            before=rss();result=p.process_frame(i);after=rss()
            algorithm={k:normalized(result[k]) for k in ('step1','step2','c4','seed','prediction','c4_smooth','provenance')}
            checksum=hashlib.sha256(json.dumps(algorithm,sort_keys=True).encode()).hexdigest()
            row=dict(result['summary'],variant=variant,category=category,first_frame=i==start,
                initialization_seconds=initialization if i==start else 0.,preload_seconds=preload,
                baseline_rss=base_memory['rss_bytes'],rss_before=before['rss_bytes'],**after,output_sha256=checksum)
            if not records:row['fresh_process_startup_seconds']=(ready-int(os.environ.get('PIPELINE_LAUNCH_NS',ready)))/1e9
            records.append(row);del result
            print('TIMING',variant,run,i,round(row['timing']['T_TOTAL'],4),flush=True)
    lc.save(OUT/'profiling_ablation'/f'{variant}.json',dict(rows=records,variant=variant,process_id=os.getpid(),scope='32 starts, 4 runs x 8 sequential frames, serial fresh process, no external evaluator'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['baseline','raw_cache','cache','preloaded']);a=ap.parse_args()
    if a.variant:child(a.variant);return
    for variant in ('baseline','raw_cache','cache','preloaded'):
        env=os.environ.copy();env['PIPELINE_LAUNCH_NS']=str(time.time_ns())
        subprocess.run([sys.executable,'-B','-m','MVP.final_pipeline.benchmark','--variant',variant],cwd=ROOT,env=env,check=True)
    baseline=lc.load(OUT/'profiling_ablation/baseline.json')['rows']
    for variant in ('raw_cache','cache','preloaded'):
        rows=lc.load(OUT/'profiling_ablation'/f'{variant}.json')['rows']
        assert [(r['run'],r['frame'],r['output_sha256']) for r in rows]==[(r['run'],r['frame'],r['output_sha256']) for r in baseline],variant
    lc.save(OUT/'equivalence/streaming_32.json',dict(cases=32,variants=4,all_outputs_bitwise_equal=True,max_difference=0))


if __name__=='__main__':main()
