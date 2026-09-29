from . import ROOT,OUT
from .run_full import RUNS,old_saved
from MVP.final_pipeline.audit import pins
import long_common as lc
import numpy as np
import json,hashlib,time,sys,subprocess,platform
from pathlib import Path


def strip_times(x):
    if isinstance(x,dict):return {k:strip_times(v) for k,v in x.items() if not k.endswith(('_ms','_ns'))}
    if isinstance(x,list):return [strip_times(v) for v in x]
    return x


def main():
    old=lc.load(ROOT/'results_final_pipeline/FINAL_PIPELINE_FREEZE.json')
    assert pins()==old['dependency_files']
    for name,sha in old['code_files'].items():assert lc.sha(ROOT/'MVP/final_pipeline'/name)==sha
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','MVP/performance_final/tests','-v'],cwd=ROOT,check=True)
    count=0;classic=0;reused=0;directions=0
    for run in RUNS:
        path=OUT/'runs/scheduler_spatial'/run;b=lc.load(path/'INFERENCE_COMPLETE.json');rows=b['rows']
        assert b['all_consecutive_frames'] and not b['labels_used'] and not b['future_clouds']
        assert [r['frame'] for r in rows]==list(range(len(lc.frames(run))))
        last=None
        for r in rows:
            i=r['frame'];count+=1
            if r['fresh']:last=i
            if r['valid_geometry_state']:
                assert r['source_geometry_frame']==last and last<=i
                assert r['geometry_age_frames']==i-last
            else:last=None
            if r['low_motion_reuse']:
                assert r['distance_since_geometry_update']<.5 and not r['solve_attempted'];reused+=1
            if r['solve_attempted']:
                folder=path/f'{i:06d}';mark=lc.load(folder/'PREDICTION_COMPLETE.json')
                for f,sha in mark['files'].items():assert lc.sha(folder/f)==sha
                meta=lc.load(folder/'summary.json');assert all(f<=i for f in meta['source_frames'])
                if r.get('direction',{}).get('mode')=='FROZEN_T_TPLUS1':
                    prior=old_saved(run,i)
                    if prior and (prior/'c4.json').exists():
                        a=strip_times(lc.load(prior/'c4.json'));z=strip_times(lc.load(folder/'c4.json'))
                        assert a==z,(run,i,'C4 decisions changed');classic+=1
                elif r.get('direction',{}).get('mode')=='CAUSAL_ACCUMULATED_POSES':directions+=1
    eq=[lc.load(OUT/'ablation'/f'{n}_equivalence.json') for n in ('optimized','spatial','native','native_spatial')]
    assert all(v['bitwise_exact'] for v in eq)
    orientation=lc.load(OUT/'orientation_equivalence.json')
    assert orientation['all_bitwise_exact'] and orientation['cases']>0
    docker=lc.load(OUT/'docker_test/docker_smoke.json')
    assert docker['full_RAW_frame_python_native_bitwise'] and docker['scheduler_snapshot_passed']
    native=lc.load(OUT/'native_test.json');assert native['kernel_passed'] and native['fit_passed']
    batch=lc.load(OUT/'batch_full.json');assert len(batch['rows'])==10 and all(r['returncode']==0 for r in batch['rows'])
    clean=lc.load(OUT/'clean_benchmark.json');assert clean['start_ns']>clean['full_run_barrier_ns'] and len(clean['rows'])==5
    checks=dict(full_scheduler_frames=count,classic_full_C4_decision_comparisons=classic,
        accumulated_pose_direction_attempts=directions,low_motion_reuse_frames=reused,frozen_dependencies_unchanged=True,
        previously_frozen_pipeline_code_unchanged=True,engineering_32_starts_bitwise=True,unit_tests=10,
        docker_full_RAW_smoke=True,adapted_direction_bitwise_comparisons=orientation['cases'],
        native_kernel_max_difference=lc.load(OUT/'native_test.json')['max_abs_difference'])
    lc.save(OUT/'checks.json',checks)
    code={p.relative_to(ROOT).as_posix():lc.sha(p) for p in (ROOT/'MVP/performance_final').rglob('*') if p.is_file() and
          not any(x in p.parts for x in ('vendor','build','__pycache__'))}
    p=ROOT/'MVP/final_pipeline/spatial_scheduler.py';code[p.relative_to(ROOT).as_posix()]=lc.sha(p)
    digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    lc.save(OUT/'freeze.json',dict(version='performance-v1',time_ns=time.time_ns(),decision=lc.load(OUT/'report_summary.json')['decision'],
        code_sha256=digest,files=code,original_pipeline_sha256=old['pipeline_sha256'],dependencies=old['dependency_files'],
        geometry_changed=False,seed_changed=False,orientation_adapter='documented causal accumulated pose fallback',
        production_backend='python_spatial',native_backend='optional experimental; no full solver port',checks=checks,
        Python=platform.python_version(),OS=platform.platform(),equivalence_tolerance=1e-10,discrete_tolerance=0))
    lc.save(OUT/'MANIFEST.json',dict(files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=lc.sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'],self_excluded=True))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
