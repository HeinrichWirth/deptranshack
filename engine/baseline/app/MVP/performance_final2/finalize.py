from . import ROOT,OUT
from MVP.final_pipeline.audit import pins
import long_common as lc
import subprocess,sys,hashlib,json,time,platform

REQUIRED='REPORT_C4_OPTIMIZATION_FINAL.md REPORT_C4_OPTIMIZATION_FINAL.html ASYNC_GEOMETRY_REPORT.md hypothesis_results.csv waterfall.csv c4_before_after.csv call_counts.csv nn_benchmark.csv tree_reuse.csv residual_benchmark.csv jacobian_benchmark.csv candidate_batch_benchmark.csv thread_scaling.csv native_equivalence.csv full_equivalence.csv async_runs.csv geometry_age.csv horizon_age.csv docker_benchmark.csv memory.csv FINAL_PERFORMANCE_RECOMMENDATION.md'.split()

def main():
    original=lc.load(ROOT/'results_final_pipeline/FINAL_PIPELINE_FREEZE.json');prior=lc.load(ROOT/'results_performance_final/freeze.json')
    assert pins()==original['dependency_files']
    for name,digest in original['code_files'].items():assert lc.sha(ROOT/'MVP/final_pipeline'/name)==digest,name
    for name,digest in prior['files'].items():assert lc.sha(ROOT/name)==digest,name
    eq=lc.load(OUT/'FINAL_EQ_COMPLETE.json');assert eq['cases']==200 and eq['bitwise'] and eq['max_difference']==0
    final=lc.load(OUT/'FINAL_BUILD_RECHECK.json');assert final['cases']==200 and final['bitwise']
    assert lc.load(ROOT/'MVP/performance_final2/runtime_config.json')['thresholds']==lc.load(OUT/'micro_summary.json')['thresholds']
    assert final['native_sha256']==lc.sha(ROOT/'MVP/performance_final2/native/_c4_sprint.cp312-win_amd64.pyd')
    rows=lc.csv_read(OUT/'full_equivalence.csv');assert len(rows)==200 and all(r['bitwise']=='True' for r in rows)
    for marker in (OUT/'equivalence').glob('*/*/PREDICTION_COMPLETE.json'):
        m=lc.load(marker);assert not m['labels_used'] and m['original_hash']==m['best_hash']
        for name,digest in m.get('files',{}).items():assert lc.sha(marker.parent/name)==digest
    for variant,n in [('docker_portable',200),('docker_host',32)]:
        d=lc.load(OUT/variant/'DOCKER_COMPLETE.json');assert d['cases']==n and d['bitwise'] and d['no_build_tools']
        micro=lc.load(OUT/variant/'linux_micro_summary.json');assert micro['max_residual']==0 and micro['max_jacobian']==0 and micro['fd_fits_passed']
    paths=list((OUT/'async').glob('*/*/result.json'));assert len(paths)==6
    for path in paths:
        b=lc.load(path);assert b['max_pending']<=1
        assert [r['frame'] for r in b['frames']]==list(range(len(b['frames'])))
        for r in b['frames']:
            assert r['pending_depth']<=1
            if r['state_available']:assert r['geometry_source_frame']<=r['frame'] and r['geometry_age_seconds']>=0
        for r in b['solves']:assert max(r['source_frames'])<=r['frame']<r['ready_pose']
    assert lc.load(OUT/'async_live.json')['max_pending']<=1
    for name in REQUIRED:assert (OUT/name).stat().st_size>0,name
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','MVP/performance_final2/tests','-v'],cwd=ROOT,check=True)
    code={p.relative_to(ROOT).as_posix():lc.sha(p) for p in (ROOT/'MVP/performance_final2').rglob('*') if p.is_file() and not any(x in p.parts for x in ('build','__pycache__'))}
    summary=lc.load(OUT/'report_summary.json');assert summary['default_native_gate']
    checks=dict(frozen_dependencies_unchanged=True,previous_stages_unchanged=True,windows_full_starts=200,linux_portable_full_starts=200,
        linux_host_full_starts=32,all_bitwise=True,discrete_changes=0,native_residual_and_FD_max_difference=0,async_full_runs=len(paths),
        async_raw_frames=sum(len(lc.load(p)['frames']) for p in paths),max_pending=1,unit_tests=8,gt_used_in_inference=False,future_clouds=False)
    lc.save(OUT/'checks.json',checks)
    lc.save(OUT/'freeze.json',dict(version='performance-v2',time_ns=time.time_ns(),production_backend='native_batch_spatial',internal_alias='batch',
        workers=1,BLAS_threads=1,verdict='D',geometry_changed=False,solver='frozen scipy least_squares TRF cauchy',
        native_solver_spike='EXPERIMENTAL rejected; not connected',batching='partial: retained SciPy/Python per-seed callbacks',
        thresholds={'forward':256,'reverse':1024},equivalence_absolute_tolerance=1e-10,discrete_tolerance=0,
        code_sha256=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),files=code,
        previous_freeze_sha256=lc.sha(ROOT/'results_performance_final/freeze.json'),original_pipeline_sha256=original['pipeline_sha256'],
        dependency_files=original['dependency_files'],checks=checks,metrics=summary['metrics'],final_confirmation=summary['final_confirmation'],Python=platform.python_version(),OS=platform.platform()))
    lc.save(OUT/'MANIFEST.json',dict(self_excluded=True,files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=lc.sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print('FROZEN',checks,flush=True)

if __name__=='__main__':main()
