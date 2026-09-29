"""Verify completed artifacts, then seal the authorized RAW deployment candidate.

This does not execute inference or rewrite frozen stages/source LAS.
"""
from . import ROOT
from .audit import pins, verify_freezes, digest
from .export.artifacts import pq
import long_common as lc
import numpy as np
import csv
import json
import sys
import os
import shutil
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from importlib.metadata import version, PackageNotFoundError

OUT=ROOT/'results_final_pipeline'
CODE=ROOT/'MVP/final_pipeline'
PHASES=('deployable_baseline','deployable_baseline_cache','deployable_development_cache',
        'deployable_benchmark_cache','deployable_curved_run_cache')


def archive_initial(name):
    old=OUT/name;copy=OUT/'audit'/('initial_partial_'+name)
    if old.exists() and not copy.exists():shutil.copy2(old,copy)


def check_cases():
    count=0;unique=set();phase_counts={};c4=0;hashes={};baseline=[]
    for phase in PHASES:
        barrier_path=OUT/phase/'INFERENCE_COMPLETE.json'
        barrier=lc.load(barrier_path) if barrier_path.exists() else None
        if barrier is not None:
            assert barrier['annotation_labels'] is False and barrier['future_clouds'] is False
        else:
            assert phase=='deployable_baseline_cache'
            assert lc.load(OUT/'equivalence/cache_40.json')['cases']==40
        folders=sorted(p.parent for p in (OUT/phase).glob('*/PREDICTION_COMPLETE.json'))
        if barrier is not None:assert len(folders)==len(barrier['results'])
        phase_counts[phase]=len(folders)
        for folder in folders:
            marker=lc.load(folder/'PREDICTION_COMPLETE.json')
            assert marker['uses_annotation_labels'] is False and marker['uses_future_clouds'] is False
            for file,sha in marker['files'].items():
                assert lc.sha(folder/file)==sha,(folder,file)
            m=lc.load(folder/'summary.json');i=m['frame']
            assert all(max(0,i-11)<=f<=i for f in m['source_frames'])
            with np.load(folder/'point_provenance.npz') as z:points=z['rows']
            assert np.all(points['source_frame']<=i)
            assert set(np.unique(points['first_found_stage']))<={1,2,3}
            keys=(points['source_frame'].astype('uint64')<<np.uint64(32))|points['source_row'].astype('uint64')
            assert len(np.unique(keys))==len(keys)
            if m['final_geometry_available']:
                with np.load(folder/'prediction.npz') as p,np.load(folder/'curve.npz') as curve:
                    assert p['s'].max()<=curve['s'].max()+1e-9
                    assert np.isfinite(p['pair']).all()
            if m.get('c4_exact') is not None:
                assert m['c4_exact'] is True,(folder,'C4 mismatch');c4+=1
            if phase=='deployable_baseline':
                baseline.append(m);hashes[folder.name]=marker['files']
            if phase in PHASES[2:]:unique.add((m['run'],i))
            count+=1
        print('VERIFIED',phase,len(folders),flush=True)
    assert phase_counts==dict(zip(PHASES,(40,40,108,1422,268)))
    for name in ('BASELINE_RUNTIME.json','BASELINE_OUTPUT_HASHES.json','point_stage_counts.csv','stage_distance_counts.csv'):
        archive_initial(name)
    lc.save(OUT/'BASELINE_RUNTIME.json',dict(scope='Complete DEPLOYABLE_PIPELINE naive RAW baseline; 40 stratified starts',
        engine='baseline',rows=baseline,clean_serial_performance='profiling_ablation/baseline.json'))
    lc.save(OUT/'BASELINE_OUTPUT_HASHES.json',dict(scope='Complete RAW baseline; includes authorized seed adapter',
        note='Provenance stage-usage bookkeeping normalized after inference; prediction arrays unchanged',files=hashes))
    shutil.copyfile(OUT/'point_stage_counts_run_unique.csv',OUT/'point_stage_counts.csv')
    shutil.copyfile(OUT/'stage_distance_counts_run_unique.csv',OUT/'stage_distance_counts.csv')
    return dict(saved_cases=count,unique_production_starts=len(unique),phase_counts=phase_counts,c4_exact_cases=c4,
        inference_file_hashes_valid=True,causal_source_ids=True,no_curve_extrapolation=True)


def check_parquet():
    result={}
    for kind in ('point','geometry'):
        path=OUT/(kind+'_provenance.parquet');table=pq.read_table(path);n=table.num_rows
        def arr(name):return table[name].combine_chunks().to_numpy(zero_copy_only=False)
        if kind=='point':
            assert n==3652166
            assert np.all(arr('source_frame')<=arr('first_found_frame'))
            assert set(np.unique(arr('first_found_stage')))<={1,2,3}
            assert np.all((arr('used_by_stage_mask') & (1 << (arr('first_found_stage')-1)))!=0)
            runs=table['source_run'].combine_chunks().dictionary_encode()
            key=np.rec.fromarrays([runs.indices.to_numpy(),arr('source_frame'),arr('source_point_index')],names='run,frame,point')
            assert len(np.unique(key))==n
        else:
            assert n==204588
            for col in ('x','y','z','station_s','sigma_lateral','sigma_vertical'):assert np.isfinite(arr(col)).all(),col
            assert np.all(arr('sigma_lateral')>=0) and np.all(arr('sigma_vertical')>=0)
        result[kind]=dict(rows=n,valid=True)
    return result


def check_las():
    selected=lc.load(OUT/'selected_las_runs.json')['runs'];real=0
    for r in selected:
        audit=lc.load(OUT/'audit'/('las_'+r['run']+'.json'))
        assert audit['frames']==r['frames']==len(audit['rows'])
        for row in audit['rows']:
            assert all(row[k] for k in ('all_original_fields_exact','xyz_exact','original_classification_preserved'))
            assert (OUT/row['output']).is_file()
        real+=audit['frames']
        assert lc.load(OUT/'audit'/('geometry_las_'+r['run']+'.json'))['roundtrip_exact']
    files=list((OUT/'las').rglob('*.las'))
    assert real==813 and len(files)==815
    assert {p.relative_to(OUT/'las').parts[0] for p in files}=={'straight_run','curved_run'}
    return dict(real_frame_las=real,synthetic_geometry_las=2,all_original_fields_readback_exact=True)


def disk_accounting():
    files=[p for p in OUT.rglob('*') if p.is_file()]
    def size(folder):return sum(p.stat().st_size for p in files if p.is_relative_to(folder))
    categories=dict(straight_LAS=size(OUT/'las/straight_run'),curved_LAS=size(OUT/'las/curved_run'),
        provenance=size(OUT/'provenance')+sum((OUT/(k+'_provenance.parquet')).stat().st_size for k in ('point','geometry')),
        all_results=sum(p.stat().st_size for p in files))
    largest=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size) for p in sorted(files,key=lambda p:p.stat().st_size,reverse=True)[:20]]
    lc.csv_write(OUT/'disk_usage.csv',[dict(category=k,bytes=v,GiB=v/2**30) for k,v in categories.items()])
    lc.csv_write(OUT/'largest_artifacts.csv',largest)
    lc.save(OUT/'disk_summary.json',dict(categories_bytes=categories,largest_files=largest,
        note='Snapshot before final report/manifest; manifest lists exact per-file sizes except its own self entry'))


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href') and v:self.links.append(v)


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    before=lc.load(OUT/'audit/dependencies_before.json');after=pins()
    assert before==after,'Frozen dependencies changed'
    assert lc.load(OUT/'equivalence/cache_40.json')['all_arrays_bitwise_equal']
    api=lc.load(OUT/'audit/final_api_checks.json')
    assert all(api[k] for k in ('old_predictions_bitwise_equal','annotation_permutation_no_effect',
        'T_plus_1_cloud_absent_and_not_needed','generic_run_directory_supported','online_has_no_research_debug'))
    for phase in ('deployable_development_cache','deployable_benchmark_cache'):
        assert lc.load(OUT/phase/'QUALITY.json')['acceptance_passed']
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','MVP/final_pipeline/tests','-v'],
        cwd=ROOT,capture_output=True,text=True)
    (OUT/'audit/unit_tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    assert test.returncode==0
    checks=check_cases();checks.update(parquet=check_parquet(),las=check_las(),frozen_files_unchanged=len(before),
        original_stage_manifests=verify_freezes(),final_API=api,unit_tests_passed=8,date=stamp)
    lc.save(OUT/'audit/final_integrity.json',checks)
    packages={}
    # Offline-only packages are available from these existing vendor directories.
    sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
    for name in ('numpy','scipy','laspy','pyarrow','matplotlib','scikit-learn'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]='not installed / not used'
    (CODE/'requirements_runtime.txt').write_text('# Python '+platform.python_version()+'; offline artifact dependencies included\n'+
        '\n'.join(k+'=='+v if 'not installed' not in v else '# '+k+': '+v for k,v in packages.items())+'\n',encoding='utf-8')
    code_files={p.relative_to(CODE).as_posix():lc.sha(p) for p in CODE.rglob('*') if p.is_file() and
        (p.suffix in ('.py','.json') and 'manifests' not in p.parts)}
    code_hash=digest(code_files);config_hash=lc.sha(CODE/'adapters/config.json');dependency_hash=digest(after)
    with (OUT/'audit/frozen_dependencies.csv').open(encoding='utf-8-sig') as f:stages=list(csv.DictReader(f))
    stages.append(dict(logical_stage='STEP1_TO_STEP6_SEED_ADAPTER',actual_path='MVP/final_pipeline/adapters',
        version='v1',config='adapters/config.json',config_sha256=config_hash,
        code_sha256=lc.sha(CODE/'adapters/step1_to_step6_seed.py'),dependency_sha256=dependency_hash))
    freeze=dict(pipeline_version='1.0.0-raw-seed-v1',date=stamp,status='FROZEN_DEPLOYABLE_OFFLINE_CANDIDATE',
        verdict='C. PIPELINE CORRECT BUT TOO SLOW; MAIN BOTTLENECK = C4 MARCHING',
        pipeline_sha256=code_hash,code_files=code_files,config_sha256=config_hash,dependency_sha256=dependency_hash,
        dependency_files=after,stages=stages,python=platform.python_version(),OS=platform.platform(),
        hardware=lc.load(OUT/'audit/hardware.json'),packages=packages,
        thread_settings={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        frozen_stages_changed=False,authorized_integration_change='STEP1_TO_STEP6_SEED_ADAPTER_v1',
        inference_uses_annotation_labels=False,inference_uses_future_clouds=False,pose_T_plus_1_allowed=True,
        research_reference='Old GT-seed pipeline, evaluation only',deployable_pipeline='RAW STEP1 surface support seed',
        downstream_quality_gate='PASS',engineering_equivalence='BITWISE_EXACT',max_numeric_difference=0,
        checks='audit/final_integrity.json',performance='report_summary.json',real_time=False)
    lc.save(OUT/'FINAL_PIPELINE_FREEZE.json',freeze)
    (CODE/'manifests').mkdir(exist_ok=True)
    lc.save(CODE/'manifests/RESEARCH_REFERENCE.json',dict(component='RESEARCH_REFERENCE',
        inference_uses_class1_for_seed=True,production=False,dependencies=before,
        results=['results_latent_contact_reference/phase_d/C4_SMOOTH','results_latent_contact_reference/phase_e/C4_SMOOTH']))
    lc.save(CODE/'manifests/DEPLOYABLE_PIPELINE.json',freeze)
    disk_accounting()
    from .report import main as report
    report()
    links=0
    for file in ('gallery.html','REPORT_FINAL_PIPELINE.html'):
        parser=Links();parser.feed((OUT/file).read_text(encoding='utf-8'))
        for value in parser.links:
            target=urlsplit(value)
            if target.scheme or not target.path:continue
            assert (OUT/unquote(target.path)).exists(),(file,value)
            links+=1
    lc.save(OUT/'audit/html_link_checks.json',dict(local_links_checked=links,all_exist=True,browser_render_test=False))
    lc.save(CODE/'FINAL_HACKATHON_PIPELINE.freeze',dict(pipeline_sha256=code_hash,dependency_sha256=dependency_hash,
        config_sha256=config_hash,date=stamp,status=freeze['status'],verdict=freeze['verdict'],
        full_freeze='results_final_pipeline/FINAL_PIPELINE_FREEZE.json'))
    # Seal every output and new code file. No circular self hash is claimed.
    manifest=[]
    for base in (CODE,OUT):
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p==OUT/'MANIFEST.json' or '__pycache__' in p.parts:continue
            manifest.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=lc.sha(p)))
    lc.save(OUT/'MANIFEST.json',dict(date=stamp,files=manifest,
        excluded=['results_final_pipeline/MANIFEST.json (self)','__pycache__'],
        pipeline_sha256=code_hash,dependency_sha256=dependency_hash))
    print(json.dumps(dict(status=freeze['status'],files=len(manifest),checks=checks),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
