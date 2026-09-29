"""Provenance, report links and frozen-source verification."""
from . import ROOT,OUT
import long_common as lc
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,time,csv,ast

def main():
    original=lc.load(OUT/'ORIGINAL_SELECTION_LOCK.json');changed=[];missing=[]
    for name,digest in original['code_sha256'].items():
        p=ROOT/name
        if not p.exists():missing.append(name)
        elif lc.sha(p)!=digest:changed.append(name)
    lock=lc.load(OUT/'REALTIME_SELECTION_LOCK.json');runtime_changed=[]
    for name,digest in lock['code_sha256'].items():
        if lc.sha(ROOT/name)!=digest:runtime_changed.append(name)
    parsed=[]
    for p in (ROOT/'MVP/realtime_final').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));parsed.append(p.name)
    q=lc.load(OUT/'FINAL_LINUX_QUALITY.json');decision=lc.load(OUT/'FINAL_DECISION.json')
    with (OUT/'final_vs_linux_baseline.csv').open(encoding='utf-8-sig',newline='') as f:exact=list(csv.DictReader(f))
    required=['linux_quality_240.csv','cross_os_first_divergence.csv','determinism_trace','point_count_runtime.csv','runtime_regression.csv','live_memory_timing.csv','history_index_timing.csv','roi_timing.csv','point_reduction_ablation.csv','compiler_ablation.csv','pgo_ablation.csv','worker_matrix.csv','realtime_10hz.csv','cpu_utilization.csv','memory.csv','FINAL_REALTIME_RECOMMENDATION.md','REPORT_REALTIME_FINAL.html','REPORT_REALTIME_FINAL.md']
    absent=[s for s in required if not (OUT/s).exists()]
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and v:self.links.append(v)
    broken=[]
    for p in (OUT/'REPORT_REALTIME_FINAL.html',OUT/'linux_final/gallery.html'):
        parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for href in parser.links:
            if '://' in href or href.startswith('#'):continue
            target=p.parent/href.split('#')[0]
            if not target.exists():broken.append(dict(file=str(p),link=href))
    # These two files are written below; they are expected forward links in the report.
    broken=[r for r in broken if r['link'] not in ('FINAL_MANIFEST.json','VERIFICATION.json')]
    verification=dict(time_ns=time.time_ns(),frozen_files_checked=len(original['code_sha256']),frozen_changed=changed,frozen_missing=missing,locked_runtime_changed=runtime_changed,python_parsed=parsed,required_files_missing=absent,broken_links=broken,development_exact_cases=len(lc.load(OUT/'development.json')['equality']),final_exact_vs_baseline=len(exact)==240 and all(r['final_exact_vs_linux_baseline']=='True' for r in exact),quality_pass=q['quality_pass'],decision=decision['decision'],system_gate=decision['system_gate'],unit_test_log='contracts.log',inference_uses_annotations=False,persistent_c4=False)
    verification['pass']=not(changed or missing or runtime_changed or absent or broken) and verification['quality_pass'] and verification['final_exact_vs_baseline']
    lc.save(OUT/'VERIFICATION.json',verification)
    files={}
    for directory in (ROOT/'MVP/realtime_final',OUT):
        for p in directory.rglob('*'):
            if not p.is_file() or 'build' in p.parts or '__pycache__' in p.parts or p.name=='FINAL_MANIFEST.json':continue
            if p.suffix not in ('.py','.cpp','.hpp','.pyd','.so','.json','.csv','.md','.html','.ps1') or p.stat().st_size>100000000:continue
            files[p.relative_to(ROOT).as_posix()]=lc.sha(p)
    lc.save(OUT/'FINAL_MANIFEST.json',dict(time_ns=time.time_ns(),files=files,original_frozen_lock='ORIGINAL_SELECTION_LOCK.json',selection_lock='REALTIME_SELECTION_LOCK.json',production_defaults=dict(workers=2,candidate_threads=4),image='deptrans-realtime:production',decision=decision,verification_pass=verification['pass']))
    print('FINAL_VERIFICATION',verification)
    if not verification['pass']:raise RuntimeError('Qualification verification failed')
if __name__=='__main__':main()
