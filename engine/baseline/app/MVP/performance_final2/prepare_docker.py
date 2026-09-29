from . import ROOT,OUT
import long_common as lc
import shutil

def main():
    source=ROOT/'MVP/performance_final2';context=OUT/'docker_context';context.mkdir(exist_ok=True)
    for p in source.rglob('*'):
        if not p.is_file() or any(k in p.parts for k in ('build','__pycache__')):continue
        if p.suffix not in ('.py','.md','.txt','.json'):continue
        target=context/'app/MVP/performance_final2'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    p=ROOT/'MVP/performance_final/tests/test_scheduler.py';target=context/'app/MVP/performance_final/tests/test_scheduler.py';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('kernel.cpp','CMakeLists.txt'):
        target=context/'native'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(source/'native'/name,target)
    shutil.copy2(source/'docker/Dockerfile',context/'Dockerfile')
    for folder in ('docker_portable','docker_host'):
        target=OUT/folder;target.mkdir(exist_ok=True)
        for name in ('benchmark_cohort.json','selection.json','micro_summary.json'):shutil.copy2(OUT/name,target/name)
    lc.save(OUT/'docker_context_manifest.json',dict(files={p.relative_to(context).as_posix():lc.sha(p) for p in context.rglob('*') if p.is_file()},
        data_in_image=False,base_image='deptrans-final-runtime:performance-v1'))

if __name__=='__main__':main()
