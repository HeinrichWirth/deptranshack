from . import ROOT,OUT
import long_common as lc
import shutil


def main():
    source=ROOT/'MVP/c4_v2';context=OUT/'docker_context';context.mkdir(exist_ok=True)
    for p in source.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('build','__pycache__')):continue
        if p.suffix not in ('.py','.md','.json','.txt'):continue
        target=context/'app/MVP/c4_v2'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('solver.cpp','march.hpp','CMakeLists.txt'):
        target=context/'native'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(source/'native'/name,target)
    shutil.copy2(source/'docker/Dockerfile',context/'Dockerfile')
    lc.save(OUT/'docker_context_manifest.json',dict(files={p.relative_to(context).as_posix():lc.sha(p) for p in context.rglob('*') if p.is_file()},data_in_image=False,portable=True))

if __name__=='__main__':main()
