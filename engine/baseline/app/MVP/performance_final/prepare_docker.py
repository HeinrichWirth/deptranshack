from . import ROOT,OUT
import shutil
from pathlib import Path
import long_common as lc


def main():
    stage=ROOT/'MVP/performance_final';context=OUT/'docker_context';context.mkdir(exist_ok=True)
    pinned=lc.load(ROOT/'results_final_pipeline/FINAL_PIPELINE_FREEZE.json')['dependency_files']
    paths=set(pinned)
    paths.add('MVP/stages/03_contact_generalization/results_contact_generalization/protocol.json')
    for directory in (ROOT/'MVP/final_pipeline',stage):
        paths.update(p.relative_to(ROOT).as_posix() for p in directory.rglob('*.py') if not any(x in p.parts for x in ('vendor','build','tests','docker')))
    paths.add('MVP/final_pipeline/adapters/config.json')
    # Headers/templates frozen pins include numeric artifacts used at startup.
    for path in paths:
        source=ROOT/path;target=context/'app'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in ('kernel.cpp','CMakeLists.txt'):
        target=context/'native'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(stage/'native'/name,target)
    shutil.copy2(stage/'docker/Dockerfile',context/'Dockerfile')
    (context/'requirements_runtime.txt').write_text('numpy==2.5.3\nscipy==1.18.1\nlaspy==2.7.0\n',encoding='utf-8')
    lc.save(OUT/'docker_context_manifest.json',dict(files={p.relative_to(context).as_posix():lc.sha(p) for p in context.rglob('*') if p.is_file()},
        no_source_LAS=True,no_compiler_in_runtime=True,native_default=False,portable_flags=True))
    print('DOCKER_CONTEXT',context,len(paths),flush=True)


if __name__=='__main__':main()
