"""Final packaging/default alias gate after report and CLI files are assembled."""
from . import ROOT,OUT
from .engine import Engine
from .native_api import startup
from MVP.performance_final.ablation import fingerprint
import long_common as lc
from pathlib import Path
import subprocess,sys,hashlib,os

def main():
    assert os.environ['C4_BACKEND']=='native_batch_spatial'
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','MVP/performance_final2/tests','-v'],check=True)
    case=lc.load(OUT/'benchmark_cohort.json')['clean'][0];hashes=[]
    for name in ('python','native_batch_spatial'):
        e=Engine(name);e.start_run(Path('/data')/case['run']);hashes.append(fingerprint(e.process_frame(case['frame'])));e.close()
    assert hashes[0]==hashes[1]
    lc.save(OUT/'FINAL_PACKAGE_SMOKE.json',dict(public_default='native_batch_spatial',case=case,bitwise=True,hash=hashes[0],
        native_sha256=lc.sha(Path(startup().__file__)),python_files={p.relative_to(ROOT).as_posix():lc.sha(p) for p in (ROOT/'MVP/performance_final2').rglob('*.py')}))
    print('FINAL_PACKAGE_SMOKE_PASSED',flush=True)

if __name__=='__main__':main()
