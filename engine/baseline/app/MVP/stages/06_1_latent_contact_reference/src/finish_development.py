"""Sequential development gates; does not automatically open benchmark."""
from lc_common import *
import subprocess

def main():
    commands=[('tests','../tests/test_latent.py',[]),('phase_a','run_phase.py',['phase_a','--workers','6']),('select_a','select_models.py',['phase_a']),('phase_b','run_phase.py',['phase_b','--workers','6']),('select_b','select_models.py',['phase_b']),('freeze','freeze_finalists.py',[]),('phase_d','run_phase.py',['phase_d','--workers','6'])]
    for log,script,args in commands:
        with (OUT/'audit'/(log+'.log')).open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-B',str(STAGE/'src'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('GATE COMPLETE',log,flush=True)

if __name__=='__main__':main()
