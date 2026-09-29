"""Ordered development pipeline. Does NOT freeze or run the reused benchmark."""
from rr_common import *
import subprocess

def main():
    jobs=[('prepare_inputs',['--workers','6']),('run_inference',['a','--workers','6']),('evaluate_rails',['phase_a','--workers','6']),('roll_diagnostics',['phase_a']),('oracle_study',['phase_a','--workers','6']),('select_methods',['a']),('run_inference',['b','--workers','6']),('evaluate_rails',['phase_b','--workers','6']),('roll_diagnostics',['phase_b']),('oracle_study',['phase_b','--workers','6']),('select_methods',['b'])]
    for name,args in jobs:
        print('BEGIN',name,*args,flush=True)
        logfile=OUT/'audit'/('development_'+name+'_'+('_'.join(args) or 'main')+'.log')
        with logfile.open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,'-B',str(STAGE/'src'/(name+'.py')),*args],stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(str(logfile))
        print('DONE',name,*args,flush=True)
    print('DEVELOPMENT COMPLETE; human-readable review and remaining diagnostic checks precede research freeze.',flush=True)

if __name__=='__main__':main()
