"""One frozen reused-benchmark evaluation; stops on the first failed step."""
from rr_common import *
from run_inference import check_freeze
import subprocess

def main():
    check_freeze()
    if (OUT/'phase_e/FINAL_PIPELINE_COMPLETE.json').exists():raise RuntimeError('Benchmark already completed; do not rerun or retune.')
    jobs=[('run_inference',['e','--workers','6']),('evaluate_rails',['phase_e','--workers','6']),('oracle_study',['phase_e','--workers','6']),('extended_oracles',['phase_e']),('geometry_audit',['phase_e','--workers','6']),('runtime_benchmark',[])]
    for name,args in jobs:
        check_freeze();print('BEGIN',name,*args,flush=True)
        logfile=OUT/'audit'/('final_'+name+'.log')
        with logfile.open('w',encoding='utf-8') as log:done=subprocess.run([sys.executable,'-B',str(STAGE/'src'/(name+'.py')),*args],stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(str(logfile))
        print('DONE',name,*args,flush=True)
    check_freeze();save(OUT/'phase_e/FINAL_PIPELINE_COMPLETE.json',dict(time_ns=time.time_ns(),freeze_sha256=sha(OUT/'research_freeze.json'),configuration_modified_after_benchmark=False))

if __name__=='__main__':main()
