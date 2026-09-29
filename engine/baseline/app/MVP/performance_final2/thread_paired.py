from . import OUT
from .engine import Engine
from MVP.performance_final.ablation import fingerprint
import long_common as lc
import numpy as np

def main():
    rows=[];cohort=lc.load(OUT/'benchmark_cohort.json')['clean'];cases=[c for j,c in enumerate(cohort) if j%8<2]
    for j,case in enumerate(cases):
        order=[1,2,4,8,16];order=order[j%5:]+order[:j%5];hashes=[]
        for workers in order:
            e=Engine('batch',workers);e.start_run(lc.dataset()/case['run']);r=e.process_frame(case['frame']);e.close();hashes.append(fingerprint(r))
            rows.append(dict(**case,workers=workers,full_ms=r['summary']['timing']['T_TOTAL']*1000,C4_ms=r['summary']['timing']['T_C4_MARCHING']*1000,bitwise=True))
        assert len(set(hashes))==1
        print('THREAD_PAIRED',j+1,len(cases),flush=True)
    lc.csv_write(OUT/'thread_paired.csv',rows)

if __name__=='__main__':main()
