"""Resume verified development phases; never freezes or reads the benchmark."""
from long_common import *
import subprocess

def main():
    commands=[('infer.py',['a','--workers','8']),('assess_long.py',['phase_a_screen','--workers','8']),('select_long.py',['a']),
      ('infer.py',['b','--workers','8']),('assess_long.py',['phase_b_full_dev','--workers','8']),('select_long.py',['b']),
      ('infer.py',['d','--workers','8']),('assess_long.py',['phase_d_combinations','--workers','8']),('ablation_detail.py',[])]
    for j,(script,args) in enumerate(commands):
        logfile=OUT/'audit'/f'corrected_{j:02d}_{script[:-3]}.log'
        print('BEGIN',script,args,'log',logfile,flush=True)
        with logfile.open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-B',str(STAGE/'src'/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('DONE',script,args,flush=True)
    print('DEVELOPMENT_COMPLETE_REQUIRES_RESEARCH_SELECTION',flush=True)

if __name__=='__main__':main()
