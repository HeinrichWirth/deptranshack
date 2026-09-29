"""Run timing-sensitive jobs sequentially, after Docker timing completion."""
from . import OUT,ROOT
import subprocess,sys,time

def main():
    jobs=[('solver_audit',[]),('spatial_micro',[]),('gil_benchmark',[]),('thread_paired',[]),('offline_throughput',[]),
          ('audit',[]),('quality_sanity',[]),('async_simulation',['--backend','native_spatial']),
          ('async_simulation',['--backend','batch']),('async_simulation',['--backend','batch','--live'])]
    for name,args in jobs:
        path=OUT/'logs'/('remaining_'+name+('_'+'_'.join(args).replace('--','') if args else '')+'.log');path.parent.mkdir(exist_ok=True)
        print('START',name,args,flush=True)
        with path.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-B','-m','MVP.performance_final2.'+name,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,args,flush=True)

if __name__=='__main__':main()
