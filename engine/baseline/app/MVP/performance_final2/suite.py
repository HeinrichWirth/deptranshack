from . import OUT,ROOT
import long_common as lc
import subprocess,sys,time

def main():
    jobs=[('h1',1),('h2',1),('h3',1),('residual',1),('fd',1),('batch',1),('threads',2),('threads',4),('threads',8),('threads',16),('grid',1),('soa',1),('optimized',1)]
    rows=[];start=time.time_ns();(OUT/'logs').mkdir(exist_ok=True)
    for backend,workers in jobs:
        name=backend+(f'_{workers}' if workers!=1 else '')
        with (OUT/'logs'/f'{name}.log').open('w',encoding='utf-8') as stream:
            p=subprocess.run([sys.executable,'-B','-m','MVP.performance_final2.ablation','--backend',backend,'--workers',str(workers)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        rows.append(dict(backend=backend,workers=workers,returncode=p.returncode))
        print('SUITE',name,p.returncode,flush=True)
    lc.save(OUT/'suite.json',dict(start_ns=start,end_ns=time.time_ns(),rows=rows,serial=True))

if __name__=='__main__':main()
