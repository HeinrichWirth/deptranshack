from . import ROOT,OUT
from .run_full import RUNS
import subprocess,sys,os,time,json
from concurrent.futures import ThreadPoolExecutor,as_completed


def job(task):
    mode,backend,run=task;logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/(mode+'_'+run+'.log')
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'-B','-m','MVP.performance_final.run_full','--mode',mode,'--backend',backend,'--run',run],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    return dict(mode=mode,backend=backend,run=run,returncode=p.returncode)


def main():
    tasks=[('scheduler','spatial',r) for r in RUNS]+[('every_frame','python',r) for r in RUNS]
    start=time.time();rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'batch_full.json').write_text(json.dumps(dict(rows=rows,elapsed_seconds=time.time()-start,
        concurrent_independent_processes=3,notes='Per-process CPU measured separately; full-run wall includes competition. Clean latency ablations are serial.'),indent=2),encoding='utf-8')
    assert all(r['returncode']==0 for r in rows)


if __name__=='__main__':main()
