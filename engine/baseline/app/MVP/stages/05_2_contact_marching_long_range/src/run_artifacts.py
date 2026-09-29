"""Finish reports only after the locked benchmark and all analyses are complete."""
from long_common import *
import subprocess

def launch(name,args,log):
    stream=(OUT/'audit'/log).open('w',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B',str(STAGE/'src'/name)]+args,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    return p,stream

def main():
    while not (OUT/'las/examples.json').exists():time.sleep(4)
    time.sleep(1)
    jobs=[launch('render_long.py',['--workers','4'],'render_final.log'),launch('export_long_las.py',['--workers','3'],'export_final.log')]
    print('ARTIFACTS RENDER AND LAS STARTED',flush=True)
    for p,stream in jobs:
        code=p.wait();stream.close();assert code==0,('artifact child exit',p.pid,code)
    for name,log in [('render_observability.py','render_observability.log'),('supplement_long.py','supplement_long.log'),('sampling_audit.py','sampling_audit.log'),('benchmark_long.py','runtime_final.log'),('report_long.py','report_final.log')]:
        print('BEGIN',name,flush=True);p,stream=launch(name,[],log);code=p.wait();stream.close();assert code==0,(name,code);print('DONE',name,flush=True)
    print('ARTIFACTS_READY_FOR_MANUAL_QA_AND_FINAL_VERIFICATION',flush=True)

if __name__=='__main__':main()
