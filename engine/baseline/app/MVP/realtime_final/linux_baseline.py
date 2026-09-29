"""First action: unchanged Linux reference vs unchanged native, all 240 starts."""
from . import ROOT,OUT
from MVP.full_native_final.qualify import engine,process
import long_common as lc
from pathlib import Path
import time,json,os,sys,platform

def main():
    data=Path(os.environ.get('REALTIME_DATA','/data'));root=OUT/'linux_baseline';root.mkdir(parents=True,exist_ok=True)
    selection=lc.load(OUT/'ORIGINAL_SELECTION_LOCK.json');rows=[]
    for run in selection['benchmark']:
        es={b:engine(b) for b in ('reference','full_native')}
        for e in es.values():e.start_run(data/run)
        for case in (c for c in selection['cohort'] if c['run']==run):
            folder=root/'benchmark'/run/f'{case["frame"]:06d}'
            for b,e in es.items():rows.append(process(e,b,case,folder))
            lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={p.name:lc.sha(p) for p in folder.iterdir() if p.suffix in ('.npz','.json') and p.name!='PREDICTION_COMPLETE.json'},uses_annotation_labels=False,uses_future_clouds=False,persistent_c4=False))
            print('LINUX_BASELINE',run,case['frame'],flush=True)
        for e in es.values():e.close()
    lc.save(root/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=rows,platform=platform.platform(),python=sys.version))
    lc.save(root/'FULL_NATIVE_SELECTION_LOCK.json',selection)
    print('LINUX_ALL_240_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
