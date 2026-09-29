from . import ROOT,OUT
from .engine import Engine
from MVP.final_pipeline.verify_raw import normalized
from MVP.final_pipeline.benchmark import COHORT
import long_common as lc
import numpy as np
import hashlib,json,argparse,time


def fingerprint(result):
    value={k:normalized(result[k]) for k in ('step1','step2','c4','seed','prediction','c4_smooth','provenance')}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',default='python');ap.add_argument('--frames',type=int,default=8);a=ap.parse_args()
    rows=[]
    for run,start,category in COHORT:
        e=Engine(a.backend);e.start_run(lc.dataset()/run)
        for i in range(start,start+a.frames):
            result=e.process_frame(i);h=fingerprint(result)
            row=dict(run=run,frame=i,backend=a.backend,category=category,direction=e.direction_info,
                available=result['prediction'] is not None,hash=h,**result['summary']['timing'])
            rows.append(row)
            print('ABLATION',a.backend,run,i,round(row['T_TOTAL'],3),flush=True)
    lc.save(OUT/'ablation'/f'{a.backend}.json',rows)
    if a.backend!='python':
        ref=lc.load(OUT/'ablation/python.json')
        assert [(r['run'],r['frame']) for r in rows]==[(r['run'],r['frame']) for r in ref], 'Ablation cohorts differ'
        differences=[dict(run=x['run'],frame=x['frame']) for x,y in zip(rows,ref) if x['hash']!=y['hash']]
        lc.save(OUT/'ablation'/f'{a.backend}_equivalence.json',dict(cases=len(rows),bitwise_exact=not differences,differences=differences))
        print('EQUIVALENCE',not differences, differences,flush=True)


if __name__=='__main__':main()
