from . import OUT
from .engine import Engine
from MVP.performance_final.ablation import fingerprint
from MVP.final_pipeline.memory import rss
import long_common as lc
import argparse,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--limit',type=int,default=32);a=ap.parse_args()
    cohort=lc.load(OUT/'benchmark_cohort.json')['clean'][:a.limit];reference=lc.load(OUT/'baseline/python.json');rows=[];run=None;e=None
    for case in cohort:
        if case['run']!=run:
            if e:e.close()
            run=case['run'];e=Engine(a.backend,a.workers);e.start_run(lc.dataset()/run)
        result=e.process_frame(case['frame']);hash=fingerprint(result);ref=next(r for r in reference if r['run']==run and r['frame']==case['frame'])
        row=dict(**case,backend=a.backend,workers=a.workers,bitwise=hash==ref['hash'],hash=hash,**result['summary']['timing'],
            peak_RSS_bytes=rss()['peak_rss_bytes'],stats=dict(e.trackers[-1].stats) if e.trackers else {})
        rows.append(row);print('ABLATION',a.backend,a.workers,run,case['frame'],round(row['T_TOTAL'],3),row['bitwise'],flush=True)
    if e:e.close()
    name=a.backend+(f'_{a.workers}' if a.workers!=1 else '')
    lc.save(OUT/'ablations'/f'{name}.json',rows)
    if not all(r['bitwise'] for r in rows):raise RuntimeError('Non-equivalent variant: '+name)

if __name__=='__main__':main()
