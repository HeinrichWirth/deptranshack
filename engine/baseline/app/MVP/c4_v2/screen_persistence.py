"""Bounded development parameter screen; all starts begin at frame zero."""
from . import ROOT,OUT
from .run_sequences import sequence
import long_common as lc
from concurrent.futures import ProcessPoolExecutor
import argparse,time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control',action='store_true');ap.add_argument('--limit',type=int,default=80);a=ap.parse_args()
    runs=lc.load(ROOT/'MVP/c4_v2/plan.json')['development']
    configs=[dict(name=f'r{r}_o{o}',repair=r,overlap=o,threads=4,iterations=12,grid=0,control='combined') for r,o in [(4,8),(6,8),(12,8),(8,4),(8,12)]]
    if a.control:configs=[dict(name='persistence_only_r8_o8',repair=8,overlap=8,threads=1,iterations=12,grid=0,control='persistence_only')]
    for config in configs:
        root=OUT/'development_screen'/config['name'];results=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(sequence,[(run,'development_screen',config,a.limit) for run in runs]):results.append(result)
        lc.save(root/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=[r for x in results for r in x['rows']],frames=sum(x['frames'] for x in results),config=config))

if __name__=='__main__':main()
