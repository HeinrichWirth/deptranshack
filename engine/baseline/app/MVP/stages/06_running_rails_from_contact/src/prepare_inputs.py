from rr_common import *
from seed_input import save_input,read_input
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def one(r):
    path=save_input(r['run'],r['frame']);m,_=read_input(path)
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args();check_dependencies()
    rows=load(OUT/'protocol.json')['cohort']['development'];done=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        ff=[pool.submit(one,r) for r in rows]
        for j,f in enumerate(as_completed(ff),1):
            m=f.result();done.append(m)
            if j%10==0 or j==len(ff):print('PREPARE SEED',j,len(ff),m['status'],flush=True)
    good=[m for m in done if m['status']=='AVAILABLE'];runs=sorted(set(m['run'] for m in good))
    def prior(exclude=None):
        x=np.array([np.median([m['seed']['x0'][1:] for m in good if m['run']==run],axis=0) for run in runs if run!=exclude])
        return dict(mean=np.median(x,axis=0),variance=np.var(x,axis=0)+.015**2,runs=[r for r in runs if r!=exclude])
    save(OUT/'models/global_prior.json',dict(default=prior(),leave_run_out={r:prior(r) for r in runs},source='Seed only; equal weight per development run; own run excluded for development inference.'))
    save(OUT/'seed_calibration/development_summary.json',dict(total=len(done),available=len(good),by_run={r:sum(m['run']==r for m in good) for r in runs}))
    write_csv(OUT/'seed_geometry.csv',[dict(run=m['run'],frame=m['i'],sections=m['seed']['sections'],alpha=m['seed']['x0'][0],d=m['seed']['x0'][1],h=m['seed']['x0'][2],g=m['seed']['x0'][3],h_far=m['seed']['x0'][4],alpha_sigma=np.sqrt(np.array(m['seed']['covariance'])[0,0])) for m in good])

if __name__=='__main__':main()
