"""Post-lock DEVELOPMENT-ONLY check of a short-window initialization edge case.

The original W6/advance1.5 first slab ended inside the 8m seed. Correct the
diagnostic by starting from the latest 6m confirmed interval. No main code,
selected configuration or held-out prediction is changed or selected again.
"""
from common import *
from run_study import seed_geometry,save_prediction
from tracker import predict,DEFAULT
from evaluation import reference,evaluate_prediction
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(row):
    run=row['run'];i=row['frame'];xyz,seed=seed_geometry(run,i);cfg=dict(DEFAULT,window=6.,advance=1.5)
    seed=dict(seed)
    if seed['status']=='AVAILABLE':seed.update(anchor=np.asarray(seed['anchor'])+2*np.asarray(seed['basis'])[:,0],seed_end=6.)
    pred=predict(xyz,seed,ContactRailDetector().template,cfg);folder=OUT/'ablation/window_6_quarter_corrected_initialization'/key(run,i)
    save_prediction(folder,pred,run,i);pred.update(run=run,start_frame=i);ref,info=reference(run,i,'development')
    r,s,b,p=evaluate_prediction(pred,xyz,ref,info);save(folder/'evaluation.json',dict(version=2,summary=r,steps=s,bins=b,reference=info));return r

if __name__=='__main__':
    co=load(OUT/'calibration/sweep_protocol.json')['cohort'];rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,r) for r in co]):rows.append(f.result())
    csv_write(OUT/'calibration/window_initialization_diagnostic.csv',rows)
    good=[r for r in rows if r.get('evaluation_eligible')];save(OUT/'calibration/window_initialization_diagnostic.json',dict(
        timing='post research lock; development only; no selection changes',configuration=dict(window=6,advance=1.5,gate=.10,method='M2'),
        issue='Original first short window ended inside 8m seed. Diagnostic starts from latest 6m confirmed interval.',
        reach=percentiles([r['continuous_reach_5cm'] for r in good]),main_configuration_unchanged=True))
