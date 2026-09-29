"""Confirm compact pilot across the original 108 STEP5 development starts."""
from fusion_common import *
from fusion import CausalSource
if __name__=='__main__':
    cohort=load(OUT/'audit/cohort.json');cohort['pilot_development']=cohort['development'];cohort['development']=[r for r in load(BASE_OUT/'audit/cohort.json') if r['split']=='development'];save(OUT/'audit/cohort.json',cohort)
    p=load(OUT/'protocol.json');p['development_starts']=len(cohort['development']);p['self_train_mask']='Deferred by explicit user instruction: currently not important. No invented sensor/train extrinsics.';save(OUT/'protocol.json',p)
    rows=[]
    for r in cohort['development']+cohort['heldout']:
        s=seed_from_cache(r['run'],r['frame'])
        if s['status']!='AVAILABLE':continue
        src=CausalSource(r['run'],r['frame'])
        for d in (.5,1.,2.,4.):
            ids,why=src.history(dict(distance=d));rows.append(dict(run=r['run'],frame=r['frame'],distance=d,frames=len(ids),reason=why))
    csv_write(OUT/'audit/history_available.csv',rows)
    print('AVAILABLE HISTORY',{str(d):percentiles([r['frames'] for r in rows if r['distance']==d]) for d in (.5,1.,2.,4.)},flush=True)
