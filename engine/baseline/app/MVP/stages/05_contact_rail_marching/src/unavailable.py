from common import *
from evaluate_study import one
if __name__=='__main__':
    n=0
    for p in (OUT/'heldout').glob('*/prediction.json'):
        r=load(p)
        if r['seed']['status']!='AVAILABLE':one((r['run'],r['start_frame'],'heldout'));n+=1
    print('UNAVAILABLE census complete',n,flush=True)
