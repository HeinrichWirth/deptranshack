from common import *
from evaluation import load_prediction,reference
from analyze import observations
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(folder):
    pred=load_prediction(folder)
    if pred['seed']['status']!='AVAILABLE' or (folder/'observability.json').exists():return
    run=pred['run'];i=pred['start_frame'];xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,_=reference(run,i,'heldout')
    observations(run,i,pred,ref,xyz)

if __name__=='__main__':
    tasks=sorted([p.parent for p in (OUT/'heldout').glob('*/prediction.json')],reverse=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(one,p) for p in tasks]
        for j,f in enumerate(as_completed(fs)):
            f.result()
            if j%100==0:print('OBSERVABILITY',j+1,len(fs),flush=True)
