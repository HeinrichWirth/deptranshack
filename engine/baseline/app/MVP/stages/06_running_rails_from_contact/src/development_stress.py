"""Seed/CR perturbations and seed-length diagnostics; development data only."""
from rr_common import *
from seed_input import read_input,read_seed_labels,calibrate_seed
from rail_models import predict,INPUT_KEYS
from track_geometry import rails,offsets,transport
from gt_reference import future_pair_reference
from evaluate_rails import measure
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy,argparse

def compare(base,new,s):
    rows=[]
    delta=new['pair']-base['pair'];B=base['B'];trans=delta-np.einsum('nri,ni->nr',delta,B[:,:,0])[:,:,None]*B[:,None,:,0]
    for target in (30,50,75,100):
        ids=np.flatnonzero(abs(s-target)<=2)
        if not len(ids):rows.append(dict(range=target,available=False));continue
        rows.append(dict(range=target,available=True,near_3d_cm=float(np.max(np.linalg.norm(delta[ids,0],axis=1)))*100,far_3d_cm=float(np.max(np.linalg.norm(delta[ids,1],axis=1)))*100,near_transverse_cm=float(np.max(np.linalg.norm(trans[ids,0],axis=1)))*100,far_transverse_cm=float(np.max(np.linalg.norm(trans[ids,1],axis=1)))*100))
    return rows

def one(task):
    run,i,name=task;folder=OUT/'phase_b'/name/key(run,i);rec=load(folder/'prediction.json');meta,a=read_input(OUT/rec['input_folder'])
    if meta['status']!='AVAILABLE':return [],[],[]
    cfg=load(OUT/'configs.json')[name];pr=load(OUT/'models/global_prior.json');prior=pr['leave_run_out'][run];data={k:a[k].copy() for k in INPUT_KEYS};base=predict(meta['seed'],data,meta['profiles'],meta['q'],cfg,prior);seedrows=[];crrows=[];lengthrows=[]
    for rail in (0,1):
        for axis in (1,2):
            for magnitude in (.01,.02,.05):
                for sign in (-1,1):
                    seed=copy.deepcopy(meta['seed']);pp=rails(np.zeros(3),np.eye(3),np.array(seed['x0']),meta['q']);pp[rail,axis]+=sign*magnitude;seed['x0']=offsets(np.zeros(3),np.eye(3),pp,meta['q']).tolist()
                    out=predict(seed,data,meta['profiles'],meta['q'],cfg,prior)
                    for row in compare(base,out,a['s']):seedrows.append(dict(run=run,frame=i,method=name,perturbation='rail_anchor',rail=rail,axis=axis,magnitude=sign*magnitude,**row))
    for deg in (.1,.25,.5,1.):
        for sign in (-1,1):
            seed=copy.deepcopy(meta['seed']);seed['x0'][0]+=np.deg2rad(sign*deg);out=predict(seed,data,meta['profiles'],meta['q'],cfg,prior)
            for row in compare(base,out,a['s']):seedrows.append(dict(run=run,frame=i,method=name,perturbation='initial_roll',magnitude=sign*deg,**row))
    for axis in (1,2):
        for magnitude in (.01,.02,.05):
            for sign in (-1,1):
                changed={k:v.copy() for k,v in data.items()};changed['C']+=sign*magnitude*changed['B'][:,:,axis];out=predict(meta['seed'],changed,meta['profiles'],meta['q'],cfg,prior)
                for row in compare(base,out,a['s']):crrows.append(dict(run=run,frame=i,method=name,perturbation='CR_position',axis=axis,magnitude=sign*magnitude,**row))
        for deg in (.1,.25,.5,1.):
            for sign in (-1,1):
                changed={k:v.copy() for k,v in data.items()};rv=np.zeros(3);rv[axis]=np.deg2rad(sign*deg);changed['B']=changed['B']@Rotation.from_rotvec(rv).as_matrix();out=predict(meta['seed'],changed,meta['profiles'],meta['q'],cfg,prior)
                for row in compare(base,out,a['s']):crrows.append(dict(run=run,frame=i,method=name,perturbation='CR_tangent',axis=axis,magnitude=sign*deg,**row))
    ref=future_pair_reference(run,i,folder)
    for length in (4,8,12,16):
        xyz,labels,audit=read_seed_labels(run,i,a['initial_basis'],length);model=dict(s=a['s'],C=a['C'],B=a['B'],knots=a['anchor_knots'],anchors=a['anchor_xyz']);seed=calibrate_seed(xyz,labels,model,a['initial_basis'],meta['q'],length)
        if seed['status']!='AVAILABLE':lengthrows.append(dict(run=run,frame=i,method=name,requested_seed_length=length,status=seed['status'],primary=False));continue
        out=predict(seed,data,meta['profiles'],meta['q'],cfg,prior);e=measure(out,ref,a['B'])
        for lo,hi in ((30,50),(50,75),(75,100)):
            valid=e['valid']&(e['range']>=lo)&(e['range']<hi)
            lengthrows.append(dict(run=run,frame=i,method=name,requested_seed_length=length,observed_seed_end=max(o['s'] for o in seed['observations']),seed_sections=seed['sections'],status='AVAILABLE',primary=length==8,lo=lo,hi=hi,n=int(valid.sum()),near_p95=stats(e['error'][valid,0]).get('p95'),far_p95=stats(e['error'][valid,1]).get('p95'),alpha_sigma=np.sqrt(np.array(seed['covariance'])[0,0])))
    return seedrows,crrows,lengthrows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args();proposal=load(OUT/'models/proposed_final.json');name=proposal['top3'][0];rows=load(OUT/'phase_b/per_start.json');selected=[]
    for run in load(OUT/'protocol.json')['split']['development']:
        rr=[r for r in rows if r['run']==run and r['method']==name and r['status']=='AVAILABLE']
        # Two long starts per run: enough support to measure sensitivity where available.
        selected.extend((run,r['frame'],name) for r in sorted(rr,key=lambda r:-(r.get('max_predicted_range') or 0))[:2])
    seed=[];cr=[];lengths=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j,(x,y,z) in enumerate(pool.map(one,selected),1):seed+=x;cr+=y;lengths+=z;print('STRESS',j,len(selected),flush=True)
    write_csv(OUT/'seed_stress.csv',seed);write_csv(OUT/'cr_stress.csv',cr);write_csv(OUT/'seed_calibration/length_stress.csv',lengths)
    save(OUT/'stress/COMPLETE.json',dict(time_ns=time.time_ns(),method=name,starts=selected,primary_seed_length=8,perturbations_are_synthetic=True,C4_modified=False,missing_horizon='Not measured, never extrapolated to claim 100m.'))

if __name__=='__main__':main()
