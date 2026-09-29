"""Evaluation-only upper bounds. Never imported by production run_study/tracker."""
from common import *
from evaluation import require_prediction,load_prediction,reference,evaluate_prediction,make_curve,voxel
from tracker import _engine,TemplateTracker
from run_study import seed_geometry,save_prediction
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def oracle_seed(run,i,xyz,production,length,template,matched=False):
    if production.get('basis') is None:return production
    B=np.array(production['basis']);row=frames(run)[i]
    # Offline oracle-only label access, after the production completion guard.
    cloud=laspy.read(dataset()/run/row['file']);uvw=xyz@B
    ids=np.flatnonzero((np.asarray(cloud.classification)==2)&(uvw[:,0]>=0)&(uvw[:,0]<=length))
    if len(ids)<3:return dict(production,status='START_UNAVAILABLE',reason='ORACLE_SEED_LABELS_UNAVAILABLE',indices=np.empty(0,dtype=np.int64))
    side=1 if np.median(uvw[ids,1])>=0 else -1;ids=ids[np.sign(uvw[ids,1])==side]
    cfg=dict(min_support=3,support_tolerance=.020)
    tt=TemplateTracker(template,side,cfg);center=np.median(uvw[ids,1:],axis=0);cc,_=tt.fit(uvw[ids,1:],center,.12)
    if not cc:return dict(production,status='START_UNAVAILABLE',reason='ORACLE_SEED_FIT_UNAVAILABLE',indices=np.empty(0,dtype=np.int64))
    if matched:
        anchor=cc[0]['anchor'];q=uvw[ids,1:]-anchor
        keep=np.all((q>=tt.lo)&(q<=tt.hi),axis=1)&(tt.distances(uvw[ids,1:],anchor)<=.020)
        ids=ids[keep]
        if len(np.unique(xyz[ids],axis=0))<3:return dict(production,status='START_UNAVAILABLE',reason='ORACLE_TEMPLATE_SUPPORT_UNAVAILABLE',indices=np.empty(0,dtype=np.int64))
    # A longer oracle seed is a longer CONFIRMED history. Continue from its final
    # 8m window; otherwise W=8 placed at u=4 would end before a 15m seed ends.
    start=max(0.,float(length)-8.)
    return dict(status='AVAILABLE',reason='',basis=B,anchor=np.array([start,*cc[0]['anchor']])@B.T,side=side,
                indices=ids,confidence=1.,seed_end=8.,seed_history_m=float(length),seed_origin_u=start,seed_ms=0.,diagnostic_only=True,
                annotation_seed_mode='template_consistent_class2' if matched else 'all_class2_on_selected_side')

def one(task):
    run,i,group,cfg,*flags=task;matched=bool(flags and flags[0]);folder=require_prediction(group,run,i);pred=load_prediction(folder)
    xyz,seed=seed_geometry(run,i);ref,info=reference(run,i,group);template=ContactRailDetector().template
    # The oracle can use current annotated seed geometry where future T+2 has
    # already moved past it. This stays entirely inside the diagnostic module.
    Gcurve=ref['curve'];Gbasis=ref['curve_basis']
    if seed.get('basis') is not None:
        B0=np.asarray(seed['basis']);near=ref['current'][(ref['current']@B0[:,0]>=0)&(ref['current']@B0[:,0]<=8)]
        mm=frames(run);P=np.array(mm[i]['lidar_pose_in_folder']);positions=np.array([m['lidar_pose_in_folder'] for m in mm[i:info['future_end']+1]])[:,:3,3]
        positions=(positions-P[:3,3])@P[:3,:3]
        Gcurve,Gbasis,_=make_curve(voxel(np.r_[near,ref['future']],.005),positions,B0,seed.get('side',1))
    def orientation(C,B,tail):
        if len(Gcurve)<3:raise ValueError('GT_UNAVAILABLE_AHEAD')
        d=np.linalg.norm(Gcurve-C,axis=1);j=d.argmin()
        if d[j]>2:raise ValueError('GT_UNAVAILABLE_AHEAD')
        G=Gbasis[j].copy()
        if G[:,0]@B[:,0]<0:G[:,0]*=-1;G[:,1]*=-1
        return G
    variants=[(f'ORACLE_SEED_{l}',l,False) for l in (8,10,12,15)]+[('ORACLE_FRAME',None,True),('ORACLE_SEED_FRAME',8,True)]
    if matched:variants=[('ORACLE_SEED_TEMPLATE8',8,False),('ORACLE_SEED_TEMPLATE8_FRAME',8,True)]
    rows=[]
    for name,length,frame in variants:
        out=OUT/'oracles'/name/key(run,i)
        if (out/'evaluation.json').exists():
            previous=load(out/'evaluation.json')
            if previous.get('oracle_version')==3 or length in (None,8):
                rows.append(dict(oracle=name,**previous['summary']));continue
        if length is None:s=seed
        else:s=oracle_seed(run,i,xyz,seed,length,template,matched=matched)
        result=_engine(xyz,s,template,cfg,orientation if frame else None)
        if s.get('seed_origin_u') and 's_from_seed' in result:result['s_from_seed']+=s['seed_origin_u']
        save_prediction(out,result,run,i);result.update(run=run,start_frame=i)
        row,sr,bins,p=evaluate_prediction(result,xyz,ref,info)
        save(out/'evaluation.json',dict(version=2,oracle_version=3,summary=row,steps=sr,bins=bins,reference=info,diagnostic_only=True))
        rows.append(dict(oracle=name,**row))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',choices=['development','heldout']);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--template-seed',action='store_true');args=ap.parse_args()
    cfg=load(OUT/'config.json') if (OUT/'config.json').exists() else load(OUT/'calibration/selected.json')['config']
    if args.group=='development':rr=load(OUT/'calibration/sweep_protocol.json')['cohort']
    else:rr=[r for r in load(OUT/'audit/cohort.json') if r['split']=='validation']
    rr=[r for r in rr if load(require_prediction(args.group,r['run'],r['frame'])/'prediction.json')['seed']['status']=='AVAILABLE']
    t=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(one,(r['run'],r['frame'],args.group,cfg,args.template_seed)) for r in rr]
        for j,f in enumerate(as_completed(fs)):
            rows.extend(f.result())
            if j%20==0 or j==len(fs)-1:print('ORACLE',j+1,len(fs),round(time.perf_counter()-t,1),flush=True)
    csv_write(OUT/f'oracle_comparison_{args.group}{"_template_seed" if args.template_seed else ""}.csv',rows)
if __name__=='__main__':main()
