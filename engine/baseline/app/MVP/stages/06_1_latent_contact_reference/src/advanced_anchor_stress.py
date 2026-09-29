"""Synthetic anchor errors with unchanged real raw points, including joint EM.

Uses an in-memory profile-cache replacement solely for this isolated stress.
Frozen production source files and baseline inputs are never written.
"""
from lc_common import *
import latent_inference as li
from curve_geometry import *
from concurrent.futures import ProcessPoolExecutor

def uncached(run,i,a,p,pr,cfg):
    F=plan_frame(a['initial_basis'],a['world_up_proxy']);s=grid(a['anchor_knots'][-1],2);C=interpolation(a['anchor_knots'],a['anchor_xyz'],s,F,'pchip',.25);B=curve_frames(s,C,a['initial_basis']);ps,n=assign_stations(pr['xyz'],a['anchor_knots'],a['anchor_xyz'],pr['source_frame'],p['s_from_seed']);ss,aa,cv,profiles,timing=li.fit_profiles(s,C,B,ps,pr,p,p['seed']['side'],cfg);return dict(s=ss,anchors=aa,covariance=cv,profile_s=s,initial_C=C,initial_B=B,point_station=ps),profiles,timing

def one(r):
    run,i=r;m,a,p,pr=li.read_causal(run,i);li.profile_cache=uncached;configs=[c for c in load(OUT/'protocol.json')['candidates'] if c['name'] in ('T1_LATENT_PHYSICAL','C7_JOINT_EM')];F=plan_frame(a['initial_basis'],a['world_up_proxy']);ak=a['anchor_knots'];A=a['anchor_xyz'];mid=len(ak)//2;experiments=[];rows=[]
    for axis in (1,2):
        for cm in (2,5,10,20,30):
            AA=A.copy();AA[mid]+=F[:,axis]*cm/100;experiments.append((('lateral' if axis==1 else 'vertical')+'_anchor',cm,ak,AA))
    for shift in (-2,-1,-.5,.5,1,2):
        ks=ak.copy();ks[mid]+=shift
        if np.all(np.diff(ks)>1e-5):experiments.append(('station_shift',shift,ks,A))
    for sep in (.05,.1,.2):
        for offset in (.05,.1,.2):experiments.append(('duplicate_'+str(sep),offset,np.insert(ak,mid+1,ak[mid]+sep),np.insert(A,mid+1,A[mid]+F[:,0]*sep+F[:,2]*offset,axis=0)))
    for cfg in configs:
        _,base,_=li.infer_arrays(m,a,p,pr,cfg,run,i)
        for kind,value,ks,AA in experiments:
            aa=dict(a,anchor_knots=ks,anchor_xyz=AA);start=time.perf_counter();pred,curve,info=li.infer_arrays(m,aa,p,pr,cfg,run,i);delta=np.linalg.norm(curve['C']-base['C'],axis=1);rows.append(dict(run=run,frame=i,method=cfg['name'],experiment=kind,perturbation=value,max_curve_shift=float(delta.max()),p95_curve_shift=float(np.percentile(delta,95)),**info['physics'],elapsed_ms=(time.perf_counter()-start)*1000,raw_points_unchanged=True,GT_used=False))
    return rows

def main():
    with (OUT/'stress/bad_anchor_stress.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
    keys=list(dict.fromkeys((r['run'],int(r['frame'])) for r in rr));rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for j,rr in enumerate(pool.map(one,keys),1):rows+=rr;print('ADVANCED STRESS',j,len(keys),flush=True)
    write_csv(OUT/'stress/advanced_bad_anchor_stress.csv',rows);save(OUT/'stress/ADVANCED_COMPLETE.json',dict(time_ns=time.time_ns(),rows=len(rows),cases=len(keys),methods=['T1_LATENT_PHYSICAL','C7_JOINT_EM'],GT_used=False,production_source_unchanged=True));print('ADVANCED COMPLETE',len(rows))

if __name__=='__main__':main()
