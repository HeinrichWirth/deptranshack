"""Controlled research stresses on actual causal observations; separate outputs.

Perturbed data never overwrites source data or enters main benchmark selection.
Full-vs-partial bias is repeatability, not physical absolute accuracy.
"""
from lc_common import *
from latent_inference import read_causal,profile_cache
from curve_geometry import *
from profile_likelihood import fit_anchor
from seed_input import canonical
from concurrent.futures import ProcessPoolExecutor

def partial_task(task):
    run,i,cfg=task;m,a,p,pr=read_causal(run,i)
    if m['status']!='AVAILABLE':return [],[]
    pa,profiles,timing=profile_cache(run,i,a,p,pr,cfg['profile']);result=[];multi=[];eligible=[j for j,v in enumerate(profiles) if v['n']>=30 and v['sources']>=2]
    if not eligible:return result,multi
    chosen=[eligible[j] for j in np.unique(np.linspace(0,len(eligible)-1,min(3,len(eligible)),dtype=int))]
    rng=np.random.default_rng(6101);template=canonical()*np.array([m['q'],1])
    for j in chosen:
        profile=profiles[j];ids=np.array(profile['point_ids'],int);s=profile['station'];center=pa['initial_C'][j];B=pa['initial_B'][j];centers=np.column_stack([np.interp(pa['point_station'][ids],pa['profile_s'],pa['initial_C'][:,k]) for k in range(3)]);P=(pr['xyz'][ids]-centers)@B[:,1:];src=pr['source_frame'][ids];rr=pr['sensor_distance_at_source'][ids];ring=pr['ring'][ids];state=p['point_state'][ids]
        def fit(mask,points=None,sources=None):
            return fit_anchor(P[mask] if points is None else points,src[mask] if sources is None else sources,template,range_m=rr[mask],ring=ring[mask],state=state[mask],config=dict(cfg['profile'],starts=4))
        full=fit(np.arange(len(P)));median=np.median(P,axis=0);lo=np.quantile(P,.25,axis=0);hi=np.quantile(P,.75,axis=0);corner=np.argsort(np.linalg.norm((P-[lo[0],hi[1]])/np.maximum(np.ptp(P,axis=0),.001),axis=1))[:max(2,len(P)//8)]
        masks={'FULL':np.arange(len(P)),'TOP_25':np.flatnonzero(P[:,1]>=hi[1]),'BOTTOM_25':np.flatnonzero(P[:,1]<=lo[1]),'LEFT_VERTICAL':np.flatnonzero(P[:,0]<=lo[0]),'RIGHT_FACE':np.flatnonzero(P[:,0]>=hi[0]),'CORNER':corner}
        for frac in (.5,.25,.1,.05):masks['RANDOM_'+str(frac)]=np.sort(rng.choice(len(P),max(1,int(len(P)*frac)),replace=False))
        for count in (1,2,3):masks[str(count)+'_POINT']=np.sort(rng.choice(len(P),count,replace=False))
        for name,idx in masks.items():
            f=full if name=='FULL' else fit(idx);bias=f['anchor']-full['anchor'];naive=np.median(P[idx],axis=0)-median;cov=f['covariance'];eig,U=np.linalg.eigh(cov)
            result.append(dict(run=run,frame=i,station=s,pattern=name,n=len(idx),sources=f['sources'],bias_lateral=bias[0],bias_vertical=bias[1],bias_norm=float(np.linalg.norm(bias)),median_bias_lateral=naive[0],median_bias_vertical=naive[1],median_bias_norm=float(np.linalg.norm(naive)),sigma_v=np.sqrt(cov[0,0]),sigma_w=np.sqrt(cov[1,1]),weak_axis_v=U[0,-1],weak_axis_w=U[1,-1],mode_count=len(f['modes']),fit_state=f['state'],visibility=f['visibility'],full_anchor=full['anchor'],partial_anchor=f['anchor'],ms=f['ms'],reference='same full observed profile, not physical GT'))
        # A/B/C are genuinely different source frames when available. Keep only
        # their stated fragments. If insufficient surfaces, explicitly report it.
        unique=np.unique(src);pick=unique[np.linspace(0,len(unique)-1,min(3,len(unique)),dtype=int)];fragments=[]
        for k,source in enumerate(pick):
            idx=np.flatnonzero(src==source)
            if k==0:idx=idx[P[idx,1]>=np.quantile(P[idx,1],.75)]
            elif k==1:idx=idx[P[idx,0]<=np.quantile(P[idx,0],.25)]
            else:idx=idx[np.argsort(np.linalg.norm(P[idx]-[lo[0],hi[1]],axis=1))[:2]]
            fragments.append(idx);f=fit(idx);multi.append(dict(run=run,frame=i,station=s,experiment='COMPLEMENTARY_SOURCE_'+str(k),n=len(idx),sources=f['sources'],anchor_shift=float(np.linalg.norm(f['anchor']-full['anchor'])),sigma_v=np.sqrt(f['covariance'][0,0]),sigma_w=np.sqrt(f['covariance'][1,1]),available=len(pick)==3,source=int(source)))
        idx=np.concatenate(fragments);f=fit(idx);multi.append(dict(run=run,frame=i,station=s,experiment='COMPLEMENTARY_JOINT',n=len(idx),sources=f['sources'],anchor_shift=float(np.linalg.norm(f['anchor']-full['anchor'])),sigma_v=np.sqrt(f['covariance'][0,0]),sigma_w=np.sqrt(f['covariance'][1,1]),available=len(pick)==3))
        for age in range(1,min(6,i+1)):
            idx=np.flatnonzero(src!=i-age)
            if len(idx)==len(P):continue
            f=fit(idx);multi.append(dict(run=run,frame=i,station=s,experiment='DROP_T_MINUS_'+str(age),n=len(idx),sources=f['sources'],anchor_shift=float(np.linalg.norm(f['anchor']-full['anchor'])),sigma_v=np.sqrt(f['covariance'][0,0]),sigma_w=np.sqrt(f['covariance'][1,1]),removed=int(sum(src==i-age))))
        chosen_source=unique[0];bad=(src==chosen_source)
        for kind,values in [('translation',[.01,.02,.05]),('yaw_deg',[.05,.1,.25])]:
            for value in values:
                PP=P.copy()
                if kind=='translation':PP[bad]+=np.array([value,0])
                else:
                    # Rotate actual source points about that source LiDAR origin
                    # in the saved registration frame, then project to this plane.
                    poses=frames(run);current=np.asarray(poses[i]['lidar_pose_in_folder']);sourcepose=np.asarray(poses[int(chosen_source)]['lidar_pose_in_folder']);origin=(sourcepose[:3,3]-current[:3,3])@current[:3,:3];up=a['world_up_proxy'];xyz=pr['xyz'][ids[bad]]-origin;angle=np.deg2rad(value);rot=xyz*np.cos(angle)+np.cross(up,xyz)*np.sin(angle)+np.outer(xyz@up,up)*(1-np.cos(angle));PP[bad]+=(rot-xyz)@B[:,1:]
                f=fit(np.arange(len(P)),points=PP);rob=fit_anchor(PP,src,template,range_m=rr,ring=ring,state=state,config=dict(cfg['profile'],starts=4,source_weight='robust'));multi.append(dict(run=run,frame=i,station=s,experiment='BAD_SOURCE_'+kind,perturbation=value,n=len(P),sources=f['sources'],anchor_shift=float(np.linalg.norm(f['anchor']-full['anchor'])),robust_anchor_shift=float(np.linalg.norm(rob['anchor']-full['anchor'])),sigma_v=np.sqrt(f['covariance'][0,0]),sigma_w=np.sqrt(f['covariance'][1,1]),source=int(chosen_source),available=True))
    return result,multi

def curve_task(task):
    r,cfg=task;run=r['run'];i=r['frame'];m,a,p,pr=read_causal(run,i)
    if m['status']!='AVAILABLE' or len(a['anchor_knots'])<5:return [],[]
    ak=a['anchor_knots'];A=a['anchor_xyz'];F=plan_frame(a['initial_basis'],a['world_up_proxy']);s=grid(ak[-1]);middle=len(ak)//2;experiments=[];rows=[];gaprows=[]
    def fit(ks,aa,name):
        if name in ('natural','polyline','pchip','hermite'):return interpolation(ks,aa,s,F,name)
        cc=np.repeat(np.eye(3)[None]*.035**2,len(ks),axis=0);config=dict(cfg,physics=name!='smoothing',cluster=.25)
        ks,aa,_=cluster(ks,aa,.25);cc=np.repeat(np.eye(3)[None]*.035**2,len(ks),axis=0)
        return (factor_graph if name=='graph' else smooth_spline)(ks,aa,cc,s,F,config)[0]
    methods=('natural','polyline','pchip','hermite','smoothing','physical','graph');baselines={name:fit(ak,A,name) for name in methods}
    for axis in (1,2):
        for cm in (2,5,10,20,30):
            AA=A.copy();AA[middle]+=F[:,axis]*cm/100;experiments.append((('lateral' if axis==1 else 'vertical')+'_anchor',cm,ak,AA))
    for shift in (-2,-1,-.5,.5,1,2):
        ks=ak.copy();ks[middle]+=shift
        if np.all(np.diff(ks)>1e-5):experiments.append(('station_shift',shift,ks,A))
    for sep in (.05,.1,.2):
        for offset in (.05,.1,.2):
            ks=np.insert(ak,middle+1,ak[middle]+sep);AA=np.insert(A,middle+1,A[middle]+F[:,0]*sep+F[:,2]*offset,axis=0);experiments.append(('duplicate_'+str(sep),offset,ks,AA))
    for kind,value,ks,AA in experiments:
        for name in methods:
            C=fit(ks,AA,name);ph,_=physics(s,C,F,baselines[name],cfg['vertical_prior']);shift=np.linalg.norm(C-baselines[name],axis=1);rows.append(dict(run=run,frame=i,experiment=kind,perturbation=value,method=name,max_curve_shift=float(max(shift)),p95_curve_shift=float(np.percentile(shift,95)),**ph))
    pa,profiles,timing=profile_cache(run,i,a,p,pr,cfg['profile']);ss,aa,cv=pa['s'],pa['anchors'],pa['covariance'];base,cov,_=smooth_spline(ss,aa,cv,s,F,cfg);center=float(s[-1]*.65)
    for length in (1,2,4,8,12):
        keep=abs(ss-center)>length/2
        if keep.sum()<3:gaprows.append(dict(run=run,frame=i,gap_m=length,available=False));continue
        C,cc,_=smooth_spline(ss[keep],aa[keep],cv[keep],s,F,cfg);dist=np.min(abs(s[:,None]-ss[keep]),axis=1);cc+=np.eye(3)[None]*(.003*dist)[:,None,None]**2;inside=abs(s-center)<=length/2;after=(s>center+length/2)&(s<=center+length/2+4);gaprows.append(dict(run=run,frame=i,gap_m=length,available=True,removed=int(sum(~keep)),drift_p95=float(np.percentile(np.linalg.norm(C[inside]-base[inside],axis=1),95)) if inside.any() else None,uncertainty_trace=float(np.median(np.trace(cc[inside],axis1=1,axis2=2))) if inside.any() else None,reacquisition_error=float(np.median(np.linalg.norm(C[after]-base[after],axis=1))) if after.any() else None))
    return rows,gaprows

def main():
    require_reproduction();protocol=load(OUT/'protocol.json');cfg=next(c for c in protocol['candidates'] if c['name']=='T1_LATENT_PHYSICAL');dev=protocol['development'];chosen=[]
    for run in protocol['split']['development']:
        candidates=[r for r in dev if r['run']==run]
        chosen+=candidates[len(candidates)//2:len(candidates)//2+1]
    partial=[];multi=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,(a,b) in enumerate(pool.map(partial_task,[(r['run'],r['frame'],cfg) for r in chosen]),1):partial+=a;multi+=b;print('PARTIAL STRESS',j,len(chosen),flush=True)
    write_csv(OUT/'anchor_partial_visibility.csv',partial);save(OUT/'stress/partial_profiles.json',partial);write_csv(OUT/'multiview_ablation.csv',multi);save(OUT/'stress/multiview.json',multi)
    features=load(OUT/'protocol.json')['phase_a'];clean=[]
    for r in features:
        if not r.get('selection_allowed',True):continue
        f=OUT/'phase_a/C0_CURRENT'/key(r['run'],r['frame'])/'prediction.json'
        if not f.exists():continue
        v=load(f)
        if v['status']=='AVAILABLE' and v['physics']['S_phys']<.1:clean.append(r)
    curves=[];gaps=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for j,(a,b) in enumerate(pool.map(curve_task,[(r,cfg) for r in clean[:3]]),1):curves+=a;gaps+=b;print('CURVE STRESS',j,flush=True)
    write_csv(OUT/'stress/bad_anchor_stress.csv',curves);write_csv(OUT/'stress/gap_stress.csv',gaps);save(OUT/'stress/COMPLETE.json',dict(time_ns=time.time_ns(),partial_rows=len(partial),multiview_rows=len(multi),curve_rows=len(curves),gap_rows=len(gaps),GT_used_for_inference=False,partial_reference='Full observed same-profile anchor, not absolute physical GT',synthetic_perturbations=True))

if __name__=='__main__':main()
