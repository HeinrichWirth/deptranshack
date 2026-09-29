"""Causal-only STEP6.1 adapter and inference. Evaluation is a separate process.

Frozen current 0..8m head observations are the provided initial cross-section.
No class arrays are read here, even from the old adapter cache.
"""
from lc_common import *
from curve_geometry import *
from profile_likelihood import fit_anchor
from track_geometry import offsets,rail_covariance
from rail_models import predict,INPUT_KEYS
from seed_input import canonical

def read_causal(run,i):
    p,pr=read_c4(run,i);folder=OLD_OUT/'c4_cr/seed8'/key(run,i);mark=load(folder/'COMPLETE.json')
    for f,h in mark['files'].items():assert sha(folder/f)==h
    m=load(folder/'input.json')
    if m['status']!='AVAILABLE':return m,{},p,pr
    allowed=set(INPUT_KEYS)|{'initial_basis','anchor_knots','anchor_xyz'}
    with np.load(folder/'input.npz') as z:a={k:z[k] for k in allowed}
    assert not {'seed_labels','seed_points','class1','class2'}&set(a)
    return m,a,p,pr

def seed_rebase(seed,s,C,B,q,initial):
    """Same frozen robust estimator, applied to the SAME provided head sections.

    Only the reference curve/plane changes. No fresh label or raw rail search.
    """
    obs=[];u=C@initial[:,0]
    for old in seed['observations']:
        heads=np.asarray(old['heads']).copy();long=float(np.mean(heads@initial[:,0]));station=float(np.interp(long,u,s));cr=np.array([np.interp(station,s,C[:,j]) for j in range(3)]);bb=B[int(np.argmin(abs(s-station)))];heads-=np.outer((heads-cr)@bb[:,0],bb[:,0]);x=offsets(cr,bb,heads,q);obs.append(dict(s=station,x=x,heads=heads,cr=cr,support=old['support']))
    xx=np.array([o['x'] for o in obs]);ss=np.array([o['s'] for o in obs]);xx[:,0]=np.unwrap(xx[:,0]);med=np.median(xx,axis=0);mad=1.4826*np.median(abs(xx-med),axis=0);P=np.cov(xx.T)/max(1,len(xx)/2)+np.diag(np.array([np.deg2rad(.1),.004,.004,.006,.004])**2);slope=np.polyfit(ss-ss.mean(),xx,1)[0]
    return dict(seed,x0=med,covariance=P,mad=mad,slope=slope,observations=obs,reference_s=float(np.median(ss)))

def c4_provenance(p,pr,run,i):
    out={k:pr[k] for k in pr};out.update(march_step=p['point_step'],template_residual=p['template_residual'],confidence=p['confidence'],state=p['point_state'],initial_march_station=p['s_from_seed'])
    # Actual source frame timestamp, not invented per-point acquisition time.
    rows=frames(run);out['source_header_time_ns']=np.array([rows[int(j)]['header_time_ns'] for j in pr['source_frame']],dtype=np.int64)
    assert np.all(out['source_frame']<=i)
    return out

def fit_profiles(s,C,B,ps,pr,p,q,cfg):
    profiles=[];obss=[];obsa=[];obscov=[];maskgood=(p['point_state']==2)|(p['point_state']==1);template=canonical()*np.array([q,1]);profile_time=0.;fusion_time=0.
    for k,station in enumerate(s):
        ids=np.flatnonzero((abs(ps-station)<=1.05)&maskgood)
        if cfg.get('fusion')=='current' and len(ids):ids=ids[pr['source_frame'][ids]==pr['source_frame'].max()]
        if len(ids):
            # Detrend along the provisional curve before merging the slab.
            centers=np.column_stack([np.interp(ps[ids],s,C[:,j]) for j in range(3)]);z=np.einsum('ni,ij->nj',pr['xyz'][ids]-centers,B[k,:,1:]);fit=fit_anchor(z,pr['source_frame'][ids],template,range_m=pr['sensor_distance_at_source'][ids],ring=pr['ring'][ids],age_distance=pr['age_distance'][ids],state=p['point_state'][ids],config=cfg)
            anchor=C[k]+B[k,:,1:]@fit['anchor'];cov=B[k,:,1:]@fit['covariance']@B[k,:,1:].T+np.outer(B[k,:,0],B[k,:,0])*.02**2;obss.append(station);obsa.append(anchor);obscov.append(cov)
        else:fit=fit_anchor(np.empty((0,2)),np.empty(0,dtype=int),template,config=cfg);anchor=C[k];cov=B[k,:,1:]@fit['covariance']@B[k,:,1:].T+np.outer(B[k,:,0],B[k,:,0])*.02**2
        profiles.append(dict(station=float(station),anchor_xyz=anchor,covariance_xyz=cov,point_ids=ids.tolist(),**fit));profile_time+=fit['ms'];fusion_time+=fit['fusion_ms']
    return np.array(obss),np.array(obsa),np.array(obscov),profiles,dict(profile_ms=profile_time,fusion_ms=fusion_time)

def profile_cache(run,i,a,p,pr,cfg):
    signature=hashlib.sha256((json.dumps(cfg,sort_keys=True)+sha(STAGE/'src/profile_likelihood.py')+sha(STAGE/'src/curve_geometry.py')).encode()).hexdigest()[:16];folder=OUT/'causal_profiles'/signature/key(run,i)
    if (folder/'COMPLETE.json').exists():
        marker=load(folder/'COMPLETE.json')
        for f,h in marker['files'].items():assert sha(folder/f)==h
        with np.load(folder/'anchors.npz') as z:arr={k:z[k] for k in z.files}
        return arr,load(folder/'profiles.json'),load(folder/'timing.json')
    ak=a['anchor_knots'];A=a['anchor_xyz'];initial=a['initial_basis'];F=plan_frame(initial,a['world_up_proxy']);s=grid(float(ak[-1]),2);C=interpolation(ak,A,s,F,'pchip',.25);B=curve_frames(s,C,initial);ps,corrected=assign_stations(pr['xyz'],ak,A,pr['source_frame'],p['s_from_seed']);ss,aa,cov,profiles,timing=fit_profiles(s,C,B,ps,pr,p,int(p['seed']['side']),cfg)
    arr=dict(s=ss,anchors=aa,covariance=cov,profile_s=s,initial_C=C,initial_B=B,point_station=ps);folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'anchors.npz',**arr);save(folder/'profiles.json',profiles);save(folder/'timing.json',dict(timing,station_order_corrections=corrected));np.savez_compressed(folder/'provenance.npz',**c4_provenance(p,pr,run,i),assigned_station=ps)
    save(folder/'COMPLETE.json',dict(time_ns=time.time_ns(),files={f:sha(folder/f) for f in ('anchors.npz','profiles.json','timing.json','provenance.npz')},future_GT_used=False,source_max_frame=int(pr['source_frame'].max()),current_frame=i,labels_read=False))
    return arr,profiles,timing

def infer_arrays(m,a,p,pr,cfg,run,i):
    start=time.perf_counter();initial=a['initial_basis'];F=plan_frame(initial,a['world_up_proxy']);ak=a['anchor_knots'];A=a['anchor_xyz'];s=grid(float(ak[-1]));oldC=np.column_stack([np.interp(s,a['s'],a['C'][:,j]) for j in range(3)]);linear=interpolation(ak,A,s,F,'polyline');profiles=[];ptiming={};optim={};anchor_time=time.perf_counter()
    if cfg['kind']=='current':
        C=oldC;cov=np.array([np.eye(3)*x*x for x in np.interp(s,a['s'],a['position_sigma'])]);B=np.array([a['B'][np.argmin(abs(a['s']-x))] for x in s]);seed=m['seed']
    elif cfg['kind'] in ('polyline','pchip','hermite'):
        C=interpolation(ak,A,s,F,cfg['kind'],cfg['cluster']);B=curve_frames(s,C,initial);cov=np.repeat(np.eye(3)[None]*.025**2,len(s),axis=0)
    else:
        ks,AA,groups=cluster(ak,A,cfg['cluster']);ss=ks;aa=AA;cv=np.repeat(np.eye(3)[None]*.035**2,len(ss),axis=0)
        if cfg['use_profile']:
            pa,profiles,ptiming=profile_cache(run,i,a,p,pr,cfg['profile']);ss=pa['s'];aa=pa['anchors'];cv=pa['covariance']
            # C4 anchors secondary factors have deliberately lower information
            # than a supported raw-profile anchor; they preserve gap observability.
            secondary=np.repeat(np.eye(3)[None]*.15**2,len(ks),axis=0);secondary[0]=np.eye(3)*.02**2
            ss=np.r_[ss,ks];aa=np.vstack([aa,AA]);cv=np.concatenate([cv,secondary]);order=np.argsort(ss);ss,aa,cv=ss[order],aa[order],cv[order]
        anchor_time=(time.perf_counter()-anchor_time)*1000;curve_start=time.perf_counter();fit_frame=np.eye(3) if cfg.get('generic_xyz') else F
        if cfg['kind']=='graph':C,cov,optim=factor_graph(ss,aa,cv,s,fit_frame,cfg)
        else:C,cov,optim=smooth_spline(ss,aa,cv,s,fit_frame,cfg)
        if cfg['kind']=='joint':
            history=[];B=curve_frames(s,C,initial);ps=pa['point_station'];previous=np.inf
            for iteration in range(6):
                # E: correspondence to empirical surface; station order retained.
                ps,order_corrections=assign_stations(pr['xyz'],s,C,pr['source_frame'],p['s_from_seed'])
                sub=grid(float(s[-1]),2);CC=np.column_stack([np.interp(sub,s,C[:,j]) for j in range(3)]);BB=curve_frames(sub,CC,initial)
                es,ea,ec,ep,et=fit_profiles(sub,CC,BB,ps,pr,p,m['q'],dict(cfg['profile'],starts=2))
                order=np.argsort(np.r_[es,ks]);ss=np.r_[es,ks][order];aa=np.vstack([ea,AA])[order];cv=np.concatenate([ec,np.repeat(np.eye(3)[None]*.15**2,len(ks),axis=0)])[order]
                # M: solve physically regularized curve, with soft measurements.
                new,cov,op=smooth_spline(ss,aa,cv,s,F,cfg);change=float(np.max(np.linalg.norm(new-C,axis=1)));objective=op['final_objective'];C=new;profiles=ep;history.append(dict(iteration=iteration+1,max_change=change,complete_M_objective=objective,relative_objective_change=abs(previous-objective)/max(1,objective),station_order_corrections=order_corrections));ptiming['profile_ms']+=et['profile_ms'];ptiming['fusion_ms']+=et['fusion_ms']
                if iteration>=2 and abs(previous-objective)/max(1,objective)<1e-4:break
                previous=objective
            optim=dict(optim,em_history=history)
        curve_ms=(time.perf_counter()-curve_start)*1000;B=curve_frames(s,C,initial)
    if cfg['kind']!='current':seed=seed_rebase(m['seed'],s,C,B,m['q'],initial)
    ph,phys=physics(s,C,F,linear,cfg['vertical_prior']);n=len(s);support=np.interp(s,a['s'],a['support']);state=np.array([a['cr_state'][np.argmin(abs(a['s']-x))] for x in s]);tangent=np.maximum(np.deg2rad(.1),np.interp(s,a['s'],a['tangent_sigma']));pos=np.sqrt(np.maximum(np.trace(cov,axis1=1,axis2=2)/3,1e-8))
    if profiles:
        ps=np.array([v['station'] for v in profiles]);gapdist=np.min(abs(s[:,None]-ps[np.array([v['n']>0 for v in profiles])][None,:]),axis=1) if any(v['n']>0 for v in profiles) else s
        cov+=np.eye(3)[None]*(.003*gapdist)[:,None,None]**2
    if cfg.get('calibration'):
        for j in range(n):
            distance=float(np.linalg.norm(C[j]));record=next(v for v in cfg['calibration'] if v['lo']<=distance<v['hi']);T=B[j,:,1:];local=T.T@cov[j]@T;floor=np.array([record['sigma_v'],record['sigma_w']])**2;cov[j]+=T@np.diag(np.maximum(floor-np.diag(local),0))@T.T
    data=dict(s=s,C=C,B=B,legacy_B=B,cr_state=state,support=support,position_sigma=pos,tangent_sigma=tangent,world_up_proxy=a['world_up_proxy']);oldgroup='phase_b' if run in load(OUT/'protocol.json')['split']['development'] else 'phase_e';record=load(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/key(run,i)/'prediction.json');priors=load(OLD_OUT/'models/global_prior.json');prior=priors['leave_run_out'].get(run,priors['default']);pred=predict(seed,data,[dict(beta=np.nan,sigma=.2,informative=False) for _ in s],m['q'],record['config'],prior)
    if cfg['kind']!='current':
        # Exact frozen propagation Jacobians and roll/offset state; replace only
        # the CR position information block with anisotropic latent covariance.
        scale=record['config']['corridor_scale']
        for j in range(n):
            Pc=cov[j]+np.eye(3)*(.004*pred['cr_gap_age'][j])**2;rc,J=rail_covariance(C[j],B[j],pred['state'][j],m['q'],pred['state_cov'][j],Pc,tangent[j]);pred['rail_cov'][j]=rc*scale**2;Jc=(J[0]+J[1])/2;rr=pred['pair'][j].mean(axis=0)-C[j];bb0,nn0=__import__('track_geometry').cross_axes(B[j],pred['state'][j,0]);TT=np.column_stack((np.cross(bb0,rr),np.cross(nn0,rr)));pred['center_cov'][j]=(Pc+Jc@pred['state_cov'][j]@Jc.T+TT@TT.T*tangent[j]**2)*scale**2
            bb,nn=__import__('track_geometry').cross_axes(B[j],pred['state'][j,0]);pred['sigma_lateral'][j]=np.sqrt(np.maximum(np.einsum('i,rij,j->r',bb,pred['rail_cov'][j],bb),0));pred['sigma_vertical'][j]=np.sqrt(np.maximum(np.einsum('i,rij,j->r',nn,pred['rail_cov'][j],nn),0))
    else:
        # C0 is exactly the gated reproduction, including unused diagnostic arrays
        # and the frozen endpoint convention. No new seed rebasing or covariance.
        with np.load(OUT/'reproduction'/oldgroup/key(run,i)/'prediction.npz') as z:pred={k:z[k] for k in z.files}
    info=dict(seed=seed,physics=ph,optimizer=optim,profiles=profiles,timing=dict(profile_likelihood_ms=ptiming.get('profile_ms',0),anchor_estimation_ms=anchor_time if isinstance(anchor_time,float) and anchor_time<1e7 else 0,curve_optimization_ms=locals().get('curve_ms',0),fusion_ms=ptiming.get('fusion_ms',0),total_ms=(time.perf_counter()-start)*1000),frame_matrix=F,GT_used=False,labels_read=False,source_max_frame=int(pr['source_frame'].max()),current_frame=i)
    arrays=dict(s=s,C=C,B=B,covariance=cov,tangent_sigma=tangent,old_C=oldC,linear_C=linear,anchor_s=ak,anchor_xyz=A,**phys)
    return pred,arrays,info

def run_case(task):
    r,group,configs=task;run=r['run'];i=r['frame'];m,a,p,pr=read_causal(run,i);rows=[]
    for cfg in configs:
        dest=OUT/group/cfg['name']/key(run,i)
        if (dest/'PREDICTION_COMPLETE.json').exists():
            marker=load(dest/'PREDICTION_COMPLETE.json')
            if marker.get('producer')==producer_hash():rows.append(load(dest/'prediction.json'));continue
        dest.mkdir(parents=True,exist_ok=True)
        if m['status']=='AVAILABLE':pred,curve,info=infer_arrays(m,a,p,pr,cfg,run,i)
        else:pred,curve,info={},{},dict(timing={},physics={})
        np.savez_compressed(dest/'prediction.npz',**pred);np.savez_compressed(dest/'curve.npz',**curve);save(dest/'details.json',info);row=dict(run=run,i=i,method=cfg['name'],status=m['status'],config=cfg,physics=info['physics'],timing=info['timing']);save(dest/'prediction.json',row);save(dest/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),producer=producer_hash(),files={f:sha(dest/f) for f in ('prediction.npz','curve.npz','details.json','prediction.json')},GT_used=False,labels_read=False));rows.append(row)
    return rows

def producer_hash():return {n:sha(STAGE/'src'/n) for n in ('latent_inference.py','curve_geometry.py','profile_likelihood.py')}
