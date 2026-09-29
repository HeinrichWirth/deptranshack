"""Causal STEP6 adapter. The only label access is CURRENT T inside the seed."""
from rr_common import *
from track_geometry import *
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
sys.path[:0]=[str(ROOT/'MVP/stages/01_rail2d/src'),str(ROOT/'MVP/stages/04_contact_rail_final')]
from rail2d_head_v2 import detect as detect_rails
from contact_rail_step2 import ContactRailDetector

def read_seed_labels(run,i,B,length=8.):
    if length not in (4,8,12,16):raise ValueError('Unsupported seed length')
    row=frames(run)[i];path=dataset()/run/row['file'];pose=np.asarray(row['lidar_pose_in_folder'])
    with laspy.open(path) as f:h=f.header
    raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
    # Coordinate filtering precedes any indexing of classification values.
    xyz=np.column_stack([raw[a].astype(float)*h.scales[k]+h.offsets[k] for k,a in enumerate(('X','Y','Z'))]);xyz=(xyz-pose[:3,3])@pose[:3,:3]
    u=xyz@B[:,0];ids=np.flatnonzero((u>=0)&(u<=length)&np.isfinite(xyz).all(axis=1))
    field='raw_classification' if h.point_format.id<6 else 'classification';mask=31 if h.point_format.id<6 else 255
    labels=(raw[field][ids]&mask).copy();seed=xyz[ids].copy();del raw
    return seed,labels,dict(frame=i,label_rows_read=len(ids),label_u_min=float(u[ids].min()) if len(ids) else None,label_u_max=float(u[ids].max()) if len(ids) else None,seed_length=length,source_size=path.stat().st_size,source_mtime_ns=path.stat().st_mtime_ns)

def head_center(points):
    """Upper surface anchor; keep the estimator identical for seed and offline GT."""
    p=np.unique(np.round(points,5),axis=0)
    if len(p)<5:return None
    upper=p[p[:,2]>=np.quantile(p[:,2],.7)]
    if len(upper)<3:return None
    center=np.median(upper,axis=0)
    return center,dict(points=len(p),surface_points=len(upper),spread=np.median(abs(upper-center),axis=0)*1.4826)

def calibrate_seed(seed_points,labels,model,initial,q,length=8.):
    vw=seed_points@initial
    cfg=load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json')
    det=detect_rails(vw[:,1:],cfg,details=False)
    if det['status']!='ok':return dict(status='SEED_PAIR_UNAVAILABLE',reason=det['reason'])
    pair=np.asarray(det['pair']);observations=[]
    for k in range(int(length)):
        inside=(vw[:,0]>=k)&(vw[:,0]<k+1)&(labels==1);heads=[];diag=[]
        for a in pair:
            mask=inside&(np.linalg.norm(vw[:,1:]-a,axis=1)<.20)
            result=head_center(vw[mask])
            if result is None:break
            heads.append(result[0]@initial.T);diag.append(result[1])
        if len(heads)!=2:continue
        u=float(np.mean(np.array(heads)@initial[:,0]));station=float(np.interp(u,model['C']@initial[:,0],model['s']));C=sample_curve(model,station)
        if not np.isfinite(C).all():continue
        j=int(np.argmin(abs(model['s']-station)));B=model['B'][j]
        near=1 if q>0 else 0;heads=np.array([heads[near],heads[1-near]])
        # Project both heads to one plane before measuring the cross-section.
        heads-=np.outer((heads-C)@B[:,0],B[:,0])
        x=offsets(C,B,heads,q)
        observations.append(dict(s=station,x=x,heads=heads,cr=C,support=diag))
    if len(observations)<2:return dict(status='SEED_TOO_SPARSE',sections=len(observations),detector_pair=pair)
    xx=np.array([o['x'] for o in observations]);ss=np.array([o['s'] for o in observations]);xx[:,0]=np.unwrap(xx[:,0]);med=np.median(xx,axis=0);mad=1.4826*np.median(abs(xx-med),axis=0)
    # Section-to-section variability is preserved, not divided by raw duplicate point count.
    P=np.cov(xx.T)/max(1,len(xx)/2)+np.diag(np.array([np.deg2rad(.1),.004,.004,.006,.004])**2)
    slope=np.polyfit(ss-ss.mean(),xx,1)[0]
    return dict(status='AVAILABLE',x0=med,covariance=P,mad=mad,slope=slope,sections=len(xx),observations=observations,reference_s=float(np.median(ss)),seed_length=length,detector_pair=pair)

@lru_cache(maxsize=1)
def canonical():return ContactRailDetector().template

def fit_profile(q,side):
    q=np.unique(np.round(q,3),axis=0);n=len(q)
    if n<8:return dict(beta=np.nan,sigma=np.deg2rad(10),n=n,informative=False)
    if n>500:q=q[np.linspace(0,n-1,500,dtype=int)]
    template=canonical()*np.array([side,1]);tree=cKDTree(template)
    def residual(x):
        c,s=np.cos(x[0]),np.sin(x[0]);R=np.array([[c,-s],[s,c]])
        z=(q-x[1:])@R;_,ids=tree.query(z)
        return (z-template[ids]).ravel()/.008
    solutions=[]
    for a in (0.,-.04,.04):
        fit=least_squares(residual,[a,0,0],bounds=([-.175,-.08,-.08],[.175,.08,.08]),loss='soft_l1',max_nfev=30,diff_step=1e-3)
        solutions.append(fit)
    best=min(solutions,key=lambda f:f.cost);error=float(np.sqrt(np.mean(residual(best.x)**2))*.008)
    cov=np.linalg.pinv(best.jac.T@best.jac)*max(1.,np.sum(residual(best.x)**2)/max(1,2*len(q)-3))
    sigma=max(np.deg2rad(.35),float(np.sqrt(max(cov[0,0],0))))
    competing=[r.x[0] for r in solutions if r.cost<=best.cost*1.05+1e-6]
    span=float(np.ptp(competing)) if len(competing)>1 else 0.
    extent=np.ptp(q,axis=0);informative=bool(min(extent)>.035 and sigma<np.deg2rad(3) and abs(best.x[0])<.17 and span<np.deg2rad(2) and error<.025)
    return dict(beta=float(best.x[0]),sigma=sigma,n=n,informative=informative,residual=error,translation=best.x[1:],multi_start_angle_span=span,extent=extent)

def build_input(run,i,length=8.):
    start=time.perf_counter();p,points=read_c4(run,i)
    if p['seed']['status']!='AVAILABLE':return dict(status='C4_SEED_UNAVAILABLE',reason=p['seed'].get('reason'),run=run,i=i),{}
    initial=np.asarray(p['seed']['basis']);q=int(p['seed']['side']);a=np.asarray(p.get('curve',[p['seed']['anchor']]))
    if len(a)<2:a=np.array([p['seed']['anchor'],np.asarray(p['seed']['anchor'])+8*initial[:,0]])
    model=curve_model(a,initial)
    if model is None:return dict(status='C4_CURVE_UNAVAILABLE',run=run,i=i),{}
    xyz,labels,audit=read_seed_labels(run,i,initial,length);seed=calibrate_seed(xyz,labels,model,initial,q,length)
    if seed['status']!='AVAILABLE':return dict(status=seed['status'],run=run,i=i,seed=seed,label_audit=audit),{}
    C,B,s=model['C'],model['B'],model['s'];pts=points['xyz'];states=p['point_state'];profiles=[];counts=[];crstate=[];legacy=[];position_sigma=[];tangent_sigma=[]
    stepanchors=np.array([p['seed']['anchor']]+[st['anchor_3d'] for st in p['steps'] if st.get('anchor_3d') is not None]);stepbases=np.array([initial]+[st['basis'] for st in p['steps'] if st.get('anchor_3d') is not None])
    for k,(center,BB) in enumerate(zip(C,B)):
        rel=(pts-center)@BB;inside=(abs(rel[:,0])<=2)&(np.linalg.norm(rel[:,1:],axis=1)<.30);take=inside&(states==2)
        z=rel[take,1:];fit=fit_profile(z,q);profiles.append(fit);n=len(np.unique(np.round(pts[take],3),axis=0));counts.append(n)
        state=2 if n>=2 else (1 if inside.any() else 3);crstate.append(state)
        legacy.append(stepbases[np.argmin(np.linalg.norm(stepanchors-center,axis=1))])
        # C4 confidence is not a probability. These are explicit uncertainty hypotheses.
        residuals=p['template_residual'][take];finite=residuals[np.isfinite(residuals)];res=float(np.median(finite)) if len(finite) else .02
        position_sigma.append(max(.01,res)+(.015 if n<5 else .005))
        local=max(0,k-2);high=min(len(C)-1,k+2);angle=np.arccos(np.clip(B[local,:,0]@B[high,:,0],-1,1));tangent_sigma.append(max(np.deg2rad(.1),float(angle)/4))
    meta=dict(status='AVAILABLE',run=run,i=i,q=q,seed=seed,label_audit=audit,profiles=profiles,input_kind='CR1_C4',c4_folder=c4_path(run,i).relative_to(ROOT).as_posix(),source_max_frame=int(points['source_frame'].max()),current_frame=i,side_switch_capability='C4 carries initial side only; no confirmed side-switch event in this frozen output.',ms=(time.perf_counter()-start)*1000)
    arrays=dict(s=s,C=C,B=B,legacy_B=np.array(legacy),cr_state=np.array(crstate,dtype=np.uint8),support=np.array(counts),position_sigma=np.array(position_sigma),tangent_sigma=np.array(tangent_sigma),cr_points=pts,cr_point_state=states,cr_source_frame=points['source_frame'],seed_points=xyz,seed_labels=labels,initial_basis=initial,anchor_knots=model['knots'],anchor_xyz=model['anchors'],world_up_proxy=np.asarray(frames(run)[i]['lidar_pose_in_folder'])[:3,:3].T@np.array([0.,0.,1.]))
    return meta,arrays

def save_input(run,i,length=8):
    folder=OUT/'c4_cr'/('seed'+str(length))/key(run,i)
    producer={n:sha(STAGE/'src'/n) for n in ('seed_input.py','track_geometry.py')}
    if (folder/'COMPLETE.json').exists() and load(folder/'COMPLETE.json').get('producer_sha256')==producer:return folder
    meta,arrays=build_input(run,i,length);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'input.npz',**arrays);save(folder/'input.json',meta)
    save(folder/'COMPLETE.json',dict(time_ns=time.time_ns(),files={n:sha(folder/n) for n in ('input.json','input.npz')},producer_sha256=producer,future_labels_used=False,post_seed_raw_rail_points_read=False))
    return folder

def read_input(folder):
    mark=load(folder/'COMPLETE.json')
    for n,h in mark['files'].items():assert sha(folder/n)==h
    meta=load(folder/'input.json')
    with np.load(folder/'input.npz') as z:a={k:z[k] for k in z.files}
    return meta,a
