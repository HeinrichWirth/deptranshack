from lc_common import *
from curve_geometry import plan_frame,physics

def main():
    require_reproduction();check_baseline();old=load(OLD_OUT/'protocol.json');dev=old['cohort']['development'];bench=old['cohort']['heldout'];features=[];vertical=[]
    for r in dev:
        folder=OLD_OUT/'c4_cr/seed8'/key(r['run'],r['frame']);m=load(folder/'input.json')
        if m['status']!='AVAILABLE':features.append(dict(**r,available=False));continue
        with np.load(folder/'input.npz') as z:a={k:z[k] for k in ('s','C','B','initial_basis','world_up_proxy','support','anchor_xyz','anchor_knots','cr_points')}
        s=a['s'];keep=np.r_[True,np.diff(s)>1e-7];F=plan_frame(a['initial_basis'],a['world_up_proxy']);ph,arr=physics(s[keep],a['C'][keep],F);vertical.extend(abs(arr['curvature_v'][2:-2]).tolist())
        features.append(dict(**r,available=True,horizon=float(s[-1]),sparse=float(np.median(a['support'])),curved=ph['heading_jump_8m_deg'],grade=ph['max_grade'],gap=float(np.max(np.diff(a['anchor_knots']))),**{k:ph[k] for k in ('S_phys','max_vertical_curvature')}))
    # Engineering smoothing scale, not a physical grade bound. Freeze before any
    # new candidate/future-GT evaluation; empirical from development C4 only.
    vp=max(.003,float(np.quantile(vertical,.9)))
    required=[('doubleT_platform',31),('roundT_pressureGate_roundT',169)]
    selected=[]
    def add(r):
        if not any(x['run']==r['run'] and x['frame']==r['frame'] for x in selected):selected.append(dict(r,selection_allowed=True))
    for run,i in required:add(next(r for r in dev if r['run']==run and r['frame']==i))
    avail=[r for r in features if r['available']]
    # Stratification uses frozen baseline geometry/support, not new benchmark GT.
    for field,reverse in [('S_phys',True),('horizon',True),('sparse',False),('curved',True),('grade',True),('gap',True)]:
        for r in sorted(avail,key=lambda r:r[field],reverse=reverse)[:3]:add(dict(run=r['run'],frame=r['frame'],split='development',stratum=field))
    for run in old['split']['development']:
        rr=[r for r in avail if r['run']==run]
        for r in rr[::max(1,len(rr)//3)][:3]:
            if len(selected)<36:add(dict(run=r['run'],frame=r['frame'],split='development',stratum='run_balance'))
    for r in dev:
        if len(selected)<36:add(r)
    mandatory=[('squareT_platform_squareT_switch',190),('roundT_squareT_pressureGate_squareT',51),('squareT_platform_squareT_switch',76),('roundT_squareT_pressureGate_squareT',44)]
    selected += [dict(run=r,frame=i,split='reused_benchmark_diagnostic',selection_allowed=False,stratum='mandatory_known_failure') for r,i in mandatory]
    assert len(selected)==40
    base=dict(knots=4,loss='huber',prior='all',smooth=1.,physics=True,vertical_prior=vp,cluster=.25,profile=dict(representation='pointset',fusion='balanced',source_weight='equal',loss='cauchy',starts=4))
    candidates=[]
    def addcfg(name,kind,latent=False,**changes):
        cfg=dict(base,name=name,kind=kind,use_profile=latent);cfg.update(changes);candidates.append(cfg)
    addcfg('C0_CURRENT','current',False)
    for kind,num in [('polyline',1),('pchip',2),('hermite',3)]:addcfg(f'C{num}_{kind.upper()}',kind,False,cluster=0.)
    addcfg('C4_SMOOTH','spline',False,physics=False)
    addcfg('C5_PHYSICAL','spline',False)
    addcfg('P1_HARD_PROJECTION','spline',False,hard_projection=True)
    addcfg('P2_PROFILE_WEAK','spline',True,physics=False,prior='none')
    addcfg('T1_LATENT_SMOOTH','spline',True,physics=False)
    addcfg('T1_LATENT_PHYSICAL','spline',True)
    addcfg('C6_FACTOR_GRAPH','graph',True)
    addcfg('C7_JOINT_EM','joint',True)
    for k in (1,2,6,8):addcfg(f'KNOT_{k}','spline',True,knots=k)
    for loss in ('cauchy','tukey','student'):addcfg('LOSS_'+loss.upper(),'spline',True,loss=loss)
    for prior in ('none','kappa','kappa_prime','vertical'):addcfg('PRIOR_'+prior.upper(),'spline',True,prior=prior,physics=prior!='none')
    for window in (.1,.5):addcfg('CLUSTER_'+str(window),'spline',False,cluster=window)
    for rep in ('occupancy','envelope','subsets'):addcfg('PROFILE_'+rep.upper(),'spline',True,profile=dict(base['profile'],representation=rep))
    for fusion in ('concat','current'):addcfg('FUSION_'+fusion.upper(),'spline',True,profile=dict(base['profile'],fusion=fusion))
    for weight in ('quality','robust'):addcfg('SOURCE_'+weight.upper(),'spline',True,profile=dict(base['profile'],source_weight=weight))
    # Coordinate-dependent generic XYZ counterpart to the plan+vertical model.
    addcfg('XYZ_GENERIC_DIAGNOSTIC','spline',False,generic_xyz=True,physics=False)
    protocol=dict(created_ns=time.time_ns(),baseline=sha(OUT/'baseline_freeze.json'),split=old['split'],phase_a=selected,development=dev,benchmark=bench,candidates=candidates,vertical_prior=vp,vertical_prior_source='90th percentile absolute vertical curvature of frozen development C4 curves, floor 0.003/m; smoothing scale, not normative bound',constraints=[dict(file='tools/trajectory_predictor.py',symbol='KMAX',value=.01,usage='hard diagnostic and soft optimized factor; no curvature clipping'),dict(file='tools/trajectory_predictor.py',symbol='GMAX',value=.045,usage='documented project heuristic only; NOT enforced as raw-Z physical limit'),dict(file='docs/SMOOTH_RAILS.md',usage='engineering parameters are not normative radii')],downstream='frozen B4_SPLINE_50__O1; same seed observed heads rebased to each curve; all hyperparameters fixed',selection='Only development; compulsory benchmark examples diagnostic-only; no ranking on them. Successive halving then leave-one-run-out ranking. Top3 frozen before single benchmark prediction/evaluation.',new_holdout='NOT consumed; list directory candidates in final audit',fresh_GT_used_for_protocol=False)
    save(OUT/'protocol.json',protocol);write_csv(OUT/'development_features.csv',features);save(STAGE/'config/candidates.json',candidates);print('PROTOCOL',len(selected),'phase A',len(candidates),'candidates; vertical scale',vp)

if __name__=='__main__':main()
