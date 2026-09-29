"""Observation-only spatial state/beam tracker. No labels, reference or future access."""
from long_common import np, CFG, time, copy
from roi_source import point_keys
from tracker import TemplateTracker
from geometry import pca, transport, angles, unit, binned
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

CONFIRMED=2
TENTATIVE=1

def template_part(template,mode):
    if mode in ('full','subset'):return template
    v=np.round(template[:,0],3);w=np.round(template[:,1],3)
    vv,vc=np.unique(v,return_counts=True);ww,wc=np.unique(w,return_counts=True)
    vx=vv[np.argmax(vc)];wx=ww[np.argmax(wc)]
    if mode=='vertical':keep=abs(template[:,0]-vx)<.004
    elif mode=='horizontal':keep=abs(template[:,1]-wx)<.004
    else:keep=np.linalg.norm(template-[vx,wx],axis=1)<=.04
    return template[keep] if keep.sum()>=3 else template

def confirmed_blocks(path):return [b for b in path['blocks'] if b['state']==CONFIRMED]

def track_length(path):
    a=[b['anchor_3d'] for b in confirmed_blocks(path)]
    return float(np.linalg.norm(np.diff(a,axis=0),axis=1).sum()) if len(a)>1 else 0.

def prior_direction(path,cfg,default):
    blocks=confirmed_blocks(path)
    if cfg['geometry']=='T':blocks=[b for b in blocks if np.any(b['age']==0)]
    n=cfg['smooth'] or (3 if cfg['prior']=='P1' else 6)
    a=np.array([b['anchor_3d'] for b in blocks[-n:]])
    if cfg['prior']=='P0' or len(a)<3:return default,np.zeros(3),0.
    ds=np.linalg.norm(np.diff(a,axis=0),axis=1);s=np.r_[0,np.cumsum(ds)]
    if np.any(ds<.1):return default,np.zeros(3),0.
    x=s-s[-1]
    if cfg['prior']=='P1':
        t0=unit(a[-2]-a[-3]);t1=unit(a[-1]-a[-2]);curv=(t1-t0)/max(.5,(ds[-1]+ds[-2])/2)
        tangent=unit(t1+curv*min(cfg['advance'],4.))
    elif cfg['prior']=='P3':
        # Robust smooth local anchors before a natural cubic derivative; observed points unchanged.
        co=np.polyfit(x,a,2);fit=np.polyval(co,x[:,None]) if False else np.stack([np.polyval(co[:,j],x) for j in range(3)],axis=1)
        d=np.linalg.norm(fit-a,axis=1);weight=np.minimum(1.,.03/np.maximum(d,1e-9))
        sm=a*weight[:,None]+fit*(1-weight[:,None]);sp=CubicSpline(x,sm,bc_type='natural')
        tangent=unit(sp(0,1));curv=sp(0,2)/max(np.linalg.norm(sp(0,1))**2,1e-8)
    else:
        def fun(z):return ((np.c_[np.ones(len(x)),x,x*x]@z.reshape(3,3)-a)/.03).ravel()
        initial=np.linalg.lstsq(np.c_[np.ones(len(x)),x,x*x],a,rcond=None)[0]
        coef=least_squares(fun,initial.ravel(),loss='huber',max_nfev=15).x.reshape(3,3)
        tangent=unit(coef[1]+2*coef[2]*min(cfg['advance'],4.));curv=2*coef[2]
    # Priors only cover the next short segment; reject numerically implausible fits.
    curvature=float(np.linalg.norm(curv));turn=np.rad2deg(np.arccos(np.clip(tangent@default,-1,1)))
    if curvature>.03 or turn>5:return default,np.zeros(3),curvature
    return tangent,curv,curvature

def frame_for(path,cfg):
    B=path['B'];C=path['C'];blocks=confirmed_blocks(path)
    support=np.concatenate([b['xyz'] if cfg['geometry']=='accepted' else b['xyz'][b['age']==0] for b in blocks])
    support_keys=np.concatenate([b['keys'] if cfg['geometry']=='accepted' else b['keys'][b['age']==0] for b in blocks])
    # Frozen M1 uses source-row order through its seen mask. Preserve that summation
    # order: tiny PCA roundoff otherwise changes nearest-template local minima.
    support=support[np.argsort(support_keys,kind='stable')]
    if not len(support):raise ValueError('NO_CURRENT_GEOMETRY')
    su=(support-C)@B[:,0];end=su.max();recent=support[su>=end-8.]
    t,center=pca(binned(recent,B[:,0],.5) if cfg['tube'] else recent,B[:,0],robust=cfg['tube'])
    t,curv,kappa=prior_direction(path,cfg,t)
    BB=transport(B,t);advance=cfg['advance'] if np.linalg.norm(C)>=30 else 4.
    CC=center+B[:,0]*(advance-float((center-C)@B[:,0]))
    # Pending observations guide only the next search position, never the confirmed PCA.
    if path['pending'] or path['gap_count']:
        CC=C+B[:,0]*advance
        if path['pending']:
            last=path['blocks'][-1];d=(last['anchor_3d']-CC)@BB[:,1:]
            CC=CC+np.clip(d,-.08,.08)@BB[:,1:].T
    a=angles(B,BB)
    if np.hypot(a['delta_yaw'],a['delta_pitch'])>5:raise ValueError('ORIENTATION_JUMP')
    current=np.concatenate([b['xyz'][b['age']==0] for b in blocks])
    current_keys=np.concatenate([b['keys'][b['age']==0] for b in blocks])
    current=current[np.argsort(current_keys,kind='stable')]
    q=(current-CC)@BB;old=q[(q[:,0]>=0)&(q[:,0]<=cfg['window']),1:]
    return CC,BB,old,curv,kappa,a

def limits(cfg,path,C):
    shape=cfg['shape_limit'];gate=cfg['anchor_gate'];overlap=cfg['overlap_limit']
    maturity=track_length(path);r=np.linalg.norm(C)
    if cfg['relaxation']!='none':
        # All weak relaxations also require an already observed continuous track.
        if maturity>=24:
            z=maturity if cfg['relaxation']=='age' else r
            level=int(z>=40)+int(z>=60)+int(z>=80)
            shape=max(shape,[.025,.03,.04,.05][level]);gate=max(gate,[.03,.05,.08,.12][level]);overlap=max(overlap,[.03,.035,.04,.05][level])
    return shape,gate,overlap,maturity

def overlap_check(d,cfg,limit):
    if not len(d):return False,True,dict(p90=None,q80=None,trimmed=None,fraction3=None,fraction4=None,robust=None)
    p90=float(np.quantile(d,.9));q80=float(np.quantile(d,.8));trim=float(np.quantile(np.sort(d)[:max(1,int(.9*len(d)))],.9))
    f3=float(np.mean(d<=.03));f4=float(np.mean(d<=.04));rob=float(np.sqrt(np.mean(np.minimum(d,.06)**2)))
    mode=cfg['overlap'];ok={'q80':q80<=limit,'trimmed':trim<=limit,'fraction':f3>=.80 and f4>=.95,
        'soft':rob<=.028 and p90<=.04,'regularized':p90<=limit,'hysteresis':p90<=limit}.get(mode,p90<=limit)
    weak=not ok and cfg['tentative'] and p90<=max(.05,limit) and rob<=.04
    return bool(ok),bool(weak),dict(p90=p90,q80=q80,trimmed=trim,fraction3=f3,fraction4=f4,robust=rob)

def candidate_records(source,path,tt,template,cfg,C,B,old,curv,kappa,step):
    start=time.perf_counter();shape,gate,olimit,maturity=limits(cfg,path,C)
    localcfg=dict(cfg,anchor_gate=gate)
    oc,_=tt.fit(old,np.zeros(2),.12)
    fallback_anchor=(confirmed_blocks(path)[-1]['anchor_3d']-C)@B[:,1:]
    oa=oc[0]['anchor'] if oc else fallback_anchor
    first=tt.distances(old,oa)
    checklimit=.03 if cfg.get('strict_old') else olimit
    firstok,firstweak,firststats=overlap_check(first,cfg,checklimit)
    if cfg.get('strict_old'):firstweak=False
    stateful=cfg['tentative'] or cfg['gaps'] or cfg['beam']>1
    if not oc and (not stateful or cfg.get('strict_old')):return [],dict(reason='NO_CURRENT_OVERLAP',first_overlap=firststats,overlap_n=len(old),window=cfg['window']),[]
    if oc and not firstok and not firstweak:return [],dict(reason='OVERLAP_FIRST',first_overlap=firststats,overlap_n=len(old),window=cfg['window']),[]
    if cfg['history']=='fallback':levels=cfg['levels']
    elif cfg['history']=='range':levels=[[1,2,4,8,16][int(np.linalg.norm(C)>=20)+int(np.linalg.norm(C)>=40)+int(np.linalg.norm(C)>=60)+int(np.linalg.norm(C)>=80)]]
    else:levels=[cfg['frames']]
    if cfg['window_mode']=='support':windows=[8.,12.,16.,24.]
    elif cfg['window_mode']=='range':windows=[[8.,12.,16.,24.][int(np.linalg.norm(C)>=30)+int(np.linalg.norm(C)>=60)+int(np.linalg.norm(C)>=90)]]
    else:windows=[cfg['window']]
    trials=[];outputs=[];reason='NO_POINTS'
    for W in windows:
        for history in levels:
            pool=source.query(C,B,W,path['C'],path['B'],path['W'],oa,tt,localcfg,history)
            q=pool['uvw'].copy()
            if cfg['dewarp'] and np.linalg.norm(curv)>0:
                # 4m subwindows, local second-order short prior; canonical profile pooling.
                u=4*np.floor(q[:,0]/4)+2;offset=.5*u[:,None]**2*curv
                q[:,1:]-=offset@B[:,1:]
            nc,detail=tt.fit(q[:,1:],oa,gate)
            trial=dict(window=W,history=history,candidate_n=detail['candidate_n'],candidates=[],current_roi_n=int(np.sum(pool['age_frames']==0)))
            outputs=[]
            for rank,best in enumerate(nc[:max(3 if cfg['evidence']=='score' else 1,cfg['beam'])]):
                a=np.array(best['anchor']);alpha=cfg['alpha']
                if cfg['overlap']=='regularized':alpha=min(alpha,.5)
                a=oa+alpha*(a-oa)
                d=tt.distances(q[:,1:],a);inside=np.all((q[:,1:]-a>=tt.lo)&(q[:,1:]-a<=tt.hi),axis=1)
                take=inside&(d<=.02)
                xyz=pool['xyz'][take];ages=pool['age_frames'][take];keys=pool['keys'][take]
                N=len(np.unique(xyz,axis=0));sources=len(np.unique(ages));rings=len(np.unique(pool['ring'][take]));az=len(np.unique(pool['azimuth'][take]))
                cells=len(np.unique(np.floor(xyz/.01).astype(np.int64),axis=0))
                bins=len(np.unique(tt.tree.query(q[take,1:]-a)[1]//4)) if take.any() else 0
                span=float(np.ptp(q[take,0])) if take.any() else 0.
                displacement=float(np.ptp(pool['age_distance'][take])) if take.any() else 0.
                comp=next((x for x in nc if np.linalg.norm(x['anchor']-a)>.04 and x['support_unique']>=max(2,cfg['support'])),None)
                margin=best['score']-comp['score'] if comp else 1.
                secondok,secondweak,secondstats=overlap_check(tt.distances(old,a),cfg,checklimit)
                if cfg.get('strict_old'):secondweak=False
                shift=float(np.linalg.norm(a-oa));shape_error=float(np.mean(np.sort(d[inside])[:max(1,int(.7*inside.sum()))])) if inside.any() else .5
                disagree=trial['current_roi_n']>=12 and np.sum(take&(pool['age_frames']==0))==0 and np.mean(d[pool['age_frames']==0]>.08)>.8
                strong=(N>=3 and best['support_unique']>=3 and shape_error<=shape and firstok and secondok and margin>=.025 and sources>=cfg['consensus'])
                weak=(N>=cfg['support'] and N>=1 and shape_error<=max(shape,.04) and shift<=gate*1.42 and margin>=.01 and (secondok or secondweak or (not len(old) and not cfg.get('strict_old'))))
                if cfg['support']==2 and N==2:weak=weak and (az>=2 or rings>=2) and cells>=2 and span>=.04
                if cfg['partial']!='full':strong=False
                if N==1:strong=False
                if cfg['relaxation']!='none' and (shape>cfg['shape_limit'] or olimit>.03):strong=False
                if len(old) and (secondstats['p90']>.08 or firststats['p90']>.08):weak=False;strong=False
                if cfg['current_veto'] and disagree:weak=False;strong=False
                if cfg['evidence']=='diversity':weak=weak and (sources>=2 or (rings>=2 and az>=2))
                elif cfg['evidence']=='cells':weak=weak and cells>=3 and bins>=2
                elif cfg['evidence']=='balanced':weak=weak and ((sources>=2 and displacement>=.25) or N>=4)
                if sources<cfg['consensus']:strong=False
                terms=dict(shape=float(np.exp(-shape_error/.02)),position=float(np.exp(-shift/max(gate,.01))),
                    tangent=float(np.exp(-np.hypot(*[path['angles'].get(x,0) for x in ('delta_yaw','delta_pitch')])/3)),
                    curvature=float(np.exp(-kappa/.01)),sources=min(1.,sources/3),template=min(1.,bins/4),competitor=max(0.,.05-margin)/.05)
                if cfg.get('curvature_change') and take.any():
                    aa=[b['anchor_3d'] for b in confirmed_blocks(path)[-3:]]
                    if len(aa)>=3:
                        proposed=C+np.array([np.median(q[take,0]),*a])@B.T
                        aa=np.array(aa+[proposed]);segments=np.diff(aa,axis=0);lengths=np.linalg.norm(segments,axis=1)
                        if np.all(lengths>.5):
                            directions=segments/lengths[:,None]
                            old_curv=(directions[1]-directions[0])/((lengths[1]+lengths[0])/2)
                            new_curv=(directions[2]-directions[1])/((lengths[2]+lengths[1])/2)
                            change=float(np.linalg.norm(new_curv-old_curv));terms['curvature_change_per_m']=change
                            terms['curvature']=float(np.exp(-change/.005))
                score=(terms['shape']+terms['position']+terms['tangent']+terms['curvature']+terms['sources']+terms['template'])/6-terms['competitor']*.3
                if cfg.get('score_balance'):
                    weight=cfg['score_balance'];score=(terms['shape']+terms['position']+weight*(terms['tangent']+terms['curvature'])+terms['sources']+terms['template'])/(4+2*weight)-terms['competitor']*.3
                if cfg.get('ring_score',0) and np.linalg.norm(C)>=20 and take.any():
                    available_rings=pool['ring'][pool['ring']>=0];supported_rings=pool['ring'][take&(pool['ring']>=0)]
                    rank_score=1-float(np.mean(available_rings<=np.median(supported_rings))) if len(available_rings) and len(supported_rings) else .5
                    terms['channel_rank']=rank_score;score+=cfg['ring_score']*(rank_score-.5)
                if cfg['evidence']=='score':weak=weak and score>=.60/cfg['score_weight']
                state=CONFIRMED if strong else TENTATIVE if cfg['tentative'] and weak else 0
                why='CONFIRMED' if strong else 'TENTATIVE' if state else 'TOO_FEW_SUPPORT' if N<cfg['support'] else 'CURRENT_VETO' if cfg['current_veto'] and disagree else 'COMPETITOR' if margin<.025 else 'OVERLAP_SECOND' if not secondok else 'SHAPE_MISMATCH'
                cand=dict(rank=rank,anchor=a,shape_residual=shape_error,support_unique=N,template_support_unique=best['support_unique'],template_coverage=best['coverage'],
                  source_count=sources,ring_count=rings,azimuth_count=az,spatial_cells=cells,template_regions=bins,longitudinal_span=span,source_displacement_span=displacement,
                  competitor=comp,competitor_margin=margin,first_overlap=firststats,second_overlap=secondstats,current_disagrees=bool(disagree),anchor_shift=shift,
                  continuity_score=score,score_terms=terms,state=state,reason=why,track_length=maturity,history=history,window=W)
                trial['candidates'].append(cand)
                if state and len(keys):
                    aq=np.array([np.median(q[take,0]),*a]);anchor3=C+aq@B.T
                    if cfg['dewarp']:anchor3+=.5*aq[0]**2*curv
                    block=dict(step=step,keys=keys,xyz=xyz,age=ages,uvw=q[take],residual=d[take],state=state,original_state=state,
                      confidence=float(max(0,score)*(1-np.exp(-N/8))),continuity_score=score,template_score=best['score'],track_prior_score=terms['position'],
                      anchor=a,anchor_3d=anchor3,window=W,history=history,meta=cand)
                    outputs.append(block)
            trials.append(trial)
            if cfg['evidence']=='score' or cfg.get('ring_score',0):outputs.sort(key=lambda b:b['continuity_score'],reverse=True)
            if outputs:
                # A successful T-only attempt is returned before any history is even read.
                if any(b['state']==CONFIRMED for b in outputs) or history==levels[-1]:break
            reason=trial['candidates'][0]['reason'] if trial['candidates'] else 'TOO_FEW_SUPPORT' if detail['candidate_n'] else 'NO_POINTS'
            if cfg['history']=='fallback' and cfg['frames']<=8 and not outputs and reason not in ('TOO_FEW_SUPPORT','NO_POINTS','SHAPE_MISMATCH'):
                break
        if outputs:break
    info=dict(reason='' if outputs else reason,first_overlap=firststats,overlap_n=len(old),overlap_anchor=oa,
        total_ms=(time.perf_counter()-start)*1000,window=windows[-1] if not outputs else outputs[0]['window'])
    return outputs,info,trials

def can_promote(path,cfg,newblock):
    pending=[b for b in path['blocks'] if b['state']==TENTATIVE]
    if not pending:return False,'NO_PENDING'
    if newblock['state']!=CONFIRMED and len(pending)<cfg['lookahead']:return False,'WAIT_SPATIAL_EVIDENCE'
    allblocks=confirmed_blocks(path)[-3:]+pending
    # The candidate remains an actual observation: no synthetic point is introduced.
    a=np.array([b['anchor_3d'] for b in allblocks]);u=(a-a[0])@path['B'][:,0]
    if len(a)<3 or np.ptp(u)<2:return False,'SHORT_CONNECTION'
    design=np.c_[np.ones(len(u)),u,u*u]
    fit=design@np.linalg.lstsq(design,a,rcond=None)[0]
    residual=float(np.max(np.linalg.norm(a-fit,axis=1)))
    independent=np.unique(np.concatenate([b['xyz'] for b in pending]+[newblock['xyz']]),axis=0)
    if len(independent)<3:return False,'INSUFFICIENT_JOINT_POSITIONS'
    if residual>.05:return False,'JOINT_CURVE_INCONSISTENT'
    if cfg['backward']:
        # Far observed anchors must reconnect to the latest confirmed anchor.
        confirmed=confirmed_blocks(path)
        if len(pending)>=2 and confirmed:
            tangent=unit(pending[-1]['anchor_3d']-pending[-2]['anchor_3d'])
            d=confirmed[-1]['anchor_3d']-pending[-1]['anchor_3d']
            back=np.linalg.norm(d-(d@tangent)*tangent)
            if back>.08:return False,'BACKWARD_RECONNECT_FAILED'
    return True,'JOINT_SPATIAL_CONFIRMATION'

def clone_path(p):
    return dict(p,blocks=[dict(b) for b in p['blocks']],steps=[dict(s) for s in p['steps']])

def infer(source,seed,template,cfg):
    begin=time.perf_counter();source.reset_audit()
    if seed['status']!='AVAILABLE':
        c=source.materialize()
        return dict(status=seed['status'],reason=seed.get('reason',''),seed=seed,config=cfg,steps=[],hypotheses=[],indices=np.empty(0,dtype=np.int64),point_step=np.empty(0,dtype=np.uint16),point_state=np.empty(0,dtype=np.uint8),total_ms=0.),c
    B=np.asarray(seed['basis']);C=np.asarray(seed['anchor']);ids=np.asarray(seed['indices'],dtype=int)
    current=source.current;xyz=current['xyz'][ids];uvw=(xyz-C)@B
    seedblock=dict(step=0,keys=point_keys(current)[ids],xyz=xyz,age=np.zeros(len(ids),dtype=int),uvw=uvw,residual=np.full(len(ids),np.nan),
      state=CONFIRMED,original_state=CONFIRMED,confidence=seed['confidence'],continuity_score=1.,template_score=1.,track_prior_score=1.,anchor=np.zeros(2),anchor_3d=C.copy())
    path=dict(B=B,C=C,W=float(seed.get('seed_end',8)),blocks=[seedblock],steps=[],pending=0,gap_count=0,score=0.,reason='RANGE_LIMIT',
      search_range=float(np.linalg.norm(xyz,axis=1).max()),angles={})
    active=[path];finished=[];hypotheses=[]
    fitcfg=dict(CFG,min_support=cfg['support'],shape_limit=cfg['shape_limit']);tt=TemplateTracker(template_part(template,cfg['partial']),seed['side'],fitcfg)
    for k in range(1,cfg['max_steps']+1):
        branches=[]
        for parent_id,p in enumerate(active):
            st=time.perf_counter()
            try:Cnew,Bnew,old,curv,kappa,ang=frame_for(p,cfg)
            except ValueError as e:
                dead=clone_path(p);dead['reason']=str(e);finished.append(dead);continue
            p=dict(p,angles=ang)
            rec=dict(step_index=k,status='STOP',state='STOP',plane_origin=Cnew,basis=Bnew,previous_origin=p['C'],previous_basis=p['B'],
              window_start_s=4*k,window_end_s=4*k+cfg['window'],new_support_n=0,**ang)
            blocks,info,trials=candidate_records(source,p,tt,template,cfg,Cnew,Bnew,old,curv,kappa,k)
            rec.update(info);rec['trials']=trials;rec['total_ms']=(time.perf_counter()-st)*1000
            search=float(np.linalg.norm(Cnew+Bnew[:,0]*info['window']))
            for rank,b in enumerate(blocks[:cfg['beam']]):
                child=clone_path(p);child['B']=Bnew;child['C']=Cnew;child['W']=b['window'];child['blocks'].append(b);child['search_range']=max(search,p['search_range'])
                rr=dict(rec,state='CONFIRMED' if b['state']==CONFIRMED else 'TENTATIVE',status='ACCEPTED' if b['state']==CONFIRMED else 'TENTATIVE',
                  anchor_3d=b['anchor_3d'],anchor_v=float(b['anchor'][0]),anchor_w=float(b['anchor'][1]),new_support_n=len(b['keys']),
                  range_near=float(np.linalg.norm(b['xyz'],axis=1).min()),range_far=float(np.linalg.norm(b['xyz'],axis=1).max()),
                  confidence=b['confidence'],template_score=b['template_score'],continuity_score=b['continuity_score'],selected_candidate_rank=rank,
                  failure_reason='',history=b['history'],window=b['window'],candidate=b['meta'],original_state='CONFIRMED' if b['state']==CONFIRMED else 'TENTATIVE')
                child['steps'].append(rr)
                pending=sum(z['state']==TENTATIVE for z in child['blocks']);child['pending']=pending
                if pending:
                    promote,why=can_promote(child,cfg,b);rr['promotion_test']=why
                    if promote:
                        for z in child['blocks']:
                            if z['state']==TENTATIVE:z['state']=CONFIRMED
                        for z in child['steps']:
                            if z['state']=='TENTATIVE':z.update(state='CONFIRMED',status='ACCEPTED',promoted_at=k)
                        child['pending']=0
                    elif b['state']==CONFIRMED:
                        # Do not silently confirm through an incompatible pending segment.
                        b['state']=TENTATIVE;rr.update(state='TENTATIVE',status='TENTATIVE');child['pending']+=1
                if child['pending']>max(1,cfg['lookahead']):
                    child['reason']='TENTATIVE_UNCONFIRMED';finished.append(child);continue
                child['gap_count']=0
                if b['state']==CONFIRMED and child['pending']==0:
                    for z in child['steps']:
                        if z['state']=='GAP' and not z.get('bridged'):z.update(bridged=True,reconnected_at=k)
                penalty=.4 if b['state']==TENTATIVE else 0.
                child['score']+=b['continuity_score']-.20*kappa/.01-penalty
                hypotheses.append(dict(step=k,parent=parent_id,rank=rank,path_score=child['score'],state=rr['state'],range=rr['range_far'],candidate=b['meta'],
                    plane_origin=Cnew,basis=Bnew,anchor_3d=b['anchor_3d'],support_keys=b['keys']))
                if rr['range_far']>=cfg['max_range']:child['reason']='RANGE_LIMIT';finished.append(child)
                else:branches.append(child)
            if not blocks or (cfg['beam']>1 and cfg['gaps']):
                gross=(info.get('first_overlap',{}).get('p90') or 0)>.08
                gap_allowed=p['gap_count']<cfg['gaps'] and not gross and info['reason'] not in ('CURRENT_VETO','COMPETITOR','ORIENTATION_JUMP')
                if gap_allowed:
                    child=clone_path(p);child.update(B=Bnew,C=Cnew,W=8.,gap_count=p['gap_count']+1,score=p['score']-.6,search_range=max(search,p['search_range']))
                    child['steps'].append(dict(rec,state='GAP',status='GAP',observed=False,unobserved_length=4.,bridged=False,failure_reason=info['reason']))
                    branches.append(child);hypotheses.append(dict(step=k,parent=parent_id,rank=-1,path_score=child['score'],state='GAP',range=search))
                elif not blocks:
                    child=clone_path(p);child['steps'].append(dict(rec,failure_reason=info['reason']));child.update(reason=info['reason'],search_range=max(search,p['search_range']));finished.append(child)
        if not branches:break
        # Equivalent end positions collapse, retaining the strongest path and exact provenance.
        branches.sort(key=lambda p:p['score'],reverse=True);active=[];positions=[]
        for p in branches:
            end=p['blocks'][-1]['anchor_3d']
            if any(np.linalg.norm(end-a)<.008 and p['gap_count']==g for a,g in positions):continue
            active.append(p);positions.append((end,p['gap_count']))
            if len(active)>=cfg['beam']:break
    else:finished+=active
    if not finished:finished=active or [path]
    best=max(finished,key=lambda p:(p['score'],max(np.linalg.norm(b['xyz'],axis=1).max() for b in confirmed_blocks(p))))
    c=source.materialize();order=np.argsort(c['keys']);sortedkeys=c['keys'][order]
    allkeys=np.concatenate([b['keys'] for b in best['blocks']]);ix=order[np.searchsorted(sortedkeys,allkeys)]
    assert np.array_equal(c['keys'][ix],allkeys)
    assert len(np.unique(allkeys))==len(allkeys),'A source point was accepted twice'
    blocks=best['blocks'];arr=lambda f:np.concatenate([b[f] for b in blocks]);rep=lambda f,dtype=float:np.concatenate([np.full(len(b['keys']),b[f],dtype=dtype) for b in blocks])
    consecutive=0;max_gap=0
    for s in best['steps']:
        consecutive=consecutive+4 if s.get('state')=='GAP' else 0;max_gap=max(max_gap,consecutive)
    p=dict(status='STOPPED',reason=best['reason'],seed=seed,config=dict(CFG,**cfg),steps=best['steps'],hypotheses=hypotheses,
      indices=ix,point_step=rep('step',np.uint16),point_state=rep('state',np.uint8),local_uvw=arr('uvw'),template_residual=arr('residual'),
      confidence=rep('confidence'),continuity_score=rep('continuity_score'),template_score=rep('template_score'),track_prior_score=rep('track_prior_score'),
      point_anchor=np.concatenate([np.repeat(b['anchor'][None,:],len(b['keys']),axis=0) for b in blocks]),
      s_from_seed=np.concatenate([4*b['step']+b['uvw'][:,0] for b in blocks]),curve=np.array([b['anchor_3d'] for b in blocks if b['state']==CONFIRMED]),
      max_search_hypothesis_range=max(p['search_range'] for p in finished),max_selected_search_range=best['search_range'],
      max_tentative_range=max([float(np.linalg.norm(b['xyz'],axis=1).max()) for b in blocks if b['state']==TENTATIVE]+[0]),
      max_confirmed_observed_range=max(float(np.linalg.norm(b['xyz'],axis=1).max()) for b in blocks if b['state']==CONFIRMED),
      max_unobserved_length=max_gap,path_score=best['score'],total_ms=(time.perf_counter()-begin)*1000)
    return p,c
