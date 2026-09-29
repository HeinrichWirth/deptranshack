"""Post-prediction diagnostics: same future reference, never detector features."""
from fusion_common import *
from fusion import CausalSource,assemble
from assess import check_marker,reference_readonly
from evaluation import evaluate_prediction,make_curve,voxel
from tracker import _engine,TemplateTracker
from contact_rail_step2 import ContactRailDetector
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def frame_oracle(run,i,ref,info,seed):
    target=OUT/'diagnostics/oracle_frames'/(key(run,i)+'.npz');target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():
        with np.load(target) as z:curve=z['curve'];basis=z['basis']
    else:
        B=np.asarray(seed['basis']);near=ref['current'][(ref['current']@B[:,0]>=0)&(ref['current']@B[:,0]<=8)]
        mm=frames(run);P=np.asarray(mm[i]['lidar_pose_in_folder']);poses=np.array([r['lidar_pose_in_folder'] for r in mm[i:info['future_end']+1]])[:,:3,3];poses=(poses-P[:3,3])@P[:3,:3]
        curve,basis,_=make_curve(voxel(np.r_[near,ref['future']],.005),poses,B,seed['side']);np.savez_compressed(target,curve=curve,basis=basis)
    def orientation(C,B,tail):
        if len(curve)<3:raise ValueError('GT_UNAVAILABLE_AHEAD')
        distance=np.linalg.norm(curve-C,axis=1);j=distance.argmin()
        if distance[j]>2:raise ValueError('GT_UNAVAILABLE_AHEAD')
        G=basis[j].copy()
        if G[:,0]@B[:,0]<0:G[:,0]*=-1;G[:,1]*=-1
        return G
    return orientation

def local_counts(c,st,ref,near_ids=None):
    if st.get('plane_origin') is None:return dict(returns=0,unique=0,future_points=0,near_ids=np.empty(0,dtype=int))
    C=np.asarray(st['plane_origin']);B=np.asarray(st['basis']);f=(ref['future']-C)@B;a=np.asarray(st.get('overlap_anchor',[0,0]))
    if near_ids is None:
        d=cKDTree(ref['future']).query(c['xyz'],distance_upper_bound=np.nextafter(.05,np.inf))[0] if len(ref['future']) else np.full(len(c['xyz']),np.inf)
        near_ids=np.flatnonzero(d<=.05)
    q=(c['xyz'][near_ids]-C)@B;near=near_ids[(q[:,0]>=4)&(q[:,0]<=8)]
    return dict(returns=len(near),unique=len(np.unique(c['xyz'][near],axis=0)),future_points=int(np.sum((f[:,0]>=4)&(f[:,0]<=8)&(np.linalg.norm(f[:,1:]-a,axis=1)<.4))),near_ids=near)

def support_counts(c,ids):
    xyz=c['xyz'][ids];age=c['age_frames'][ids]
    unique=len(np.unique(xyz,axis=0))
    return dict(raw_returns=len(ids),unique_xyz=unique,distinct_rings=len(np.unique(c['ring'][ids])),distinct_azimuth_samples=len(np.unique(c['azimuth'][ids])),distinct_frame_azimuth=len(np.unique(np.column_stack((age,c['azimuth'][ids])),axis=0)),past_returns=int(np.sum(age>0)))

def registration(c,base,template):
    st=next((s for s in reversed(base['steps']) if s['status']=='ACCEPTED' and s.get('overlap_anchor') is not None),None)
    if st is None:return []
    B=np.asarray(st['basis']);C=np.asarray(st['plane_origin']);a=np.asarray(st['overlap_anchor']);tt=TemplateTracker(template,base['seed']['side'],CFG)
    q=(c['xyz']-C)@B;region=(q[:,0]>=0)&(q[:,0]<=4)&np.all((q[:,1:]-a>=tt.lo-.08)&(q[:,1:]-a<=tt.hi+.08),axis=1);rows=[]
    for age in np.unique(c['age_frames']):
        ids=np.flatnonzero(region&(c['age_frames']==age));p=q[ids,1:]
        if len(p)<3:continue
        d=tt.distances(p,a);cc,_=tt.fit(p,a,.05);delta=cc[0]['anchor']-a if cc else np.array([np.nan,np.nan])
        rows.append(dict(age_frames=int(age),age_distance=float(c['age_distance'][ids[0]]),points=len(p),residual_median=float(np.median(d)),residual_p90=float(np.percentile(d,90)),residual_p95=float(np.percentile(d,95)),dv=float(delta[0]),dw=float(delta[1]),profile_v_q95q05=float(np.quantile(p[:,0],.95)-np.quantile(p[:,0],.05)),profile_w_q95q05=float(np.quantile(p[:,1],.95)-np.quantile(p[:,1],.05)),baseline_step=st['step_index']))
    ids=np.flatnonzero(region)
    if len(ids):
        d=tt.distances(q[ids,1:],a);rows.append(dict(age_frames=-1,points=len(ids),residual_median=float(np.median(d)),residual_p90=float(np.percentile(d,90)),residual_p95=float(np.percentile(d,95)),profile_v_q95q05=float(np.ptp(np.quantile(q[ids,1],[.05,.95]))),profile_w_q95q05=float(np.ptp(np.quantile(q[ids,2],[.05,.95]))),baseline_step=st['step_index']))
    return rows

def one(task):
    run,i,names=task;target=OUT/'diagnostics'/key(run,i);target.mkdir(parents=True,exist_ok=True)
    if (target/'COMPLETE.json').exists():return run,i,'cached'
    seed=seed_from_cache(run,i)
    if seed['status']!='AVAILABLE':return run,i,'unavailable'
    for name in names:check_marker(OUT/'heldout'/name/key(run,i))
    base=load_prediction(BASE_OUT/'heldout'/key(run,i));baseline_row=load(BASE_OUT/'failures'/(key(run,i)+'.json'))['summary'];ref,info=reference_readonly(run,i)
    oracle=frame_oracle(run,i,ref,info,seed);template=ContactRailDetector().template;gtree=cKDTree(ref['future']) if len(ref['future']) else None
    contributions=[];density=[];blur=[];results=[];ranges=[];packets=[]
    source=CausalSource(run,i);read_original=source.read;read_cache={};near_cache={};oracle_cache={};blur_cache={}
    def cached_read(k):
        if k not in read_cache:
            read_cache[k]=read_original(k)
            pp=read_cache[k][0]['xyz']
            d=gtree.query(pp,distance_upper_bound=np.nextafter(.05,np.inf))[0] if gtree else np.full(len(pp),np.inf)
            near_cache[k]=d<=.05
        return read_cache[k]
    source.read=cached_read
    for name in names:
        folder=OUT/'heldout'/name/key(run,i);pred=load_prediction(folder);e=load(folder/'evaluation.json');r=e['summary'];config=pred['fusion']['config']
        assert config.get('representation','RAW_CONCAT')=='RAW_CONCAT' and not config.get('alignment_bound')
        c,_=assemble(source,config,seed);xyz=c['xyz'];signature=(tuple(c['meta']['source_frames']),pred['config']['method'])
        near_ids=np.flatnonzero(np.concatenate([near_cache[k] for k in c['meta']['source_frames']]))
        common=dict(run=run,start_frame=i,variant=name)
        # Exact STEP5 oracle-frame construction, now applied to this causal input.
        opath=folder/'oracle_frame_evaluation.json'
        if opath.exists():oe=load(opath)
        else:
            if signature in oracle_cache:oe=oracle_cache[signature]
            elif name=='F0':oe=load(BASE_OUT/'oracles/ORACLE_FRAME'/key(run,i)/'evaluation.json')
            else:
                if config.get('representation')=='FRAME_BALANCED':
                    from weighted_tracker import _engine as weighted_engine
                    op=weighted_engine(xyz,pred['seed'],template,pred['config'],oracle,c['source_frame'],c['age_distance'],config.get('tau'))
                else:op=_engine(xyz,pred['seed'],template,pred['config'],oracle)
                op.update(run=run,start_frame=i);orr,os,ob,pt=evaluate_prediction(op,xyz,ref,info);oe=dict(summary=orr,steps=os)
            save(opath,oe)
        oracle_cache[signature]=oe
        terminal=pred['steps'][-1] if pred['steps'] else {};obs=local_counts(c,terminal,ref,near_ids);baseline_next=local_counts(c,base['steps'][-1] if base['steps'] else {},ref,near_ids)
        reach=r.get('continuous_reach_5cm') or 0;oracle_reach=oe['summary'].get('continuous_reach_5cm') or 0
        if not r.get('evaluation_eligible'):reason='GT_UNAVAILABLE_AHEAD'
        elif r.get('wrong_structure_review'):reason='CR_GT_GAP'
        elif r.get('wrong_structure_suspected'):reason='WRONG_STRUCTURE_SUSPECTED'
        elif obs['future_points']<3:reason='RUN_END' if info['horizon_reason']=='RUN_END' else 'CR_GT_GAP'
        elif oracle_reach>reach+4:reason='ORIENTATION_FAILURE'
        elif obs['unique']<3:reason='SENSOR_SPARSITY'
        else:reason='OBSERVATION_TEMPLATE_LIMIT'
        if name=='F0':reason=baseline_row['terminal_evaluation_reason']
        dreach=reach-(baseline_row.get('continuous_reach_5cm') or 0)
        r.update(terminal_evaluation_reason=reason,oracle_frame_reach=oracle_reach,terminal_returns=obs['returns'],terminal_unique=obs['unique'],terminal_future_points=obs['future_points'],baseline_terminal_fused_returns=baseline_next['returns'],baseline_terminal_fused_unique=baseline_next['unique'],delta_reach=dreach,
            baseline_stop=baseline_row['stop_reason'],baseline_diagnosis=baseline_row['terminal_evaluation_reason'],baseline_terminal_returns=baseline_row.get('terminal_observable_returns'),baseline_terminal_future_points=baseline_row.get('terminal_future_points'),actual_frames=c['meta']['actual_frames'],input_points=len(xyz),input_bytes=c['meta']['input_bytes'])
        # New range must be supported by historical real measurements and future GT.
        pp=xyz[pred['indices']];past=c['age_frames'][pred['indices']]>0;rrange=np.linalg.norm(pp,axis=1)
        pd=gtree.query(pp)[0] if gtree else np.full(len(pp),np.inf)
        past_extension=past&(rrange>(baseline_row.get('max_accepted_range') or 0))&(pd<=.05)&(pred['point_step']>0)
        r['confirmed_past_extension_points']=int(past_extension.sum());r['true_range_extension']=bool(dreach>4 and past_extension.any());results.append(r)
        for st in pred['steps']:
            if st['status']!='ACCEPTED':continue
            ids=np.asarray(st['support_indices'],dtype=int);current=ids[c['age_frames'][ids]==0];pastids=ids[c['age_frames'][ids]>0]
            u=np.unique(np.floor(xyz[pastids]/.01).astype(np.int64),axis=0);v=np.unique(np.floor(xyz[current]/.01).astype(np.int64),axis=0)
            currentkeys={tuple(q) for q in v};pastonly=sum(tuple(q) not in currentkeys for q in u)
            tt=TemplateTracker(template,seed['side'],CFG);q=(xyz[ids]-np.asarray(st['plane_origin']))@np.asarray(st['basis']);node=tt.tree.query(q[:,1:]-np.array([st['anchor_v'],st['anchor_w']]))[1]
            current_nodes=set(node[c['age_frames'][ids]==0]);past_nodes=set(node[c['age_frames'][ids]>0]);new_nodes=len(past_nodes-current_nodes)
            for age in np.unique(c['age_frames'][ids]):
                take=ids[c['age_frames'][ids]==age];contributions.append(dict(**common,step_index=st['step_index'],age_frames=int(age),support=len(take),fraction=len(take)/len(ids),past_only_voxels_1cm=pastonly,unique_template_samples=len(set(node[c['age_frames'][ids]==age])),past_only_template_samples=new_nodes,template_samples_total=len(tt.template),range_far=st['range_far'],**support_counts(c,take)))
        # Near-GT density is evaluated AFTER prediction, never used for acceptance.
        norm=np.linalg.norm(xyz,axis=1);near_ids=near_ids[norm[near_ids]<=75]
        for lo,hi in zip((0,10,20,30,40,50,60),(10,20,30,40,50,60,75)):
            ids=near_ids[(norm[near_ids]>=lo)&(norm[near_ids]<hi)];grange=np.linalg.norm(ref['future'],axis=1);gn=int(np.sum((grange>=lo)&(grange<hi)));density.append(dict(**common,lo=lo,hi=hi,reference_available=gtree is not None,reference_points_in_bin=gn,**support_counts(c,ids)))
        bsig=tuple(c['meta']['source_frames'])
        if bsig not in blur_cache:blur_cache[bsig]=registration(c,base,template)
        blur.extend([dict(**common,**b) for b in blur_cache[bsig]])
        packet=dict(**common,summary=r,reference=info,failed_step=terminal,last_successful_step=next((s for s in reversed(pred['steps']) if s['status']=='ACCEPTED'),None),baseline_failed_step=base['steps'][-1] if base['steps'] else {},baseline_next_counts={k:v for k,v in baseline_next.items() if k!='near_ids'},oracle_frame_summary=oe['summary'])
        if terminal.get('plane_origin') is not None:
            B=np.asarray(terminal['basis']);C=np.asarray(terminal['plane_origin']);q=(xyz-C)@B;a=np.asarray(terminal.get('overlap_anchor',[0,0]));tt=TemplateTracker(template,seed['side'],CFG)
            ids=np.flatnonzero((q[:,0]>=0)&(q[:,0]<=8)&np.all((q[:,1:]-a>=tt.lo-.12)&(q[:,1:]-a<=tt.hi+.12),axis=1))
            current=ids[c['age_frames'][ids]==0];pastids=ids[c['age_frames'][ids]>0];distance=cKDTree(xyz[current]).query(xyz[pastids])[0] if len(current) else np.full(len(pastids),np.inf)
            ghost=pastids[distance>.03];r['past_only_terminal_roi']=len(ghost);r['terminal_roi_points']=len(ids)
            packet['candidate_points']={k:c[k][ids] for k in ('xyz','source_frame','source_row','source_point_index','age_frames')};packet['candidate_gt_distance']=gtree.query(xyz[ids])[0] if gtree else None;packet['past_only_candidate_rows']=ghost
        path=OUT/'failures'/name/(key(run,i)+'.json');save(path,packet)
    save(target/'results.json',results);csv_write(target/'source_contribution.csv',contributions);csv_write(target/'fusion_density.csv',density);csv_write(target/'registration_blur.csv',blur)
    save(target/'COMPLETE.json',dict(variants=names,time_ns=time.time_ns()));return run,i,'done'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lock=load(OUT/'research_lock.json');names=[n for n in lock['heldout_variants'] if n!='FUSED_BOOTSTRAP'];rr=load(OUT/'audit/cohort.json')['heldout'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,(r['run'],r['frame'],names)) for r in rr]
        for j,f in enumerate(as_completed(ff)):
            v=f.result()
            if j%20==0 or j==len(ff)-1:print('DIAGNOSE',j+1,len(ff),v,'elapsed',round(time.perf_counter()-t,1),flush=True)
if __name__=='__main__':main()
