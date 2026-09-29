"""Final tables, matched oracle comparison, failure corpus, gallery case selection."""
from lc_common import *
from study_evaluation import BINS

def npz(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def main():
    freeze=load(OUT/'research_freeze.json');best=freeze['best'];protocol=load(OUT/'protocol.json');assert (OUT/'phase_e/EVALUATION_COMPLETE.json').exists();allstarts=[];allstations=[];methodmetrics=[]
    for group in ('phase_d','phase_e'):
        allstarts +=[dict(r,group=group) for r in load(OUT/group/'per_start.json')];allstations +=[dict(r,group=group) for r in load(OUT/group/'per_station.json')];methodmetrics +=[dict(r,group=group) for r in load(OUT/group/'range_metrics.json')]
    for m in methodmetrics:
        rr=[r for r in allstations if r['group']==m['group'] and r['method']==m['method'] and r['valid'] and m['lo']<=r['range_m']<m['hi']]
        for axis in ('v','w'):m['far_median_fullwidth95_'+axis]=float(np.median([r['far_fullwidth95_'+axis] for r in rr])) if rr else None
    write_csv(OUT/'method_summary.csv',methodmetrics);write_csv(OUT/'downstream_step6.csv',methodmetrics);write_csv(OUT/'per_start.csv',allstarts);write_csv(OUT/'uncertainty_calibration.csv',[{k:v for k,v in r.items() if k in ('method','group','lo','hi','starts','stations','joint95','joint99','cr_coverage95','cr_coverage99','far_fullwidth95_v','far_fullwidth95_w','far_median_fullwidth95_v','far_median_fullwidth95_w')} for r in methodmetrics])
    phy=[dict(group=r['group'],run=r['run'],frame=r['frame'],method=r['method'],**{k:r.get(k) for k in ('radius_min','radius_p1','radius_p5','curvature_violation_fraction','physical_violation_fraction','heading_jump_1m_deg','heading_jump_2m_deg','heading_jump_4m_deg','heading_jump_8m_deg','max_grade','max_vertical_curvature','max_curvature_derivative','max_tangent_change_deg','self_intersection','backward_count','station_monotonic','overshoot_p50','overshoot_p95','overshoot_max','S_phys')}) for r in allstarts if r['status']=='AVAILABLE'];write_csv(OUT/'curve_physics.csv',phy);write_csv(OUT/'curvature_diagnostics.csv',phy);write_csv(OUT/'vertical_diagnostics.csv',phy)
    screening=load(OUT/'phase_a/per_start.json');write_csv(OUT/'interpolator_ablation.csv',[r for r in screening if not next(c for c in protocol['candidates'] if c['name']==r['method'])['use_profile']]);write_csv(OUT/'profile_ablation.csv',[r for r in screening if next(c for c in protocol['candidates'] if c['name']==r['method'])['use_profile']]);save(OUT/'method_selection.json',dict(phase_a=load(OUT/'selection/phase_a.json'),phase_b=load(OUT/'selection/phase_b.json')))
    oldbad=set(load(OUT/'baseline_freeze.json')['expected']['gt20cm_cases']);cases=[];perstation=[];oracle_rows=[];decomposition=[];provenance_jobs=[]
    for r in allstarts:
        if r['method']!=best or r['status']!='AVAILABLE':continue
        run,i=r['run'],r['frame'];group=r['group'];casekey=key(run,i);folder=OUT/group/best/casekey;detail=load(folder/'details.json');p=npz(folder/'prediction.npz');curve=npz(folder/'curve.npz');e=npz(folder/'evaluation.npz');oldgroup='phase_b' if group=='phase_d' else 'phase_e';op=npz(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/casekey/'prediction.npz');oe=npz(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/casekey/'evaluation.npz');base=next(v for v in allstarts if v['group']==group and v['method']=='C0_CURRENT' and v['run']==run and v['frame']==i);profile=detail.get('profiles',[]);profile_s=np.array([v['station'] for v in profile]);take=e['valid']&(e['station']>=8);bad=bool(np.any(e['error'][take,1]>.2));worst=int(np.nanargmax(np.where(take,e['error'][:,1],np.nan))) if take.any() else 0
        for j,station in enumerate(e['station']):
            k=int(np.argmin(abs(curve['s']-station)));prof=profile[int(np.argmin(abs(profile_s-station)))] if profile else {};B=p['B'][min(k,len(p['B'])-1)];cv=B[:,1:].T@curve['covariance'][k]@B[:,1:];aa=curve['anchor_xyz'][int(np.argmin(abs(curve['anchor_s']-station)))];oldc=op['C'][j]
            row=dict(run=run,start=i,group=group,station=float(station),range_m=float(np.linalg.norm(oldc)),raw_point_count=prof.get('n'),source_count=prof.get('sources'),visibility_mode_est=prof.get('visibility','NO_PROFILE_MODEL'),template_coverage=prof.get('coverage'),latent_anchor_x=prof.get('anchor_xyz',[None]*3)[0],latent_anchor_y=prof.get('anchor_xyz',[None]*3)[1],latent_anchor_z=prof.get('anchor_xyz',[None]*3)[2],sigma_v=float(np.sqrt(cv[0,0])),sigma_w=float(np.sqrt(cv[1,1])),tangent_x=float(B[0,0]),tangent_y=float(B[1,0]),tangent_z=float(B[2,0]),curvature_h=float(curve['curvature_h'][k]),grade=float(curve['grade'][k]),curvature_v=float(curve['curvature_v'][k]),physical_flags=bool(curve['physical_flags'][k]),old_anchor_x=float(aa[0]),old_anchor_y=float(aa[1]),old_anchor_z=float(aa[2]),old_curve_x=float(oldc[0]),old_curve_y=float(oldc[1]),old_curve_z=float(oldc[2]),new_curve_x=float(p['C'][k,0]),new_curve_y=float(p['C'][k,1]),new_curve_z=float(p['C'][k,2]),future_CR_error_eval=float(np.linalg.norm(e['cr_error'][j])),future_CR_lateral_eval=float(e['cr_error'][j,0]),future_CR_vertical_eval=float(e['cr_error'][j,1]),rail_far_error_old=float(oe['error'][j,1]) if oe['valid'][j] else None,rail_far_error_new=float(e['error'][j,1]) if e['valid'][j] else None,common_plane_available=bool(e['alignment_available'][j]),rail_GT_available=bool(e['valid'][j]),anchor_status=prof.get('state'),mode_count=len(prof.get('modes',[])),residual=prof.get('residual'),sigma_tangent=float(curve['tangent_sigma'][k]))
            # Observation status remains diagnostic: no silent deletion.
            if prof and np.linalg.norm(np.asarray(prof['anchor_xyz'])-p['C'][k])>.15 and prof['state'] not in ('LOW_SUPPORT','OUTLIER'):row['anchor_status']='PHYSICALLY_INCONSISTENT'
            perstation.append(row)
        tags=[];causes=[]
        if base['overshoot_max']>.05:tags.append('overshoot');causes.append('SPLINE_OVERSHOOT')
        if r['max_vertical_curvature']>3*protocol['vertical_prior'] or base['max_vertical_curvature']>3*protocol['vertical_prior']:tags.append('vertical-wave');causes.append('VERTICAL_WAVE')
        if r['curvature_violation_fraction']>0 or base['curvature_violation_fraction']>0:tags.append('horizontal-kink');causes.append('HORIZONTAL_KINK')
        if r['heading_jump_8m_deg']>1:tags.append('curve')
        if r['max_grade']>.025:tags.append('grade')
        if np.diff(curve['anchor_s']).min()<.5:causes.append('DUPLICATE_STATION')
        if np.diff(curve['anchor_s']).max()>8:tags.append('gap');causes.append('IRREGULAR_GAP')
        if profile:
            if np.mean([v['n']<5 for v in profile])>.1:tags.append('sparse');causes.append('SPARSE_ANCHOR')
            if np.mean([v['visibility'] in ('ONE_FACE','SPARSE_MULTI') for v in profile])>.1:tags.append('partial-profile');causes.append('ANCHOR_PARTIAL_PROFILE')
            if any(len(v.get('modes',[]))>1 for v in profile):causes.append('MULTIMODAL_PROFILE')
            if any(v.get('source_weights') and max(x['cost'] for x in v['source_weights'])>3*max(.1,np.median([x['cost'] for x in v['source_weights']])) for v in profile):causes.append('SOURCE_REGISTRATION')
        spread=stats(e['gt_uncertainty']).get('p95',0)
        if spread>.04:causes.append('GT_AMBIGUITY')
        if bad:tags.append('gt20')
        if casekey in oldbad:tags.append('old-failure')
        oldfar=base.get('far_p95') or 0.;far=r.get('far_p95') or 0.;text=f'Дальний рельс: p95 {oldfar*100:.1f} → {far*100:.1f} см; максимум нового {np.nanmax(e["error"][take,1])*100:.1f} см.' if take.any() else 'Нет доступного будущего rail GT.'
        if 'SPLINE_OVERSHOOT' in causes:text+=f' Старый cubic отклонялся от C4-полилинии до {base["overshoot_max"]*100:.1f} см.'
        if 'MULTIMODAL_PROFILE' in causes:text+=' У части профилей сохраняется несколько допустимых положений reference.'
        if 'SOURCE_REGISTRATION' in causes:text+=' Есть несогласие источников; оно может быть связано с регистрацией или видимой поверхностью, причина не доказана.'
        if 'GT_AMBIGUITY' in causes:text+=' Расчётный GT имеет повышенный разброс.'
        case=dict(id=casekey,run=run,frame=i,file=frames(run)[i]['file'],group=group,method=best,tags=tags,causes=causes,old_far_p95=oldfar,far_p95=far,far_max=float(np.nanmax(e['error'][take,1])) if take.any() else 0.,worst_station=float(e['station'][worst]),max_range=r['max_range'],evaluated_count=int(sum(take)),max_evaluated_range=float(np.max(np.linalg.norm(op['C'][take],axis=1))) if take.any() else 0.,analysis=text,physical_fraction=r['physical_violation_fraction'],old_physical_fraction=base['physical_violation_fraction'],alignment_count=r['alignment_count'],old_station_count=r['old_station_count']);cases.append(case);decomposition.append(dict(run=run,frame=i,group=group,old_far_p95=oldfar,new_far_p95=far,far_gt20cm=r['far_gt20cm'],old_far_gt20cm=base['far_gt20cm'],categories='|'.join(causes),analysis=text,diagnostic_not_exact_causal_attribution=True))
        # Oracle comparison restricted to identical stations and compatible GT.
        path=OLD_OUT/'oracle_decomposition'/oldgroup/'B4_SPLINE_50__O1'/casekey/'curves.npz'
        if path.exists():
            d=npz(path)
            if 'oracle_CR_production_alpha' in d:
                gt_agree=np.nanmax(np.abs(d['gt']-e['gt']),axis=(1,2))<1e-7;valid=d['valid']&take&gt_agree
                def err(name):
                    delta=d[name]-d['gt'];delta-=np.einsum('nri,ni->nr',delta,d['common_B'][:,:,0])[:,:,None]*d['common_B'][:,None,:,0];return np.linalg.norm(delta[:,1],axis=1)
                current=err('production_CR_production_alpha');oracle=err('oracle_CR_production_alpha');oracle_roll=err('production_CR_oracle_alpha');oracle_both=err('oracle_CR_oracle_alpha')
                for lo,hi in BINS:
                    select=valid&(np.linalg.norm(d['C'],axis=1)>=lo)&(np.linalg.norm(d['C'],axis=1)<hi)
                    if not select.any():continue
                    oldv=float(np.percentile(current[select],95));newv=float(np.percentile(e['error'][select,1],95));orv=float(np.percentile(oracle[select],95));oracle_rows.append(dict(run=run,frame=i,group=group,lo=lo,hi=hi,stations=int(sum(select)),current_per_start_p95=oldv,new_per_start_p95=newv,oracle_CR_per_start_p95=orv,oracle_roll_per_start_p95=float(np.percentile(oracle_roll[select],95)),oracle_both_per_start_p95=float(np.percentile(oracle_both[select],95)),gap_fraction=(oldv-newv)/(oldv-orv) if abs(oldv-orv)>1e-8 else None))
    write_csv(OUT/'per_station.csv',perstation);save(OUT/'per_station.json',perstation);write_csv(OUT/'failure_decomposition.csv',decomposition);write_csv(OUT/'oracle_gap_per_start.csv',oracle_rows)
    oracle_summary=[]
    for group in ('phase_d','phase_e'):
        for lo,hi in BINS:
            rr=[r for r in oracle_rows if r['group']==group and r['lo']==lo and r['hi']==hi]
            if not rr:continue
            oldv=float(np.median([r['current_per_start_p95'] for r in rr]));newv=float(np.median([r['new_per_start_p95'] for r in rr]));orv=float(np.median([r['oracle_CR_per_start_p95'] for r in rr]));oracle_summary.append(dict(group=group,lo=lo,hi=hi,starts=len(rr),stations=sum(r['stations'] for r in rr),current=oldv,new=newv,oracle_CR=orv,oracle_roll=float(np.median([r['oracle_roll_per_start_p95'] for r in rr])),oracle_both=float(np.median([r['oracle_both_per_start_p95'] for r in rr])),gap_closed=(oldv-newv)/(oldv-orv) if abs(oldv-orv)>1e-8 else None,aggregation='median of matched per-start p95, same planes, same GT, same available stations'))
    write_csv(OUT/'oracle_gap.csv',oracle_summary);save(OUT/'oracle_gap.json',oracle_summary)
    # Selection of human examples is post-evaluation reporting only.
    long=[c for c in cases if c['max_evaluated_range']>=30 and c['evaluated_count']>=10];sortedcases=sorted(long,key=lambda r:r['far_p95']);chosen={}
    def add(c,tag):
        if c['id'] not in chosen:chosen[c['id']]=c
        if tag not in chosen[c['id']]['tags']:chosen[c['id']]['tags'].append(tag)
    for c in sortedcases[:5]:add(c,'best')
    middle=len(sortedcases)//2
    for c in sortedcases[max(0,middle-5):middle+5]:add(c,'median')
    for c in sortedcases[-20:][::-1]:add(c,'worst')
    for c in cases:
        if 'old-failure' in c['tags'] or 'gt20' in c['tags']:add(c,'old-failure' if 'old-failure' in c['tags'] else 'gt20')
    for tag in ('overshoot','partial-profile','vertical-wave','horizontal-kink','sparse','curve','grade','gap'):
        for c in sorted([c for c in cases if tag in c['tags']],key=lambda c:c['max_range'],reverse=True)[:3]:add(c,tag)
    assert oldbad <= set(chosen);items=list(chosen.values());save(OUT/'gallery/cases.json',items);save(OUT/'all_cases.json',cases)
    categories=['SPLINE_OVERSHOOT','ANCHOR_PARTIAL_PROFILE','VERTICAL_WAVE','HORIZONTAL_KINK','DUPLICATE_STATION','IRREGULAR_GAP','SPARSE_ANCHOR','MULTIMODAL_PROFILE','SOURCE_REGISTRATION','SIDE_TRANSITION','GT_AMBIGUITY']
    for cat in categories:save(OUT/'failure_corpus'/(cat+'.json'),dict(category=cat,cases=[c for c in cases if cat in c['causes']],note='Diagnostic evidence; not necessarily sole proven cause. SIDE_TRANSITION has no event in frozen C4, which carries only initial side.'))
    save(OUT/'audit/fresh_holdout.json',dict(annotated_runs=[p.name for p in dataset().iterdir() if p.is_dir()],all_annotated_runs_already_in_previous_split=True,verified_fresh_candidates=[],other_datasets='Previously processed unannotated recordings are not a certified fresh blind holdout; not consumed in this study.',new_holdout_consumed=False))
    summary=dict(best=best,benchmark_available=sum(r['status']=='AVAILABLE' for r in allstarts if r['group']=='phase_e' and r['method']==best),old_bad_cases=sorted(oldbad),new_bad_benchmark=[c['id'] for c in cases if c['group']=='phase_e' and 'gt20' in c['tags']],new_bad_development=[c['id'] for c in cases if c['group']=='phase_d' and 'gt20' in c['tags']],gallery_cases=len(items),per_station_rows=len(perstation),metrics=methodmetrics,oracle=oracle_summary);save(OUT/'final_summary.json',summary);print('ANALYSIS',best,summary['benchmark_available'],'available',len(summary['new_bad_benchmark']),'benchmark failures',len(items),'gallery cases')

if __name__=='__main__':main()
