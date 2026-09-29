from . import ROOT,OUT
import long_common as lc
import numpy as np
from scipy.stats import pearsonr,spearmanr
import json,csv

def stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)
def rows(path):return lc.load(path)['rows']
def main():
    ab=rows(OUT/'ablation.json');dev=rows(OUT/'development.json');prof=rows(OUT/'profile.json');perf=rows(OUT/'performance_bind.json');ext4=rows(OUT/'performance_ext4.json');lock=lc.load(OUT/'REALTIME_SELECTION_LOCK.json')
    variants=[]
    base=np.median([r['wall_ms'] for r in ab if r['variant']=='offline_original'])
    for name in dict.fromkeys(r['variant'] for r in ab):
        rr=[r for r in ab if r['variant']==name];ss=stats([r['wall_ms'] for r in rr]);variants.append(dict(variant=name,**ss,speedup=base/ss['median'],quality='96/96 paired exact C4 and downstream checks' if name!='offline_original' else 'baseline',decision='KEEP' if name=='memory_exact' else 'intermediate'))
    for name,why in [('combined_index','Per-frame shared grid is already adequate; STOP rule'),('density_thinning','Exact reductions reached live system gate; no geometry approximation needed'),('PGO','STOP rule; not run'),('CPU_specific','STOP rule; portable strict build retained')]:variants.append(dict(variant=name,decision='NOT_RUN',reason=why))
    lc.csv_write(OUT/'point_reduction_ablation.csv',variants)
    timing=[]
    for name,rr in [('development_original', [r for r in dev if r['variant']=='offline_original']),('development_memory',[r for r in dev if r['variant']=='memory_exact']),('Linux_bind',perf),('Linux_ext4',ext4)]:
        for metric in ('wall_ms','c4_ms','offline_full_ms','las_read_ms','arrival_prepare_ms','index_ms','source_metadata_ms'):
            vv=[r[metric] for r in rr if metric in r]
            if vv:timing.append(dict(mode=name,metric=metric,**stats(vv)))
    lc.csv_write(OUT/'live_memory_timing.csv',timing)
    count=[]
    for cohort,rr in [('development',dev),('heldout48_bind',perf),('profile',prof)]:
        for r in rr:
            c=r['counters'];count.append(dict(cohort=cohort,run=r['run'],frame=r['frame'],variant=r.get('variant','final'),raw_points=r['source_points'],history_points=r['history_raw_points'],broad_points=c.get('broad_points'),roi_points=c.get('native_roi_points'),candidate_points=c.get('candidate_points'),fit_points=c.get('fit_points'),candidate_fits=c.get('candidate_fits'),residual_evaluations=c.get('residual_evaluations'),steps=c.get('native_march_steps'),selected_points=r['selected_real_points'],materialized_history_points=r['materialized_history_points'],horizon=r['available_c4_observed_range'],wall_ms=r['wall_ms'],c4_ms=r['c4_ms'],native_ms=c.get('native_update_seconds',0)*1000,profile_timers=cohort=='profile'))
    lc.csv_write(OUT/'point_count_runtime.csv',count)
    regress=[]
    for cohort,variant in [('development','offline_original'),('development','memory_exact'),('heldout48_bind','final')]:
        rr=[r for r in count if r['cohort']==cohort and r['variant']==variant and r['candidate_fits']>0]
        for target in ('wall_ms','c4_ms','native_ms'):
            for feature in ('raw_points','history_points','broad_points','roi_points','candidate_points','fit_points','candidate_fits','steps','horizon'):
                good=[r for r in rr if r.get(feature) is not None]
                x=np.array([r[feature] for r in good],float);y=np.array([r[target] for r in good])
                if len(x)<4 or np.std(x)==0:continue
                slope,intercept=np.polyfit(x,y,1);regress.append(dict(cohort=cohort,variant=variant,target=target,feature=feature,n=len(x),pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic),slope_ms_per_unit=float(slope),slope_ms_per_100k=float(slope*100000) if 'points' in feature else None,intercept_ms=float(intercept),interpretation='observational association; raw N, ROI, scene and horizon confounded'))
            features=['raw_points','history_points','roi_points','candidate_fits','steps']
            X=np.array([[r[f] for f in features] for r in rr],float);y=np.array([r[target] for r in rr]);scale=X.std(0);scale[scale==0]=1;M=np.column_stack([np.ones(len(X)),(X-X.mean(0))/scale]);coef,_,rank,_=np.linalg.lstsq(M,y,rcond=None);res=y-M@coef;dof=len(y)-rank;cov=np.linalg.pinv(M.T@M)*(np.sum(res*res)/max(1,dof))
            for k,f in enumerate(features):regress.append(dict(cohort=cohort,variant=variant,target=target,feature=f,model='multivariable exploratory OLS',n=len(y),rank=int(rank),condition_number=float(np.linalg.cond(M)),slope_ms_per_unit=float(coef[k+1]/scale[k]),slope_ms_per_100k=float(coef[k+1]/scale[k]*100000) if 'points' in f else None,stderr_ms_per_unit=float(np.sqrt(cov[k+1,k+1])/scale[k]),r2=float(1-np.sum(res*res)/np.sum((y-y.mean())**2)),interpretation='not a causal cost estimate; correlated frames and predictors'))
    lc.csv_write(OUT/'runtime_regression.csv',regress)
    hist=[];roi=[];steps=[]
    for r in lc.load(OUT/'profile.json')['arrivals']:hist.append(r)
    for r in prof:
        if r['variant']!='memory_exact':continue
        c=r['counters']
        for group in ('profile','solver_profile'):
            for name,value in c.get(group,{}).items():roi.append(dict(run=r['run'],frame=r['frame'],group=group,operation=name,milliseconds=value*1000,note='nested timers; candidate-worker durations are summed, not wall latency'))
        for s in c.get('step_profile',[]):steps.append(dict(run=r['run'],frame=r['frame'],step=s[0],ROI_points=s[1],broad_transformed_points=s[2],candidate_fits=s[3],residual_evaluations=s[4],step_wall_ms=s[5]*1000,candidate_points=s[6] if len(s)>6 else None,fit_points=s[7] if len(s)>7 else None))
    lc.csv_write(OUT/'history_index_timing.csv',hist);lc.csv_write(OUT/'roi_timing.csv',roi);lc.csv_write(OUT/'per_step_points.csv',steps)
    live=[];events=[];cpu=[];mem=[];arr=[]
    for path in sorted((OUT/'live').glob('*.json')):
        obj=lc.load(path);s=obj['summary'];s['file']=path.name;live.append(s);tag=dict(run=s['run'],workers=s['workers'],threads=s['threads'],frames=s['frames'],file=path.name)
        for r in obj['arrivals']:arr.append(dict(**tag,**r))
        cpu.append(dict(**tag,metric='process_active_cores',value=s['active_cores'],linux_logical_cpu_percent=100*s['active_cores']/32,note='aggregate process CPU/wall; per-job process CPU overlaps and must not be summed'))
        for worker in range(s['workers']):
            rr=[r for r in obj['solves'] if r['worker']==worker];cpu.append(dict(**tag,worker=worker,metric='worker_thread_cpu_seconds',value=sum(r.get('worker_thread_cpu_seconds',0) for r in rr),busy_seconds=sum(r['wall_ms']/1000 for r in rr),note='thread CPU excludes separate candidate threads; missing on earliest matrix runs'))
        mem.append(dict(**tag,max_rss_kb=s['max_rss_kb'],live_ring_python_bytes=s['live_ring_bytes'],native_xyz_bytes=s['native_ring']['xyz_bytes'],note='RSS includes producer-only preloaded replay fixture; native grid and active snapshot leases additional. Raw arrays/index shared across workers, not multiplied by worker count.'))
        for r in obj['solves']:events.append(dict(**tag,frame=r['frame'],published=r['published'],status=r['publish_status'],wall_ms=r['wall_ms'],request_to_completion_ms=(r['completed']-r['requested'])*1000,source_to_publish_ms=(r['completed']-r['requested'])*1000+100,worker=r['worker']))
    lc.csv_write(OUT/'worker_matrix.csv',live);lc.csv_write(OUT/'realtime_10hz.csv',arr);lc.csv_write(OUT/'publish_latency.csv',events);lc.csv_write(OUT/'cpu_utilization.csv',cpu);lc.csv_write(OUT/'memory.csv',mem)
    comp=[]
    for name in ('linux_gcc_1','linux_clang_1','linux_gcc_8'):
        trace=lc.load(OUT/'determinism_trace'/name/'SUMMARY.json')['rows'];comp.append(dict(variant=name,tests=len(trace),all_exact_vs_original=all(r['instrumentation_unchanged'] for r in trace),quality='same-input anchors and selected IDs exact on 10 starts',timing='trace ON: not performance comparable',decision='GCC portable strict retained; Clang diagnostic only'))
    comp.append(dict(variant='LTO',decision='default pybind11 CMake link-time optimization already enabled in measured build; no separate ablation'))
    lc.csv_write(OUT/'compiler_ablation.csv',comp);lc.csv_write(OUT/'pgo_ablation.csv',[dict(variant='PGO',status='NOT_RUN',reason='STOP rule after system gate; no trained profile and no claimed PGO gain')])
    control=lc.load(OUT/'controlled_plus100k.json');pairs=[]
    for r in control:
        if r['variant']!='base':continue
        b=next(q for q in control if q['variant']=='plus100k' and all(q[k]==r[k] for k in ('run','frame','repeat')))
        pairs.append(dict(run=r['run'],frame=r['frame'],repeat=r['repeat'],**{k:b[k]-r[k] for k in ('wall_ms','c4_ms','arrival_ms','index_ms','metadata_ms')}))
    lc.save(OUT/'METRICS.json',dict(timing=timing,ablations=variants,workers=live,plus100k={k:stats([r[k] for r in pairs]) for k in ('wall_ms','c4_ms','arrival_ms','index_ms','metadata_ms')},point_summary={k:stats([r[k] for r in count if r['cohort']=='heldout48_bind' and r[k] is not None]) for k in ('raw_points','history_points','broad_points','roi_points','candidate_points','fit_points','selected_points')},roi_timers=roi))
    print('METRICS_COMPLETE')
if __name__=='__main__':main()
