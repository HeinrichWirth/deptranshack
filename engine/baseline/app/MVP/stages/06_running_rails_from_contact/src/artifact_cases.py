"""Select illustrative cases AFTER predictions; never affects model or metrics."""
from rr_common import *
from seed_input import read_input
from track_geometry import unit

def main():
    name=load(OUT/'research_freeze.json')['best_development'];items=[]
    for group in ('phase_b','phase_e'):
        for r in load(OUT/group/'per_start.json'):
            if r['method']!=name or r['status']!='AVAILABLE':continue
            folder=OUT/group/name/key(r['run'],r['frame'])
            with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
            with np.load(folder/'evaluation.npz') as z:e={k:z[k] for k in z.files}
            take=e['valid']&(p['s']>=8)
            if not take.any():continue
            rec=load(folder/'prediction.json');meta,a=read_input(OUT/rec['input_folder']);k=int(np.nanargmax(np.where(take,e['error'][:,1],np.nan)));tangents=p['B'][:,:,0];turn=np.arccos(np.clip(tangents@tangents[0],-1,1));localt=tangents@a['initial_basis'];grade=np.rad2deg(np.arcsin(np.clip(localt[:,2],-1,1)))
            badgt=float(np.quantile(e['gt_uncertainty'][take],.95));decomp={};path=OUT/'oracle_decomposition'/group/name/key(r['run'],r['frame'])/'curves.npz'
            if path.exists():
                with np.load(path) as d:
                    use=d['valid']&take
                    for variant in ('production_CR_production_alpha','oracle_CR_production_alpha','production_CR_oracle_alpha','production_CR_oracle_offsets','oracle_CR_oracle_alpha','all_oracle'):
                        if variant in d.files and use.any():
                            delta=d[variant][use]-d['gt'][use];B=d['common_B'][use];delta-=np.einsum('nri,ni->nr',delta,B[:,:,0])[:,:,None]*B[:,None,:,0];decomp[variant]=stats(np.linalg.norm(delta[:,1],axis=1)).get('p95')
            prod=decomp.get('production_CR_production_alpha');cause=[]
            if prod and prod>.03:
                for variant,label in [('oracle_CR_production_alpha','CR'),('production_CR_oracle_alpha','roll'),('production_CR_oracle_offsets','offsets')]:
                    if decomp.get(variant) is not None and decomp[variant]<.7*prod:cause.append(label)
            if not cause:cause=['combined / reference ambiguity']
            if badgt>.03:cause.append('GT disagreement >3cm')
            items.append(dict(id=key(r['run'],r['frame']),run=r['run'],frame=r['frame'],file=frames(r['run'])[r['frame']]['file'],group=group,method=name,far_p95=float(np.quantile(e['error'][take,1],.95)),far_max=float(e['error'][take,1].max()),near_p95=float(np.quantile(e['error'][take,0],.95)),max_evaluated=float(e['range'][take].max()),alpha_error_p95=float(np.quantile(abs(np.rad2deg(e['alpha_error'][take])),.95)),turn_deg=float(np.rad2deg(turn.max())),grade_deg=float(np.max(abs(grade))),GT_spread_p95=badgt,worst_station=float(p['s'][k]),worst_range=float(e['range'][k]),worst_near=float(e['error'][k,0]),worst_far=float(e['error'][k,1]),evaluated=int(take.sum()),oracle_p95=decomp,cause=cause,tags=[]))
    eligible=[r for r in items if r['max_evaluated']>=30 and r['evaluated']>=10];ordered=sorted(eligible,key=lambda r:r['far_p95']);mid=len(ordered)//2
    groups=dict(best=ordered[:5],median=ordered[max(0,mid-5):mid+5],worst_far=sorted(eligible,key=lambda r:-r['far_max'])[:20],high_roll=sorted(eligible,key=lambda r:-r['alpha_error_p95'])[:10],curve=sorted(eligible,key=lambda r:-r['turn_deg'])[:10],grade=sorted(eligible,key=lambda r:-r['grade_deg'])[:10],all_gt20cm=[r for r in items if r['far_max']>.2])
    for tag,rr in groups.items():
        for r in rr:r['tags'].append(tag)
    selected=[r for r in items if r['tags']];selected.sort(key=lambda r:-r['far_max']);save(OUT/'gallery/cases.json',selected);save(OUT/'gallery/all_cases_metrics.json',items);save(OUT/'gallery/selection.json',dict(method=name,case_count=len(selected),category_counts={k:len(v) for k,v in groups.items()},representative_scope='Development + reused benchmark; best/median/worst representative sets require >=30m evaluated horizon and >=10 stations. all_gt20cm includes every available post-seed case with any far station >20cm.',time_ns=time.time_ns()));write_csv(OUT/'failures/cases.csv',selected)
    print('GALLERY CASES',len(selected),{k:len(v) for k,v in groups.items()},flush=True)

if __name__=='__main__':main()
