"""Rebuild saved causal evidence by source identity for display/export only."""
from long_common import *
from fusion import CausalSource
from roi_source import point_keys

def case_data(e):
    folder=OUT/e['folder'];p=read_prediction(folder);run=p['run'];i=p['start_frame']
    current,_=CausalSource(run,i).read(i)
    fields=('xyz','source_frame','source_row','source_point_index','age_frames','ring')
    with np.load(folder/'evidence.npz') as z:history={f:z[f] for f in fields}
    c={f:np.concatenate([current[f],history[f]]) for f in fields};c['keys']=point_keys(c)
    with np.load(folder/'provenance.npz') as z:pred={f:z[f] for f in z.files}
    pk=point_keys(pred);order=np.argsort(c['keys']);ids=order[np.searchsorted(c['keys'][order],pk)]
    assert np.array_equal(c['keys'][ids],pk);p=dict(p,indices=ids)
    ref,info=reference_readonly(run,i)
    return p,c,ref,info

def choose():
    lock=load(OUT/'research_freeze.json');rows=load(OUT/'phase_e_benchmark/analysis_rows.json');best=lock['best_development'];chosen={}
    def add(r,tag):
        id=r['variant']+'__'+key(r['run'],r['start_frame']);chosen.setdefault(id,dict(r,tags=[]))['tags'].append(tag)
    rr=[r for r in rows if r['variant']==best and r.get('evaluation_eligible')]
    bydelta=sorted(rr,key=lambda r:r.get('delta_reach',0));byreach=sorted(rr,key=lambda r:r['continuous_reach_5cm'])
    for r in bydelta[:12]:add(r,'worst_losses')
    for r in bydelta[-12:]:add(r,'best_gains')
    for r in byreach[-12:]:add(r,'best_reach')
    for r in byreach[:8]:add(r,'shortest')
    middle=len(byreach)//2
    for r in byreach[max(0,middle-6):middle+6]:add(r,'median')
    for method in lock['top3'][1:]:
        other=[r for r in rows if r['variant']==method and r.get('evaluation_eligible')]
        gain=sorted(other,key=lambda r:r.get('delta_reach',0));reach=sorted(other,key=lambda r:r['continuous_reach_5cm']);mid=len(reach)//2
        for r in gain[-2:]:add(r,'comparison_best')
        for r in gain[:2]:add(r,'comparison_worst')
        for r in reach[mid-1:mid+1]:add(r,'comparison_median')
    negative=[r for r in rr if (r.get('max_GT_available_range') or 0)>=100 and r['continuous_reach_5cm']<50]
    for r in sorted(negative,key=lambda r:-(r.get('observable_F16') or 0))[:20]:add(r,'negative_100m_detail')
    for r in rows:
        if (r.get('confirmed_transverse_segments_gt20cm') or 0)>0:add(r,'catastrophic_review')
        if (r.get('max_confirmed_observed_range') or 0)>=75:add(r,'observed_75m')
        if (r.get('max_confirmed_observed_range') or 0)>=100:add(r,'observed_100m')
        if r['variant']==best and (r.get('max_GT_available_range') or 0)>=100 and (r.get('continuous_reach_5cm') or 0)<100:add(r,'100m_failure')
    # All long development cases, not only the winning configuration.
    for r in load(OUT/'audit/current_development_review.json'):add(r,'development_visual_review')
    for group in ('phase_a_screen','phase_b_full_dev','phase_d_combinations'):
        for path in (OUT/group).glob('*/*/evaluation.json'):
            r=load(path)['summary']
            if (r.get('max_confirmed_observed_range') or 0)>=75:
                add(dict(r,folder=path.parent.relative_to(OUT).as_posix(),group=group),'development_75m')
    items=list(chosen.values())
    for e in items:e['id']=e['variant']+'__'+key(e['run'],e['start_frame']);e['tags']=list(dict.fromkeys(e['tags']))
    save(OUT/'gallery/examples.json',items)
    las=[e for e in items if (e.get('max_confirmed_observed_range') or 0)>=75 or any(t in e['tags'] for t in ('negative_100m_detail','catastrophic_review'))]
    save(OUT/'las/examples.json',las)
    print('EXAMPLES',len(items),'LAS',len(las),'negative detailed',min(20,len(negative)))

if __name__=='__main__':choose()
