from fusion_common import *

def main():
    rows=load(OUT/'analysis_rows.json');lock=load(OUT/'research_lock.json');selected=lock['selected']
    pool=[r for r in rows if r['variant'] not in ('F0','FUSED_BOOTSTRAP','M2_F0','M2_BEST') and r.get('evaluation_eligible')];cases={}
    def add(rr,tag,n=3):
        for r in rr[:n]:
            k=(r['variant'],r['run'],r['start_frame']);entry=cases.setdefault(k,dict(row=r,tags=[]))
            if tag not in entry['tags']:entry['tags'].append(tag)
    add(sorted(pool,key=lambda r:r.get('delta_reach',0),reverse=True),'rescue',4)
    add(sorted(pool,key=lambda r:r['continuous_reach_5cm'],reverse=True),'farthest',4)
    add(sorted([r for r in pool if r['variant']==selected],key=lambda r:abs(r.get('delta_reach',0))),'no_effect',3)
    add(sorted([r for r in pool if r.get('delta_reach',0)>4 and r.get('baseline_diagnosis')=='ORIENTATION_FAILURE'],key=lambda r:-r['delta_reach']),'orientation_rescue',3)
    for reason,tag in (('TOO_FEW_SUPPORT','too_few_rescue'),('NO_NEW_POINTS','no_new_rescue'),('SENSOR_SPARSITY','sparsity_rescue')):
        add(sorted([r for r in pool if r.get('delta_reach',0)>4 and reason in (r.get('baseline_stop'),r.get('baseline_diagnosis'))],key=lambda r:-r['delta_reach']),tag,3)
    add(sorted(pool,key=lambda r:r.get('delta_reach',0)),'worse',4)
    add(sorted(pool,key=lambda r:r.get('past_only_terminal_roi',0),reverse=True),'ghost_diagnostic',3)
    bb=csv_read(OUT/'registration_blur.csv');scores={}
    for b in bb:
        if int(b['age_frames'])>0:scores[b['variant'],b['run'],int(b['start_frame'])]=max(float(b['residual_p95']),scores.get((b['variant'],b['run'],int(b['start_frame'])),0))
    add(sorted(pool,key=lambda r:scores.get((r['variant'],r['run'],r['start_frame']),0),reverse=True),'blur',3)
    median=float(np.median([r['continuous_reach_5cm'] for r in pool if r['variant']==selected]))
    add(sorted([r for r in pool if r['variant']==selected],key=lambda r:abs(r['continuous_reach_5cm']-median)),'median',4)
    improved=[r for r in pool if r.get('delta_reach',0)>4]
    typical_gain=float(np.median([r['delta_reach'] for r in improved]))
    add(sorted(improved,key=lambda r:abs(r['delta_reach']-typical_gain)),'median_improvement',3)
    # All large-error and high-confidence suspects, not only attractive improvements.
    for r in rows:
        if not r.get('seed_available') or r['variant']=='FUSED_BOOTSTRAP':continue
        if r.get('raw_wrong_structure_trigger') or r.get('wrong_structure_suspected'):add([r],'wrong_structure_review',1)
    examples=list(cases.values());save(OUT/'gallery/examples.json',examples)
    chosen=[];seen=set()
    for tag in ('rescue','farthest','orientation_rescue','too_few_rescue','no_new_rescue','sparsity_rescue','worse','blur','ghost_diagnostic','median','no_effect','wrong_structure_review'):
        for e in [x for x in examples if tag in x['tags']][:2]:
            k=(e['row']['run'],e['row']['start_frame'])
            if k not in seen:chosen.append(e);seen.add(k)
    for e in examples:
        if len(chosen)>=24:break
        k=(e['row']['run'],e['row']['start_frame'])
        if k not in seen:chosen.append(e);seen.add(k)
    assert len(chosen)>=20;save(OUT/'las/examples.json',chosen)
    # An animation needs two actual recorded planes; never duplicate a static
    # failure image and call it a marching sequence.
    animated=[]
    for e in examples:
        r=e['row'];p=load(OUT/'heldout'/r['variant']/key(r['run'],r['start_frame'])/'prediction.json')
        if sum(st.get('plane_origin') is not None for st in p['steps'])>=2:animated.append(e)
    gifs=[];seen=set()
    for tag in ('sparsity_rescue','orientation_rescue','median_improvement','farthest','no_effect','worse','blur','ghost_diagnostic'):
        e=next((e for e in animated if tag in e['tags'] and (e['row']['variant'],e['row']['run'],e['row']['start_frame']) not in seen),None)
        if e is None:e=next(e for e in animated if (e['row']['variant'],e['row']['run'],e['row']['start_frame']) not in seen)
        gifs.append(e);seen.add((e['row']['variant'],e['row']['run'],e['row']['start_frame']))
    save(OUT/'animations/examples.json',gifs);print('EXAMPLES',len(examples),'LAS',len(chosen),'GIF',len(gifs),flush=True)
if __name__=='__main__':main()
