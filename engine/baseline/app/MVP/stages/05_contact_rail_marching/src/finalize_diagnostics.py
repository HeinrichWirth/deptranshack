"""Refresh diagnoses after ALL oracle experiments, reusing measured raw support."""
from common import *
from collections import Counter

def main():
    oracle_rows=[]
    for p in (OUT/'oracles').glob('*/*/evaluation.json'):
        e=load(p);oracle_rows.append(dict(oracle=p.parent.parent.name,**e['summary']))
    oracle={(r['run'],r['start_frame'],r['oracle']):r for r in oracle_rows}
    rows=load(OUT/'analysis_rows.json')
    for r in rows:
        if not r['seed_available']:continue
        run=r['run'];i=r['start_frame'];path=OUT/'failures'/(key(run,i)+'.json');packet=load(path)
        raw_evaluation=load(OUT/'heldout'/key(run,i)/'evaluation.json')
        r['raw_wrong_structure_trigger']=bool(raw_evaluation['summary'].get('wrong_structure_suspected'))
        bad=[st for st in raw_evaluation['steps'] if st['status']=='ACCEPTED' and st.get('point_p95') is not None and st['point_p95']>.20]
        reference_end=bool(bad and all(st.get('evaluated_fraction',0)<.80 for st in bad))
        if r['raw_wrong_structure_trigger'] and reference_end:
            r['wrong_structure_suspected']=False;r['wrong_structure_review']='INSUFFICIENT_GT_COVERAGE_AT_REFERENCE_END'
        of=oracle[run,i,'ORACLE_FRAME'];osd=oracle[run,i,'ORACLE_SEED_8'];reach=r.get('continuous_reach_5cm') or 0
        future=r['terminal_future_points'];returns=r['terminal_observable_returns'];prior=r['terminal_evaluation_reason']
        if not r.get('evaluation_eligible'):diagnosis='GT_UNAVAILABLE_AHEAD'
        elif reference_end:diagnosis='CR_GT_GAP'
        elif r.get('wrong_structure_suspected'):diagnosis='WRONG_STRUCTURE_SUSPECTED'
        elif future<3:diagnosis=prior if prior in ('CR_SIDE_CHANGE_SUSPECTED','RUN_END','CR_GT_GAP') else ('RUN_END' if packet['reference']['horizon_reason']=='RUN_END' else 'CR_GT_GAP')
        elif (of.get('continuous_reach_5cm') or 0)>reach+4:diagnosis='ORIENTATION_FAILURE'
        elif (osd.get('continuous_reach_5cm') or 0)>reach+4:diagnosis='SEED_OR_INITIAL_DRIFT'
        elif returns<3:diagnosis='SENSOR_SPARSITY'
        else:diagnosis='OBSERVATION_TEMPLATE_LIMIT'
        r['terminal_evaluation_reason']=diagnosis;packet.update(summary=r,diagnosis=diagnosis,oracle_frame=of,oracle_seed=osd)
        # Post-lock sensitivity study: retained separately from the original diagnosis rule.
        for name,field in (('ORACLE_SEED_TEMPLATE8','oracle_seed_template'),('ORACLE_SEED_TEMPLATE8_FRAME','oracle_seed_template_frame')):
            extra=oracle.get((run,i,name))
            if extra is not None:packet[field]=extra
        save(path,packet)
    s=load(OUT/'SUMMARY.json');good=[r for r in rows if r.get('evaluation_eligible')];s['diagnoses']=dict(Counter(r['terminal_evaluation_reason'] for r in good));s['oracle_diagnoses_complete']=True
    s['raw_wrong_structure_triggers']=sum(r.get('raw_wrong_structure_trigger',False) for r in rows)
    s['wrong_structure_after_GT_coverage_review']=sum(r.get('wrong_structure_suspected',False) for r in rows)
    save(OUT/'SUMMARY.json',s);save(OUT/'analysis_rows.json',rows);csv_write(OUT/'per_start_frame.csv',rows);csv_write(OUT/'oracle_comparison.csv',oracle_rows)
    csv_write(OUT/'failure_summary.csv',[dict(type='detector_stop',reason=k,count=v) for k,v in s['stops'].items()]+[dict(type='offline_diagnosis',reason=k,count=v) for k,v in s['diagnoses'].items()])
    # Exact requested named geometry columns, plus profile coverage by range.
    steps=csv_read(OUT/'per_step.csv');predictions={};tb=[]
    for st in steps:
        run=st['run'];i=int(st['start_frame']);k=(run,i)
        if k not in predictions:predictions[k]={x['step_index']:x for x in load(OUT/'heldout'/key(run,i)/'prediction.json')['steps']}
        source=predictions[k].get(int(st['step_index']))
        if source and source.get('template_candidates'):st['template_region_coverage']=source['template_candidates'][0]['coverage']
        for j,axis in enumerate(('x','y','z')):
            st['plane_origin_'+axis]=st.get('plane_origin_'+str(j));st['tangent_'+axis]=st.get('basis_'+str(3*j))
    csv_write(OUT/'per_step.csv',steps)
    for lo,hi in zip((0,10,20,30,40,50,60,75,100,125),(10,20,30,40,50,60,75,100,125,150)):
        rr=[st for st in steps if st.get('range_far') not in ('',None) and lo<=float(st['range_far'])<hi and st.get('template_region_coverage') not in ('',None)]
        tb.append(dict(lo=lo,hi=hi,**percentiles([float(st['template_region_coverage']) for st in rr])))
    csv_write(OUT/'template_coverage_bins.csv',tb)
    print('FINAL DIAGNOSES',s['diagnoses'],'oracle rows',len(oracle_rows),flush=True)
if __name__=='__main__':main()
