"""Descriptive development-only fusion conditions; never a benchmark selector."""
from long_common import *
from collections import defaultdict

def main():
    samples=[]
    for folder in sorted((OUT/'phase_a_screen/A0').iterdir()):
        if not (folder/'evaluation.json').exists():continue
        a=load(folder/'evaluation.json')['summary']
        if not a.get('evaluation_eligible'):continue
        p=load(folder/'prediction.json');s=p['steps'][-1] if p['steps'] else {}
        cc=[c for t in s.get('trials',[]) for c in t.get('candidates',[])];c=cc[0] if cc else {}
        q=s.get('first_overlap',{}).get('p90');n=c.get('support_unique',0)
        for name in ('B8','AF8','AF8_B','SF248','AFUSION'):
            e=load(OUT/'phase_a_screen'/name/folder.name/'evaluation.json')['summary']
            if not e.get('evaluation_eligible'):continue
            samples.append(dict(run=a['run'],start_frame=a['start_frame'],variant=name,
              terminal_reason=p['reason'],support_group='0' if n==0 else '1-2' if n<3 else '3+',
              old_overlap_group='missing' if q is None else '<=3cm' if q<=.03 else '3-5cm' if q<=.05 else '>5cm',
              current_candidate_support=n,current_overlap_p90=q,source_count=c.get('source_count',0),
              current_template_regions=c.get('template_regions',0),current_margin=c.get('competitor_margin'),
              baseline_reach=a['continuous_reach_5cm'],fusion_reach=e['continuous_reach_5cm'],
              delta=e['continuous_reach_5cm']-a['continuous_reach_5cm']))
    grouped=defaultdict(list)
    for r in samples:
        for field in ('terminal_reason','support_group','old_overlap_group'):
            grouped[r['variant'],field,r[field]].append(r)
    rows=[]
    for (method,feature,value),rr in sorted(grouped.items()):
        rows.append(dict(variant=method,feature=feature,value=value,n=len(rr),runs=len(set(x['run'] for x in rr)),
          wins4=sum(x['delta']>4 for x in rr),losses4=sum(x['delta']<-4 for x in rr),mean_delta=float(np.mean([x['delta'] for x in rr])),median_delta=float(np.median([x['delta'] for x in rr]))))
    csv_write(OUT/'fusion/development_conditions_samples.csv',samples);csv_write(OUT/'fusion/development_conditions.csv',rows)
    save(OUT/'fusion/development_conditions.json',dict(scope='Descriptive post-hoc bins of T-only terminal diagnostics on fixed 36 development starts. No learned rule, no benchmark input, no inference configuration change. Final reach deltas do not prove that local rescue alone caused the gain.',rows=rows))
    print('FUSION CONDITIONS',len(samples),len(rows))

if __name__=='__main__':main()
