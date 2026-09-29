"""Anchor comparison on complete accepted steps inside the shared range."""
from fusion_common import *
from collections import defaultdict

def main():
    steps=defaultdict(list)
    for s in csv_read(OUT/'per_step_variant.csv'):
        if s['status']=='ACCEPTED' and s.get('range_far') and s.get('GT_anchor_error'):
            steps[s['variant'],s['run'],s['start_frame']].append(s)
    out=[]
    for q in csv_read(OUT/'common_range_quality.csv'):
        limit=float(q['common_range']);r={k:q[k] for k in ('variant','run','start_frame','common_range')}
        for name,label in (('F0','baseline'),(q['variant'],'fusion')):
            ss=[float(s['GT_anchor_error']) for s in steps[name,q['run'],q['start_frame']] if float(s['range_far'])<=limit]
            r[label+'_n']=len(ss);r[label+'_p95']=percentiles(ss).get('p95')
        r['delta_p95']=r['fusion_p95']-r['baseline_p95'] if r['fusion_p95'] is not None and r['baseline_p95'] is not None else None
        out.append(r)
    csv_write(OUT/'common_anchor_quality.csv',out);print('COMMON ANCHORS',len(out),flush=True)
if __name__=='__main__':main()
