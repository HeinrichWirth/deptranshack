"""First control-flow divergence after tiny upstream bootstrap changes."""
from . import OUT
from .compare_trace import read,ulp
import numpy as np
import long_common as lc

def first_branch(a,b):
    for i in range(min(len(a['names']),len(b['names']))):
        op=str(a['names'][i]);meta=a['meta'][i].tolist()
        av=a['values'][a['value_offsets'][i]:a['value_offsets'][i+1]];bv=b['values'][b['value_offsets'][i]:b['value_offsets'][i+1]]
        why=None
        if op!=str(b['names'][i]) or not np.array_equal(a['meta'][i],b['meta'][i]):why='different iteration/event sequence'
        elif not np.array_equal(a['ids'][a['id_offsets'][i]:a['id_offsets'][i+1]],b['ids'][b['id_offsets'][i]:b['id_offsets'][i+1]]):why='different exact membership/support IDs'
        elif op in ('trial','support_acceptance') and av[-1]!=bv[-1]:why='acceptance predicate changed'
        elif op=='ranking' and not np.array_equal(av[::2],bv[::2]):why='different candidate rank order'
        elif op=='normal_equations' and (max(abs(av[6]),abs(av[7]))<1e-6)!=(max(abs(bv[6]),abs(bv[7]))<1e-6):why='gradient stopping predicate changed'
        if why:return dict(event=i,step=meta[0],pack=meta[1],candidate=meta[2],iteration=meta[3],trial=meta[4],operation=op,other_operation=str(b['names'][i]),reason=why,values_a=av.tolist()[:16],values_b=bv.tolist()[:16])
    return dict(reason='no discrete branch divergence',events_a=len(a['names']),events_b=len(b['names']))
def main():
    rows=[]
    for c in lc.load(OUT/'captures/manifest.json')['rows']:
        stem=c['file'][:-4];a=read(OUT/'input_diagnosis'/(stem+'_windows_inputs.npz'));b=read(OUT/'input_diagnosis'/(stem+'_linux_inputs.npz'))
        rows.append(dict(run=c['run'],frame=c['frame'],**first_branch(a,b)))
    lc.csv_write(OUT/'cross_os_first_branch.csv',rows);lc.save(OUT/'FIRST_BRANCH.json',rows)
    print(rows)
if __name__=='__main__':main()
