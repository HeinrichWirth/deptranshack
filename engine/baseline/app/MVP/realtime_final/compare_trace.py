from . import OUT
import long_common as lc
import numpy as np
import argparse,struct

def read(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def ulp(a,b):
    if not np.isfinite(a) or not np.isfinite(b):return None
    def order(x):
        u=struct.unpack('<Q',struct.pack('<d',float(x)))[0]
        return (~u)&((1<<64)-1) if u>>63 else u|(1<<63)
    return abs(order(a)-order(b))
def first(a,b):
    for i in range(min(len(a['names']),len(b['names']))):
        op=str(a['names'][i]);meta=a['meta'][i].tolist()
        if op!=str(b['names'][i]) or not np.array_equal(a['meta'][i],b['meta'][i]):return dict(event=i,operation=op,other_operation=str(b['names'][i]),meta=meta,reason='event/order divergence')
        ai=a['ids'][a['id_offsets'][i]:a['id_offsets'][i+1]];bi=b['ids'][b['id_offsets'][i]:b['id_offsets'][i+1]]
        if not np.array_equal(ai,bi):return dict(event=i,operation=op,meta=meta,reason='support/input IDs',count_a=len(ai),count_b=len(bi))
        av=a['values'][a['value_offsets'][i]:a['value_offsets'][i+1]];bv=b['values'][b['value_offsets'][i]:b['value_offsets'][i+1]]
        if av.shape!=bv.shape:return dict(event=i,operation=op,meta=meta,reason='value shape',count_a=len(av),count_b=len(bv))
        same=(av==bv)|(np.isnan(av)&np.isnan(bv));ix=np.flatnonzero(~same)
        if len(ix):
            k=int(ix[0]);row=dict(event=i,operation=op,meta=meta,reason='numeric',field=k,value_a=float(av[k]),value_b=float(bv[k]),ULP=ulp(av[k],bv[k]),values_a=av.tolist(),values_b=bv.tolist())
            if op=='pca_libm':row['math_inputs_identical']=bool(np.array_equal(av[:5],bv[:5]));row['field_name']=['iteration','p','q','atan2_y','atan2_x','angle','cos','sin'][k]
            return row
    return dict(reason='identical' if len(a['names'])==len(b['names']) else 'event count',events_a=len(a['names']),events_b=len(b['names']))

def main():
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');a=p.parse_args();rows=[]
    for case in lc.load(OUT/'captures/manifest.json')['rows']:
        name=case['file'];ta=read(OUT/'determinism_trace'/a.a/(name[:-4]+'_trace.npz'));tb=read(OUT/'determinism_trace'/a.b/(name[:-4]+'_trace.npz'))
        row=dict(run=case['run'],frame=case['frame'],platform_a=a.a,platform_b=a.b,**first(ta,tb))
        aa=read(OUT/'determinism_trace'/a.a/name);bb=read(OUT/'determinism_trace'/a.b/name)
        row['selected_ids_equal']=np.array_equal(aa['keys'],bb['keys']);row['anchor_shapes_equal']=aa['anchors'].shape==bb['anchors'].shape
        row['anchor_max_delta_m']=float(np.max(abs(aa['anchors']-bb['anchors']))) if row['anchor_shapes_equal'] else None
        rows.append(row);print('FIRST',row['run'],row['frame'],row['reason'],row.get('operation'),row.get('ULP'),row['selected_ids_equal'],row['anchor_max_delta_m'],flush=True)
    lc.csv_write(OUT/f'cross_os_first_divergence_{a.a}_{a.b}.csv',rows);lc.save(OUT/f'trace_comparison_{a.a}_{a.b}.json',rows)
    if a.a=='windows_1' and a.b=='linux_gcc_1':lc.csv_write(OUT/'cross_os_first_divergence.csv',rows)

if __name__=='__main__':main()
