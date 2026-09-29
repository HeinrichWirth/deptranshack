from . import ROOT,OUT
import long_common as lc
import numpy as np
import argparse,platform,time,sys,hashlib

def compact(path,events):
    values=[];ids=[];vo=[0];io=[0];meta=[];names=[]
    for e in events:
        names.append(e['operation']);meta.append([e[k] for k in ('step','pack','candidate','iteration','trial')]);values.extend(e['values']);ids.extend(e['ids']);vo.append(len(values));io.append(len(ids))
    np.savez_compressed(path,names=np.array(names),meta=np.array(meta,dtype=np.int32),values=np.array(values),ids=np.array(ids,dtype=np.uint64),value_offsets=np.array(vo),id_offsets=np.array(io))

def raw_arrays(out):
    blocks=out['blocks']
    return dict(keys=np.concatenate([np.asarray(b['keys'],np.uint64) for b in blocks]),anchors=np.asarray([b['anchor_3d'] for b in blocks]),bases=np.asarray([b['basis_columns'] for b in blocks]),
        uvw=np.concatenate([np.asarray(b['uvw']) for b in blocks]),state=np.asarray([b['state'] for b in blocks]),rank=np.asarray([b['rank'] for b in blocks]),reason=np.array(out['reason']))

def replay(path,threads=1,trace=True,profile=False,original=False,overrides=None):
    if original:from MVP.c4_v2.native_api import startup;mod=startup()
    else:from .native_api import load;mod=load()
    with np.load(path) as z:
        get=lambda k:(overrides or {}).get(k,z[k])
        m=mod.Marcher(threads,12,0)
        if not original:m.diagnostics(trace,profile)
        for k in sorted(z['history']):m.add_frame(int(k),z[f'world_{k}'])
        start=time.perf_counter();out=m.extend_track(get('origin'),get('basis'),get('seed_ids'),get('template'),get('pose'),get('inverse'),int(get('index')),get('history').tolist(),float(get('confidence')));dt=time.perf_counter()-start
        return raw_arrays(out),([] if original else m.trace()),m.stats(),dt

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--threads',type=int,default=1);p.add_argument('--profile',action='store_true');p.add_argument('--no-trace',action='store_true');a=p.parse_args();rows=[]
    manifest=lc.load(OUT/'captures/manifest.json');dest=OUT/'determinism_trace'/a.tag;dest.mkdir(parents=True,exist_ok=True)
    for case in manifest['rows']:
        path=OUT/'captures'/case['file'];assert lc.sha(path)==case['sha256']
        output,events,stats,dt=replay(path,a.threads,not a.no_trace,a.profile);np.savez_compressed(dest/case['file'],**output)
        if events:compact(dest/(path.stem+'_trace.npz'),events)
        old,_,_,_=replay(path,a.threads,False,False,True)
        eq={k:np.array_equal(output[k],old[k],equal_nan=True) if output[k].dtype.kind not in 'US' else np.array_equal(output[k],old[k]) for k in output}
        row=dict(case,threads=a.threads,trace_events=len(events),wall_ms=dt*1000,profile=stats,instrumentation_unchanged=all(eq.values()),equality=eq);rows.append(row)
        print('TRACE',a.tag,case['run'],case['frame'],len(events),all(eq.values()),flush=True)
    lc.save(dest/'SUMMARY.json',dict(rows=rows,platform=platform.platform(),python=sys.version,time_ns=time.time_ns()))

if __name__=='__main__':main()
