"""Counterbalanced same-frame comparison to diagnose chronological timing drift."""
from . import OUT
from .engine import Engine
from MVP.performance_final.ablation import fingerprint
from MVP.final_pipeline.memory import rss
import long_common as lc
import numpy as np

def main():
    modes=['python','native_spatial','fd','batch','grid','grid_native','soa'];rows={m:[] for m in modes};engines={};run=None
    cohort=lc.load(OUT/'benchmark_cohort.json')['clean']
    for ordinal,case in enumerate(cohort):
        if case['run']!=run:
            for e in engines.values():e.close()
            run=case['run'];engines={m:Engine(m) for m in modes}
            for e in engines.values():e.start_run(lc.dataset()/run)
        ordered=modes[ordinal%len(modes):]+modes[:ordinal%len(modes)];hashes=[]
        for mode in ordered:
            e=engines[mode];r=e.process_frame(case['frame']);h=fingerprint(r);hashes.append(h)
            rows[mode].append(dict(**case,backend=mode,hash=h,bitwise=True,**r['summary']['timing'],peak_RSS_bytes=rss()['peak_rss_bytes'],
                stats=dict(e.trackers[-1].stats) if e.trackers else {}))
        assert len(set(hashes))==1,(case,hashes)
        print('PAIRED',ordinal+1,len(cohort),flush=True)
    for e in engines.values():e.close()
    for mode in modes:lc.save(OUT/'paired'/f'{mode}.json',rows[mode])
    # Select equivalent measured winner, including the previously frozen backend.
    summary=[dict(backend=m,workers=1,full_median_ms=float(np.median([r['T_TOTAL'] for r in rows[m]]))*1000,
        C4_median_ms=float(np.median([r['T_C4_MARCHING'] for r in rows[m]]))*1000) for m in modes if m!='python']
    best=min(summary,key=lambda r:r['full_median_ms'])
    lc.save(OUT/'selection.json',dict(**best,selection='counterbalanced paired same 32 starts; no GT',rows=summary))
    print('SELECTED',best,flush=True)

if __name__=='__main__':main()
