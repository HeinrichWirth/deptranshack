"""Final binary, isolated two-backend paired repetition. No variant selection."""
from . import OUT
from .engine import Engine
from .native_api import startup
from MVP.performance_final.ablation import fingerprint
from MVP.final_pipeline.memory import rss
from pathlib import Path
import long_common as lc

def main():
    rows=[];engines={};run=None
    for j,case in enumerate(lc.load(OUT/'benchmark_cohort.json')['clean']):
        if case['run']!=run:
            for e in engines.values():e.close()
            run=case['run'];engines={'python':Engine('python'),'batch':Engine('native_batch_spatial')}
            for e in engines.values():e.start_run(lc.dataset()/run)
        hashes=[]
        for name in (('python','batch') if j%2==0 else ('batch','python')):
            r=engines[name].process_frame(case['frame']);hashes.append(fingerprint(r));rows.append(dict(**case,backend=name,**r['summary']['timing']))
        assert len(set(hashes))==1
        print('FINAL_TIMING',j+1,32,flush=True)
    for e in engines.values():e.close()
    lc.save(OUT/'confirmation.json',dict(rows=rows,native_sha256=lc.sha(Path(startup().__file__)),bitwise=True,
        peak_RSS_bytes=rss()['peak_rss_bytes'],protocol='isolated final build, two backends, alternating order on each same frame; no selection'))

if __name__=='__main__':main()
