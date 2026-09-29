from . import OUT
import long_common as lc
import numpy as np

def main():
    rows=[]
    for path in (OUT/'ablations').glob('*.json'):
        data=lc.load(path)
        if len(data)!=32:continue
        rows.append(dict(backend=path.stem,equivalent=all(r['bitwise'] for r in data),
            full_median_ms=float(np.median([r['T_TOTAL'] for r in data]))*1000,
            C4_median_ms=float(np.median([r['T_C4_MARCHING'] for r in data]))*1000))
    # Threads are only selectable with a persisted explicit worker count. The
    # current final integration uses single-worker backends; report any better
    # threaded result instead of silently mapping it to a different backend.
    eligible=[r for r in rows if r['equivalent'] and not r['backend'].startswith('threads')]
    best=min(eligible,key=lambda r:r['full_median_ms'])
    lc.save(OUT/'selection.json',dict(**best,selection='minimum median on fixed 32-start performance cohort; no GT',rows=rows))
    print(best,flush=True)

if __name__=='__main__':main()
