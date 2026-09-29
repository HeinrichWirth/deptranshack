from . import ROOT,OUT
import long_common as lc
import time


def main():
    files={}
    for base in (ROOT/'MVP/c4_v2',OUT):
        for p in base.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ('build','docker_context','__pycache__')):continue
            if p.name=='MANIFEST.json' or p.suffix in ('.prof',):continue
            files[str(p.relative_to(ROOT)).replace('\\','/')]=dict(sha256=lc.sha(p),bytes=p.stat().st_size)
    quality=lc.load(OUT/'quality_gate.json');performance=lc.load(OUT/'performance_gate.json')
    lc.save(OUT/'MANIFEST.json',dict(created_ns=time.time_ns(),production_default='reference',production_freeze=False,
        quality_pass=quality['pass'],performance_pass=performance['pass'],selection=lc.load(OUT/'SELECTION_LOCK.json'),
        frozen_integrity=lc.load(OUT/'frozen_integrity.json')['unchanged'],files=files))
    assert not (ROOT/'MVP/c4_v2/FINAL_C4_V2.freeze').exists(),'Do not leave a production freeze for a failed experiment'
    print('MANIFEST',len(files),flush=True)

if __name__=='__main__':main()
