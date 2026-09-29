import shutil,time,hashlib,json
from pathlib import Path
root=Path('/work');src=Path('/source/cloud_with_fake_obj_0.db3');dst=Path('/tmp/input.db3');t=time.perf_counter()
shutil.copyfile(src,dst);copy_seconds=time.perf_counter()-t
h=hashlib.sha256()
with dst.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()=='73ee5bfc2087bb160267110563b298a9629ac9e3df1459d857ec72ce1ef89d60'
(root/'results').mkdir(exist_ok=True)
(root/'results/INPUT_COPY.json').write_text(json.dumps(dict(copy_seconds=copy_seconds,sha256=h.hexdigest(),bytes=dst.stat().st_size,path=str(dst),warm_cache=True),indent=2)+'\n')
print('INPUT_READY',copy_seconds,flush=True)
time.sleep(10800)
