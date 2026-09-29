"""Decode every scientific figure and every animation frame before packaging."""
from long_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
from PIL import Image

def main():
    records=[]
    files=sorted(p for area in ('gallery','animations','observability') for p in (OUT/area).rglob('*') if p.suffix.lower() in ('.png','.gif'))
    for path in files:
        with Image.open(path) as im:
            size=im.size;count=getattr(im,'n_frames',1)
            for i in range(count):im.seek(i);im.load()
        records.append(dict(path=path.relative_to(OUT).as_posix(),size=size,frames=count))
    save(OUT/'audit/media_verification.json',dict(status='PASS',images=len(records),animation_frames=sum(r['frames'] for r in records if r['path'].endswith('.gif')),files=records))
    print('MEDIA PASS',len(records),'images',sum(r['frames'] for r in records),'total frames',flush=True)

if __name__=='__main__':main()
