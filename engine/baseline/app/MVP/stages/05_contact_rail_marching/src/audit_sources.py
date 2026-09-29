"""Read-only bit integrity audit of every used original LAS point record."""
from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(task):
    run,i=task;row=frames(run)[i];path=dataset()/run/row['file']
    with laspy.open(path) as f:h=f.header
    raw=np.memmap(path,dtype=np.uint8,mode='r',offset=h.offset_to_point_data,shape=(h.point_count*h.point_format.size,))
    digest=hashlib.sha256(raw).hexdigest();assert digest==row['point_data_sha256'],str(path)
    return dict(run=run,frame=i,source_points=h.point_count,point_data_sha256=digest,bytes=path.stat().st_size,verified=True)

def main():
    co=load(OUT/'audit/cohort.json');tasks=sorted(set((r['run'],r['frame']) for r in co));rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,t) for t in tasks]):rows.append(f.result())
    csv_write(OUT/'audit/source_integrity.csv',rows);save(OUT/'audit/source_integrity.json',dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),all_original_point_hashes_match=True));print('SOURCE HASH PASS',len(rows),flush=True)
if __name__=='__main__':main()
