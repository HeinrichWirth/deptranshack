from common import *
from bootstrap import initial_frame,pose_record

def main():
    initialize();save(OUT/'audit/frozen_before.json',locks());rows=[]
    for run in sorted(p.name for p in dataset().iterdir() if p.is_dir()):
        mm=frames(run);reasons={};valid=0
        for i,m in enumerate(mm):
            b,r=initial_frame(pose_record(m),pose_record(mm[i+1]) if i+1<len(mm) else None)
            if b is not None:valid+=1
            else:reasons[r]=reasons.get(r,0)+1
        rows.append(dict(run=run,frames=len(mm),eligible_two_pose=valid,reasons=reasons,bytes=sum(m['bytes'] for m in mm)))
    save(OUT/'audit/inventory.json',dict(data_root=str(dataset()),runs=rows,total_frames=sum(r['frames'] for r in rows)))
    for r in rows:print(r,flush=True)
if __name__=='__main__':main()
