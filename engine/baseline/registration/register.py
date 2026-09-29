"""Lidar-only local odometry. Point-to-plane ICP; no assumed train speed."""
import bootstrap
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from cloud import coordinates, decode, masks
from prepare import readonly, write_json

KD_WORKERS=1


def voxel_sample(xyz,size=.22):
    keys=np.floor(xyz/size).astype(np.int32)
    _,ids=np.unique(keys,axis=0,return_index=True)
    return xyz[ids].astype(np.float64)


def load_scan(bag,index,size=.22):
    frame=bag['frames'][index]
    with readonly(bootstrap.ROOT/frame['db']) as db:
        _,points=decode(db.execute('SELECT data FROM messages WHERE id=?',(frame['message_id'],)).fetchone()[0])
    xyz=coordinates(points)
    finite,zero=masks(xyz)
    xyz=xyz[finite&~zero]
    ranges=np.linalg.norm(xyz,axis=1)
    xyz=xyz[(ranges>1.2)&(ranges<140)]
    return voxel_sample(xyz,size)


def normals(points,k=16):
    tree=cKDTree(points)
    distances,ids=tree.query(points,k=min(k,len(points)),workers=KD_WORKERS)
    neighbors=points[ids]
    centered=neighbors-neighbors.mean(axis=1,keepdims=True)
    cov=np.einsum('nki,nkj->nij',centered,centered)
    values,vectors=np.linalg.eigh(cov)
    normal=vectors[:,:,0]
    good=(values[:,0]/np.maximum(values.sum(axis=1),1e-12)<.12)&(distances[:,-1]<3)
    return normal,good


def transform(xyz,pose):
    return xyz@pose[:3,:3].T+pose[:3,3]


def icp(source,target,normal,initial=None,iterations=24,max_distance=1.8):
    pose=np.eye(4) if initial is None else initial.copy()
    tree=cKDTree(target)
    # A bounded, reproducible sample keeps full-resolution export independent of ICP cost.
    if len(source)>12000:
        source=source[np.linspace(0,len(source)-1,12000,dtype=np.int64)]
    for iteration in range(iterations):
        moved=transform(source,pose)
        distances,ids=tree.query(moved,workers=KD_WORKERS)
        limit=max(.35,max_distance*(1-iteration/max(iterations,1)))
        keep=distances<limit
        if keep.sum()<100:
            raise ValueError('Too little overlap for ICP')
        p=moved[keep];q=target[ids[keep]];n=normal[ids[keep]]
        residual=np.einsum('ij,ij->i',p-q,n)
        weight=np.minimum(1.,.10/np.maximum(np.abs(residual),1e-10))
        jac=np.column_stack([np.cross(p,n),n])
        # Normalize columns to avoid treating large angular lever arms as translation certainty.
        scale=np.sqrt(np.maximum(np.mean(jac*jac,axis=0),1e-10))
        a=jac/scale
        h=(a*weight[:,None]).T@a
        rhs=-(a*weight[:,None]).T@residual
        step=np.linalg.lstsq(h+np.eye(6)*1e-6,rhs,rcond=1e-8)[0]/scale
        delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(step[:3]).as_matrix();delta[:3,3]=step[3:]
        pose=delta@pose
        if np.linalg.norm(step[:3])<2e-5 and np.linalg.norm(step[3:])<.0008:
            break
    moved=transform(source,pose)
    distance,ids=tree.query(moved,workers=KD_WORKERS)
    keep=distance<.5
    residual=np.einsum('ij,ij->i',moved[keep]-target[ids[keep]],normal[ids[keep]])
    eig=np.linalg.eigvalsh(h)
    metric=dict(overlap=float(keep.mean()),rmse_plane_m=float(np.sqrt(np.mean(residual**2))),
                median_nn_m=float(np.median(distance)),condition=float(eig[-1]/max(eig[0],1e-12)),iterations=iteration+1)
    return pose,metric


def register_bag(bag,limit=None):
    scans=bag['frame_count'] if limit is None else min(limit,bag['frame_count'])
    first=load_scan(bag,0)
    normal,good=normals(first)
    reference=first[good];reference_normal=normal[good]
    pose=np.eye(4);velocity=np.eye(4);reference_pose=np.eye(4);reference_index=0
    entries=[dict(index=0,pose=pose.tolist(),status='origin',overlap=1.,rmse_plane_m=0.,condition=1.)]
    started=time.monotonic()
    for i in range(1,scans):
        source=load_scan(bag,i)
        previous_pose=pose.copy()
        initial=np.linalg.inv(reference_pose)@pose@velocity
        candidates=[]
        if i==1:
            # Symmetric alternatives do not assume the direction/speed of train motion.
            for shift in (0.,-.8,-1.6,.8):
                init=np.eye(4);init[1,3]=shift
                candidates.append(icp(source,reference,reference_normal,init))
            relative,metric=min(candidates,key=lambda item:item[1]['rmse_plane_m']+(.8-item[1]['overlap'])*.1)
        else:
            relative,metric=icp(source,reference,reference_normal,initial)
        pose=reference_pose@relative
        velocity=np.linalg.inv(previous_pose)@pose
        translation=float(np.linalg.norm(velocity[:3,3]))
        angle=float(np.linalg.norm(Rotation.from_matrix(velocity[:3,:3]).as_rotvec()))
        status='ok'
        if metric['overlap']<.45 or metric['rmse_plane_m']>.22 or translation>5 or angle>.1:
            # Do not fill uncertain gaps with invented motion. A failed match needs review.
            status='review'
        elif metric['condition']>3000:
            status='weak_geometry'
        entries.append(dict(index=i,pose=pose.tolist(),reference_index=reference_index,status=status,step_m=translation,rotation_rad=angle,**metric))
        if np.linalg.norm(relative[:3,3])>4 or np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec())>.035:
            normal,good=normals(source)
            reference=source[good];reference_normal=normal[good]
            reference_pose=pose.copy();reference_index=i
        if i%20==0 or i==scans-1:
            print(f'{bag["name"]}: {i+1}/{scans} position={np.round(pose[:3,3],2)} overlap={metric["overlap"]:.3f} rmse={metric["rmse_plane_m"]:.3f} {status}; {time.monotonic()-started:.1f}s',flush=True)
    xyz=np.array([e['pose'] for e in entries])[:,:3,3]
    return dict(bag_id=bag['id'],name=bag['name'],method='Keyframe robust point-to-plane ICP, lidar only',
                coordinate_frame='first lidar frame of this recording',voxel_m=.22,range_m=[1.2,140],
                keyframe_translation_m=4,keyframe_rotation_rad=.035,loop_closure=False,deskew=False,ground_truth_available=False,
                trajectory_length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
                final_position_m=xyz[-1].tolist(),frames=entries)


def run_one(bag,limit=None):
    out=bootstrap.ROOT/'output/registration';out.mkdir(exist_ok=True)
    suffix='_pilot' if limit else ''
    result=register_bag(bag,limit)
    write_json(out/(bag['name']+suffix+'.json'),result)
    return bag['name']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bag')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    manifest=json.loads((bootstrap.ROOT/'output/manifest.json').read_text(encoding='utf-8'))
    bags=[b for b in manifest['bags'] if not args.bag or args.bag==b['name']]
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(run_one,bag,args.limit) for bag in bags]
            for job in as_completed(jobs):
                print('Finished '+job.result(),flush=True)
    else:
        for bag in bags:
            run_one(bag,args.limit)
