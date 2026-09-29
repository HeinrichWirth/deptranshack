"""Opt-in offline registration across short occlusions; never hides inferred poses.

Interpolated map poses use a later, observed recovery endpoint. They are explicitly
marked and forbidden as input to the causal trailing-window rail predictor.
"""
import bootstrap
import argparse
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree
from register import load_scan, normals, icp, transform
from prepare import write_json


def interpolate_bridge(left, right, times):
    alpha=(np.asarray(times)-times[0])/(times[-1]-times[0])
    out=np.repeat(np.eye(4)[None],len(times),axis=0)
    out[:,:3,:3]=Slerp([0,1],Rotation.from_matrix([left[:3,:3],right[:3,:3]]))(alpha).as_matrix()
    out[:,:3,3]=left[:3,3]+alpha[:,None]*(right[:3,3]-left[:3,3])
    return out


def recovery_icp(source,target,normal,initial,prior,iterations=60):
    """Motion prior constrains poorly observed roll/slide; explicit estimated pose."""
    tree=cKDTree(target);pose=initial.copy();weights=np.array([10.,10.,10.,.1,.1,.1])**2
    if len(source)>12000:source=source[np.linspace(0,len(source)-1,12000,dtype=int)]
    for iteration in range(iterations):
        moved=transform(source,pose);distance,ids=tree.query(moved);keep=distance<max(.35,1.8*(1-iteration/iterations))
        if keep.sum()<100:raise ValueError('Too little overlap')
        p=moved[keep];n=normal[ids[keep]];residual=np.einsum('ij,ij->i',p-target[ids[keep]],n)
        w=np.minimum(1.,.1/np.maximum(np.abs(residual),1e-10));w/=w.sum()
        jac=np.column_stack([np.cross(p-pose[:3,3],n),n])
        h=(jac*w[:,None]).T@jac
        offset=np.r_[Rotation.from_matrix(pose[:3,:3]@prior[:3,:3].T).as_rotvec(),pose[:3,3]-prior[:3,3]]
        step=np.linalg.solve(h+np.diag(weights),-(jac*w[:,None]).T@residual-weights*offset)
        pose[:3,:3]=Rotation.from_rotvec(step[:3]).as_matrix()@pose[:3,:3];pose[:3,3]+=step[3:]
        if np.linalg.norm(step)<1e-5:break
    moved=transform(source,pose);distance,ids=tree.query(moved);keep=distance<.5
    residual=np.einsum('ij,ij->i',moved[keep]-target[ids[keep]],normal[ids[keep]])
    eig=np.linalg.eigvalsh(h+np.diag(weights))
    return pose,dict(overlap=float(keep.mean()),rmse_plane_m=float(np.sqrt(np.mean(residual**2))),
        median_nn_m=float(np.median(distance)),condition=float(eig[-1]/eig[0]),iterations=iteration+1,
        motion_prior=True,prior_rotation_scale_rad=.01,prior_translation_scale_m=1.)


def recover_endpoint(source, predicted, references):
    """Require stability to independent initial offsets, falling back to older geometry."""
    for target,normal,reference_pose,reference_index in reversed(references):
        initial=np.linalg.solve(reference_pose,predicted);trials=[]
        try:
            for dx,dy,yaw in [(0,0,0),(.25,0,0),(-.25,0,0),(0,.5,.005),(0,-.5,-.005)]:
                seed=initial.copy();seed[:3,3]+=[dx,dy,0]
                seed[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@seed[:3,:3]
                relative,metric=recovery_icp(source,target,normal,seed,initial,iterations=60)
                if metric['overlap']<.45 or metric['rmse_plane_m']>.22:
                    break
                trials.append((relative,metric))
            if len(trials)!=5:continue
            center=trials[0][0]
            shift=max(float(np.linalg.norm(t[:3,3]-center[:3,3])) for t,_ in trials)
            angle=max(float(np.linalg.norm(Rotation.from_matrix(center[:3,:3].T@t[:3,:3]).as_rotvec())) for t,_ in trials)
            if shift>.1 or angle>.01:continue
            relative,metric=min(trials,key=lambda item:item[1]['rmse_plane_m'])
            metric.update(recovery_translation_spread_m=shift,recovery_rotation_spread_rad=angle,recovery_initializations=5)
            return relative,metric,target,normal,reference_pose,reference_index
        except ValueError:
            continue
    raise ValueError('No stable recovery across initializations and earlier references')


def register_occlusion(bag, limit=None, max_occlusion_seconds=2.):
    count=min(limit or bag['frame_count'],bag['frame_count'])
    times=np.array([int(f['header_time_ns']) for f in bag['frames'][:count]],dtype=np.int64)
    times=(times-times[0])/1e9;dt=np.diff(times)
    if len(dt) and (np.any(dt<=0) or np.any(dt>1.5*np.median(dt))):
        raise ValueError('Time gaps must not be bridged by the occlusion estimator')
    first=load_scan(bag,0);ns,good=normals(first)
    reference=first[good];reference_normal=ns[good];reference_pose=np.eye(4);reference_index=0
    references=[(reference,reference_normal,reference_pose.copy(),reference_index)]
    pose=np.eye(4);last=0;velocity=np.zeros(3);omega=np.zeros(3);history=[];bridges=[]
    entries=[dict(index=0,pose=pose.tolist(),status='origin',overlap=1.,rmse_plane_m=0.,condition=1.)]
    support_history=[(len(first),float(np.ptp(first[:,1])))]
    started=time.monotonic()
    for i in range(1,count):
        source=load_scan(bag,i);elapsed=times[i]-times[last]
        if i>last+1 and elapsed>max_occlusion_seconds:
            raise ValueError(f'Unresolved occlusion ending at frame {i+1}')
        predicted=pose.copy();predicted[:3,3]+=velocity*elapsed
        predicted[:3,:3]=pose[:3,:3]@Rotation.from_rotvec(omega*elapsed).as_matrix()
        reason=None;metric={};candidate=None
        # Preserve a useful corridor reference before a near blocker consumes it.
        baseline=np.max(support_history[-50:],axis=0)
        if (len(source)<max(1000,.65*baseline[0]) or
                np.ptp(source[:,1])<max(15,.5*baseline[1])):
            reason='insufficient_spatial_support'
        else:
            initial=np.linalg.solve(reference_pose,predicted)
            try:
                if i>last+1:
                    relative,metric,reference,reference_normal,reference_pose,reference_index=recover_endpoint(source,predicted,references)
                else:
                    relative,metric=icp(source,reference,reference_normal,initial)
                candidate=reference_pose@relative
                error=float(np.linalg.norm(candidate[:3,3]-predicted[:3,3]))
                angle=float(np.linalg.norm(Rotation.from_matrix(predicted[:3,:3].T@candidate[:3,:3]).as_rotvec()))
                recovered_support=False
                if i>last+1:
                    moved=transform(source,relative)
                    distance,_=cKDTree(moved).query(reference)
                    supported=reference[distance<.5]
                    reverse=float(np.mean(distance<.5));metric['reference_overlap']=reverse
                    metric['prediction_error_m']=error;metric['prediction_angle_rad']=angle
                    recovered_support=(reverse>=.65 and len(supported)>=500 and
                                       np.ptp(supported[:,0])>=2.5 and np.ptp(supported[:,1])>=8 and metric['overlap']>=.15)
                if (metric['overlap']<.45 and not recovered_support) or metric['rmse_plane_m']>.22 or (not metric.get('motion_prior') and metric['condition']>3000):
                    reason='weak_registration'
                elif len(history)>=3 and (error>max(.5,elapsed*1.5) or angle>max(.025,elapsed*.06)):
                    reason='inconsistent_motion'
            except ValueError:
                reason='no_icp_overlap'
        if reason:
            entries.append(dict(index=i,pose=predicted.tolist(),status='pending',reason=reason,
                                reference_index=reference_index,**metric))
            print('OCCLUDED',i+1,reason,'points',len(source),'metrics',metric,flush=True)
            if elapsed>max_occlusion_seconds:raise ValueError(f'Unresolved occlusion ending at frame {i+1}')
            continue
        if i>last+1:
            bridge=dict(left_index=last,right_index=i,duration_seconds=float(elapsed),
                        endpoint_prediction_error_m=error,endpoint_prediction_angle_rad=angle,
                        recovery_reference_index=reference_index,recovery_metrics=metric)
            poses=interpolate_bridge(pose,candidate,times[last:i+1])
            for j in range(last+1,i):
                entries[j].update(pose=poses[j-last].tolist(),status='occlusion_interpolated',
                                  uses_future_pose=True,bridge_left_index=last,bridge_right_index=i,
                                  pose_method='offline linear translation / SLERP between observed endpoints')
            bridges.append(bridge);print('RECOVERED',bridge,flush=True)
        difference=(candidate[:3,3]-pose[:3,3])/elapsed
        turn=Rotation.from_matrix(pose[:3,:3].T@candidate[:3,:3]).as_rotvec()/elapsed
        history.append((difference,turn));history=history[-5:]
        velocity=np.median([h[0] for h in history],axis=0);omega=np.median([h[1] for h in history],axis=0)
        pose=candidate;last=i
        support_history.append((len(source),float(np.ptp(source[:,1]))))
        entries.append(dict(index=i,pose=pose.tolist(),status='ok',reference_index=reference_index,**metric))
        if np.linalg.norm(relative[:3,3])>4 or np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec())>.035:
            ns,good=normals(source);reference=source[good];reference_normal=ns[good];reference_pose=pose.copy();reference_index=i
            references.append((reference,reference_normal,reference_pose.copy(),reference_index));references=references[-12:]
        if i%50==0 or i==count-1:print(bag['name'],i+1,count,np.round(pose[:3,3],2),'seconds',round(time.monotonic()-started,1),flush=True)
    if last!=count-1:raise ValueError('Recording ends during occlusion; no observed recovery endpoint')
    positions=np.asarray([f['pose'] for f in entries])[:,:3,3]
    return dict(bag_id=bag['id'],name=bag['name'],method='Keyframe point-to-plane ICP with explicit short-occlusion recovery',
                coordinate_frame='first lidar frame of this recording',voxel_m=.22,range_m=[1.2,140],
                loop_closure=False,deskew=False,ground_truth_available=False,occlusion_bridges=bridges,
                max_occlusion_seconds=max_occlusion_seconds,offline_interpolation=True,
                trajectory_length_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
                final_position_m=positions[-1].tolist(),frames=entries)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bag',required=True);parser.add_argument('--limit',type=int);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    bag=next(b for b in json.loads((bootstrap.ROOT/'output/manifest.json').read_text(encoding='utf8'))['bags'] if b['name']==args.bag)
    write_json(bootstrap.ROOT/args.output,register_occlusion(bag,args.limit))
