"""Online state adapter of the existing occlusion-aware ICP, without retroactive poses.

Numerical functions are loaded unchanged from the two original source files.
The old sequential transition is retained; only offline interpolation is omitted.
"""
import ast,hashlib,time,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def original_functions(root):
    scope=dict(np=np,cKDTree=cKDTree,Rotation=Rotation,KD_WORKERS=int(os.environ.get('COPY_KD_WORKERS','1')));hashes={}
    for file,names in [('register.py',{'voxel_sample','normals','transform','icp'}),('register_occlusion.py',{'recovery_icp','recover_endpoint'})]:
        path=Path(root)/file;data=path.read_bytes();hashes[file]=hashlib.sha256(data).hexdigest()
        tree=ast.parse(data.decode('utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope,hashes

class CausalRegistration:
    def __init__(self,source_root):
        self.source_root=source_root
        self.fn,self.source_hashes=original_functions(source_root);self.last=None;self.index=-1;self.previous_stamp=None
        self.last_timing_ms={}
        if os.environ.get('COPY_VOXEL')=='packed':
            from copy_voxel import voxel_sample
            self.fn['voxel_sample']=voxel_sample
        acceleration=os.environ.get('COPY_REGISTRATION_ACCEL','off')
        if acceleration!='off':
            from registration_acceleration import install,install_parallel_recovery
            if acceleration not in ('bounded','parallel'):raise ValueError('Unknown registration acceleration')
            install(self.fn,source_root,bounded=True)
            if acceleration=='parallel':install_parallel_recovery(self.fn)
        for name in ('voxel_sample','normals','icp','recover_endpoint'):
            original=self.fn[name]
            def measured(*args,_name=name,_fn=original,**kwargs):
                begin=time.perf_counter()
                try:return _fn(*args,**kwargs)
                finally:self.last_timing_ms[_name]=self.last_timing_ms.get(_name,0.)+(time.perf_counter()-begin)*1000
            self.fn[name]=measured
        self.pose=np.eye(4);self.velocity=np.zeros(3);self.omega=np.zeros(3);self.history=[];self.support=[];self.references=[]
    def step(self,xyz,index,stamp,update_reference=True):
        self.last_timing_ms={};begin=time.perf_counter()
        result=self._step(xyz,index,stamp,update_reference=update_reference)
        self.last_timing_ms['total']=(time.perf_counter()-begin)*1000
        self.last_timing_ms['other']=self.last_timing_ms['total']-sum(self.last_timing_ms.get(k,0.) for k in ('voxel_sample','normals','recover_endpoint' if 'recover_endpoint' in self.last_timing_ms else 'icp'))
        return result
    def _step(self,xyz,index,stamp,update_reference=True):
        if index!=self.index+1:raise ValueError('Registration must receive consecutive clouds')
        if self.previous_stamp is not None and not 0<(stamp-self.previous_stamp)/1e9<=.3:raise ValueError('Cannot register across time gap')
        self.index=index;self.previous_stamp=stamp;f=self.fn
        # Exactly the old load_scan range / voxel selection; inference keeps full raw points.
        finite=np.isfinite(xyz).all(axis=1);zero=np.all(xyz==0,axis=1);p=xyz[finite&~zero]
        ranges=np.linalg.norm(p,axis=1);source=f['voxel_sample'](p[(ranges>1.2)&(ranges<140)],.22)
        if self.last is None:
            if len(source)<100:raise ValueError('Insufficient initial registration geometry')
            normal,good=f['normals'](source);self.reference=source[good];self.normal=normal[good];self.reference_pose=np.eye(4);self.reference_index=index
            self.references=[(self.reference,self.normal,self.reference_pose.copy(),index)];self.last=index;self.last_stamp=stamp
            self.support=[(len(source),float(np.ptp(source[:,1])))]
            return dict(index=index,pose=self.pose.tolist(),status='origin',uses_future_pose=False,reference_index=index,available_after_frame=index,icp_points=len(source),overlap=1.,rmse_plane_m=0.)
        elapsed=(stamp-self.last_stamp)/1e9;predicted=self.pose.copy();predicted[:3,3]+=self.velocity*elapsed
        predicted[:3,:3]=self.pose[:3,:3]@Rotation.from_rotvec(self.omega*elapsed).as_matrix()
        reason=None;metric={};candidate=None;baseline=np.max(self.support[-50:],axis=0)
        if len(source)<max(1000,.65*baseline[0]) or np.ptp(source[:,1])<max(15,.5*baseline[1]):reason='insufficient_spatial_support'
        else:
            initial=np.linalg.solve(self.reference_pose,predicted)
            try:
                if index>self.last+1:
                    relative,metric,self.reference,self.normal,self.reference_pose,self.reference_index=f['recover_endpoint'](source,predicted,self.references)
                else:relative,metric=f['icp'](source,self.reference,self.normal,initial)
                candidate=self.reference_pose@relative
                error=float(np.linalg.norm(candidate[:3,3]-predicted[:3,3]));angle=float(np.linalg.norm(Rotation.from_matrix(predicted[:3,:3].T@candidate[:3,:3]).as_rotvec()))
                recovered=False
                if index>self.last+1:
                    moved=f['transform'](source,relative);distance,_=cKDTree(moved).query(self.reference);supported=self.reference[distance<.5]
                    reverse=float(np.mean(distance<.5));metric.update(reference_overlap=reverse,prediction_error_m=error,prediction_angle_rad=angle)
                    recovered=reverse>=.65 and len(supported)>=500 and np.ptp(supported[:,0])>=2.5 and np.ptp(supported[:,1])>=8 and metric['overlap']>=.15
                if (metric['overlap']<.45 and not recovered) or metric['rmse_plane_m']>.22 or (not metric.get('motion_prior') and metric['condition']>3000):reason='weak_registration'
                elif len(self.history)>=3 and (error>max(.5,elapsed*1.5) or angle>max(.025,elapsed*.06)):reason='inconsistent_motion'
            except ValueError:reason='no_icp_overlap'
        if reason:
            if elapsed>2.:raise ValueError(f'Unresolved occlusion ending at frame {index}; frozen registration limit exceeded')
            # Do not publish the motion prior as an accepted pose, or later fill this row.
            return dict(index=index,pose=None,status='unavailable',reason=reason,uses_future_pose=False,reference_index=self.reference_index,available_after_frame=index,icp_points=len(source),**metric)
        difference=(candidate[:3,3]-self.pose[:3,3])/elapsed;turn=Rotation.from_matrix(self.pose[:3,:3].T@candidate[:3,:3]).as_rotvec()/elapsed
        self.history.append((difference,turn));self.history=self.history[-5:]
        self.velocity=np.median([h[0] for h in self.history],axis=0);self.omega=np.median([h[1] for h in self.history],axis=0)
        recovered_from=self.last if index>self.last+1 else None
        self.pose=candidate;self.last=index;self.last_stamp=stamp;self.support.append((len(source),float(np.ptp(source[:,1]))))
        row=dict(index=index,pose=self.pose.tolist(),status='ok',reference_index=self.reference_index,uses_future_pose=False,available_after_frame=index,icp_points=len(source),recovery_after_frame=recovered_from,**metric)
        if update_reference and (np.linalg.norm(relative[:3,3])>4 or np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec())>.035):
            normal,good=f['normals'](source);self.reference=source[good];self.normal=normal[good];self.reference_pose=self.pose.copy();self.reference_index=index
            self.references.append((self.reference,self.normal,self.reference_pose.copy(),index));self.references=self.references[-12:]
        return row
