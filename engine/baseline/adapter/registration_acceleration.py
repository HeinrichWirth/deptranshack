"""Exact-distance pruning and target-tree reuse for the existing ICP equations.

This optional adapter recompiles only nearest-neighbour query bounds. Thresholds,
samples, solvers, iterations and recovery acceptance remain in the source file.
"""
import ast
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial.transform import Rotation


class TreeCache:
    def __init__(self, limit=16, profile=False):
        self.limit=limit; self.profile=profile; self.cache=OrderedDict(); self.events=[]

    def __call__(self, data, *args, **kwargs):
        key=id(data); cached=self.cache.get(key)
        if cached is not None and cached[0] is data:
            self.cache.move_to_end(key); return cached[1]
        begin=time.perf_counter(); tree=cKDTree(data,*args,**kwargs)
        owner=self
        class MeasuredTree:
            def query(self, points, *qa, **qk):
                started=time.perf_counter()
                result=tree.query(points,*qa,**qk)
                if owner.profile:owner.events.append(dict(kind='query',ms=(time.perf_counter()-started)*1000,n=len(points),target=len(data),bounded='distance_upper_bound' in qk,k=qk.get('k',qa[0] if qa else 1)))
                return result
        wrapped=MeasuredTree()
        if self.profile:self.events.append(dict(kind='build',ms=(time.perf_counter()-begin)*1000,n=len(data)))
        self.cache[key]=(data,wrapped)
        while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return wrapped


class BoundQueries(ast.NodeTransformer):
    def __init__(self, expression):self.expression=expression;self.in_loop=0
    def visit_For(self,node):
        self.in_loop+=1; self.generic_visit(node); self.in_loop-=1; return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if self.in_loop and isinstance(node.func,ast.Attribute) and node.func.attr=='query':
            node.keywords.append(ast.keyword(arg='distance_upper_bound',value=ast.parse(self.expression,mode='eval').body))
        return node


def install(scope, root, bounded=True, profile=False, cache_size=16):
    cache=TreeCache(cache_size,profile);scope['cKDTree']=cache
    if bounded:
        for filename,name,expression in (
            ('register.py','icp','max(.35,max_distance*(1-iteration/max(iterations,1)))'),
            ('register_occlusion.py','recovery_icp','max(.35,1.8*(1-iteration/iterations))')):
            tree=ast.parse((Path(root)/filename).read_text(encoding='utf-8-sig'))
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            function=BoundQueries(expression).visit(function)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(Path(root)/filename),'exec'),scope)
    return cache


def install_parallel_recovery(scope, workers=5):
    """Evaluate unchanged recovery seeds concurrently, choose in original order."""
    def recover_endpoint(source,predicted,references):
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='recover-seed') as pool:
            for target,normal,reference_pose,reference_index in reversed(references):
                initial=np.linalg.solve(reference_pose,predicted)
                # Warm the shared read-only tree before starting concurrent queries.
                scope['cKDTree'](target)
                def evaluate(offset):
                    dx,dy,yaw=offset;seed=initial.copy();seed[:3,3]+=[dx,dy,0]
                    seed[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@seed[:3,:3]
                    try:return scope['recovery_icp'](source,target,normal,seed,initial,iterations=60)
                    except ValueError:return None
                trials=list(pool.map(evaluate,((0,0,0),(.25,0,0),(-.25,0,0),(0,.5,.005),(0,-.5,-.005))))
                if any(t is None or t[1]['overlap']<.45 or t[1]['rmse_plane_m']>.22 for t in trials):continue
                center=trials[0][0]
                shift=max(float(np.linalg.norm(t[:3,3]-center[:3,3])) for t,_ in trials)
                angle=max(float(np.linalg.norm(Rotation.from_matrix(center[:3,:3].T@t[:3,:3]).as_rotvec())) for t,_ in trials)
                if shift>.1 or angle>.01:continue
                relative,metric=min(trials,key=lambda item:item[1]['rmse_plane_m'])
                metric.update(recovery_translation_spread_m=shift,recovery_rotation_spread_rad=angle,recovery_initializations=5)
                return relative,metric,target,normal,reference_pose,reference_index
        raise ValueError('No stable recovery across initializations and earlier references')
    scope['recover_endpoint']=recover_endpoint
