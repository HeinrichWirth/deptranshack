from .native_api import startup
from . import OUT
import long_common as lc
import tracker
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from collections import OrderedDict
import copy,time,hashlib
from pathlib import Path

def factory(mode,workers=1,audit=None):
    native=startup();limits=lc.load(Path(__file__).with_name('runtime_config.json'))['thresholds']
    residual_mode=mode not in ('h1','h2','h3')
    jac_mode=mode not in ('h1','h2','h3','residual')
    tree_reuse=mode!='h1';reverse_native=mode not in ('h1','h2')
    class Tracker(tracker.TemplateTracker):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.memo=OrderedDict();self.trees=OrderedDict();self.next_fit=0
            self.pool=native.Pool(workers) if workers>1 else None
            self.stats=dict(fit_calls=0,memo_hits=0,tree_requests=0,tree_hits=0,tree_builds=1,tree_hash_seconds=0.,
                optimizer_calls=0,native_fun_calls=0,native_jac_calls=0,native_distance_calls=0,batch_calls=0,workspace_peak=0)
            if audit is not None:audit.append(self)
        def close(self):
            if self.pool:self.pool.close();self.pool=None
        def distances(self,p,anchor):
            if len(self.template)<limits['forward']:
                self.stats['native_distance_calls']+=1
                return native.exact_nn_bruteforce(p-anchor,self.template)
            return self.tree.query(p-anchor)[0]
        def reverse(self,points,anchor):
            if reverse_native and len(points)<limits['reverse']:
                self.stats['native_distance_calls']+=1
                return native.exact_nn_bruteforce(self.template[::2]+anchor,points)
            self.stats['tree_requests']+=1;t=time.perf_counter();key=(points.shape,points.tobytes()) if tree_reuse else id(points)
            self.stats['tree_hash_seconds']+=time.perf_counter()-t
            if key in self.trees:self.stats['tree_hits']+=1;tree=self.trees[key][1]
            else:
                tree=cKDTree(points);self.stats['tree_builds']+=1;self.trees[key]=(points,tree)
                if len(self.trees)>128:self.trees.popitem(last=False)
            return tree.query(self.template[::2]+anchor)[0]
        def prepare(self,p,center,gate,fixed=False):
            self.stats['fit_calls']+=1;self.next_fit+=1;center=np.asarray(center);p=np.asarray(p)
            key=(p.shape,p.dtype.str,p.tobytes(),center.astype(float).tobytes(),float(gate),fixed)
            if key in self.memo:self.stats['memo_hits']+=1;return dict(done=copy.deepcopy(self.memo[key]))
            region=np.all((p-center>=self.lo-gate)&(p-center<=self.hi+gate),axis=1);points=np.unique(p[region],axis=0)
            if len(points)<self.cfg['min_support']:return dict(done=([],dict(candidate_n=len(points))))
            _,ix=np.unique(np.floor(points/.002).astype(np.int32),axis=0,return_index=True);fitp=points[ix]
            seeds=[center] if fixed else [center,center+[gate*.7,0],center+[-gate*.7,0],center+[0,gate*.7],center+[0,-gate*.7]]
            lower=center-gate;upper=center+gate
            jobs=[dict(points=points,fitp=fitp,center=center,gate=gate,seed=seed,fixed=fixed,lower=lower,upper=upper) for seed in seeds]
            return dict(jobs=jobs,points=points,key=key)
        def one(self,job):
            points=job['points'];fitp=job['fitp'];center=job['center'];gate=job['gate']
            if job['fixed']:anchor=center;success=True
            else:
                problem=None;kw={}
                if residual_mode:
                    problem=native.Problem(fitp,self.template,job['lower'],job['upper'],mode=='soa')
                    fun=problem.fun
                    if jac_mode:kw['jac']=problem.jac
                else:fun=lambda a:self.distances(fitp,a)/.015
                opt=least_squares(fun,np.clip(job['seed'],center-gate+1e-9,center+gate-1e-9),bounds=(job['lower'],job['upper']),
                    loss='cauchy',f_scale=1.,diff_step=1e-3,max_nfev=24,**kw)
                self.stats['optimizer_calls']+=1
                if problem:
                    self.stats['native_fun_calls']+=problem.fun_calls;self.stats['native_jac_calls']+=problem.jac_calls
                    self.stats['workspace_peak']=max(self.stats['workspace_peak'],problem.bytes())
                anchor=opt.x;success=bool(opt.success)
            d=self.distances(points,anchor);inside=np.all((points-anchor>=self.lo)&(points-anchor<=self.hi),axis=1)
            support=inside&(d<=self.cfg['support_tolerance']);n=int(support.sum())
            if n:
                err=float(np.mean(np.sort(d[inside])[:max(1,int(.7*inside.sum()))]));reverse=self.reverse(points,anchor)
                coverage=float(np.mean(reverse<=.025));score=float(np.exp(-err/.015)+.25*coverage+.1*(1-np.exp(-n/8)))
            else:err=.5;coverage=0.;score=-1.
            return dict(anchor=anchor,score=score,residual=err,support_unique=n,coverage=coverage,success=success)
        def finish(self,prepared,values):
            if 'done' in prepared:return prepared['done']
            candidates=[]
            for c in values:
                if not any(np.linalg.norm(c['anchor']-d['anchor'])<.006 for d in candidates):candidates.append(c)
            candidates.sort(key=lambda c:-c['score']);result=(candidates,dict(candidate_n=len(prepared['points'])))
            self.memo[prepared['key']]=copy.deepcopy(result)
            if len(self.memo)>256:self.memo.popitem(last=False)
            return result
        def fit(self,p,center,gate,fixed=False):
            prepared=self.prepare(p,center,gate,fixed)
            if 'done' in prepared:return prepared['done']
            return self.finish(prepared,[self.one(j) for j in prepared['jobs']])
        def fit_batch(self,packs):
            prepared=[self.prepare(*p) for p in packs];jobs=[j for p in prepared for j in p.get('jobs',[])];self.stats['batch_calls']+=1
            values=self.pool.map(self.one,jobs) if self.pool else native.native_evaluate_candidate_batch(self.one,jobs)
            out=[];cursor=0
            for p in prepared:
                n=len(p.get('jobs',[]));out.append(self.finish(p,values[cursor:cursor+n]));cursor+=n
            return out
    return Tracker
