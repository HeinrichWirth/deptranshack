"""Registration checkpoint adapter; real acquisition gaps remain explicit.

Numerical kernels are copied unchanged. Selected transport frames may be skipped;
accepted estimates remain distinct from extrapolation and timeout states.
"""
from rt_common import *
import ast
from causal_registration import CausalRegistration
from registration_acceleration import install,install_parallel_recovery
from direction_probe import fork

class RealtimeRegistration(CausalRegistration):
    pass

# Only the admission interval changes: source continuity is checked by the reader.
# Internal index enumerates attempted registrations, not the original bag frames.
_source=(HERE/'baseline/adapter/causal_registration.py').read_text(encoding='utf-8-sig')
_tree=ast.parse(_source);_class=next(n for n in _tree.body if isinstance(n,ast.ClassDef))
_method=next(n for n in _class.body if isinstance(n,ast.FunctionDef) and n.name=='_step')
class Interval(ast.NodeTransformer):
    def visit_Constant(self,node):return ast.copy_location(ast.Constant(2.0),node) if node.value==.3 else node
_method=Interval().visit(_method)
_scope=dict(CausalRegistration._step.__globals__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[_method],type_ignores=[])),__file__,'exec'),_scope)
RealtimeRegistration._step=_scope['_step']

class DeferredRecovery(Exception):
    def __init__(self,source,predicted):self.source=source;self.predicted=predicted
def capture(reg):return {k:v for k,v in reg.__dict__.items() if k not in ('fn','source_hashes','source_root')}
def restore(state):
    reg=RealtimeRegistration(HERE/'baseline/registration');reg.__dict__.update(state);return reg
def new():return RealtimeRegistration(HERE/'baseline/registration')
def defer(source,predicted,references):raise DeferredRecovery(source,predicted)

def provisional_checkpoint(reg,source,predicted,stamp):
    """Unpublished local motion chain; the anchor remains explicitly unconfirmed."""
    normal,good=reg.fn['normals'](source);reg.index+=1;reg.previous_stamp=stamp
    reg.reference=source[good];reg.normal=normal[good];reg.reference_pose=predicted.copy();reg.reference_index=reg.index
    reg.references=[(reg.reference,reg.normal,reg.reference_pose.copy(),reg.index)]
    reg.pose=predicted.copy();reg.last=reg.index;reg.last_stamp=stamp
    reg.support=reg.support+[(len(source),float(np.ptp(source[:,1])))]

def anchor_provisional(reg,delta):
    """Apply the observed recovery anchor to the unpublished relative chain."""
    reg.pose=delta@reg.pose;reg.reference_pose=delta@reg.reference_pose
    reg.references=[(p,n,delta@P,i) for p,n,P,i in reg.references]
    reg.velocity=delta[:3,:3]@reg.velocity
    reg.history=[(delta[:3,:3]@v,w) for v,w in reg.history]

def recovery_worker(ring,inbox,outbox,stop,ready,out,stress):
    logger=log_open(out,'recovery');injected=False;ready.set()
    while not stop.is_set():
        job=inbox.take()
        if job is None:time.sleep(.002);continue
        begin=time.perf_counter();item=ring.read(job['meta']['source_frame'])
        if item is None:
            outbox.put(dict(token=job['token'],status='INPUT_EXPIRED'));continue
        if stress and not injected:
            injected=True;log(logger,event='INJECT_RECOVERY_PAUSE',seconds=1,source_frame=job['meta']['source_frame'],clock=time.perf_counter());time.sleep(1.)
        meta,points=item;reg=restore(job['state'])
        try:
            row=reg.step(points[:,:3],reg.index+1,meta['header_time_ns'])
            result=dict(token=job['token'],status='COMPLETE',state=capture(reg),pose=row,meta=meta,
                        generation=job['generation'],ms=(time.perf_counter()-begin)*1000,completed=time.perf_counter())
        except ValueError as e:result=dict(token=job['token'],status='FAILED',reason=str(e),ms=(time.perf_counter()-begin)*1000)
        outbox.put(result);log(logger,event='RECOVERY_COMPLETE',token=job['token'],status=result['status'],source_frame=meta['source_frame'],ms=result['ms'],clock=time.perf_counter())
    logger.close()
