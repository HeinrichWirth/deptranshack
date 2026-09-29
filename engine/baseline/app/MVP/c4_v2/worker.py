"""Latest-value worker; no stale pending queue and no false fresh publication."""
from MVP.performance_final2.async_worker import Policy as FrozenPolicy
from MVP.performance_final2.engine import Engine as Reference
from .persistent import PersistentEngine
import threading,time


class Policy(FrozenPolicy):
    def complete(self,result,clock):
        if result['summary'].get('c4_update_status')=='UPDATE_FAILED_REUSE_PREVIOUS':
            # A reused map without fresh validating evidence is NOT new geometry.
            result=dict(result,prediction=None)
        return super().complete(result,clock)


class GeometryWorker:
    def __init__(self,directory,backend='reference',config=None):
        if backend not in ('reference','v2'):raise ValueError('backend reference|v2')
        self.engine=Reference('native_batch_spatial') if backend=='reference' else PersistentEngine(**(config or {}))
        self.engine.start_run(directory);self.policy=Policy(self.engine.records)
        self.cv=threading.Condition();self.stop=False;self.error=None
        self.thread=threading.Thread(target=self.work,name='C4V2-latest',daemon=True);self.thread.start()

    def on_frame(self,index):
        with self.cv:
            if self.error:raise RuntimeError(self.error)
            now=time.perf_counter();self.policy.arrival(index,now);self.policy.offer(index-1,index,now)
            value=self.policy.snapshot(index);self.cv.notify();return value

    def work(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.stop or self.policy.pending_latest_frame is not None)
                if self.stop:return
                request=self.policy.start(time.perf_counter())
            if request is None:continue
            try:result=self.engine.process_frame(request['frame'])
            except Exception as e:
                with self.cv:self.error=repr(e);self.stop=True;self.cv.notify_all()
                return
            with self.cv:self.policy.complete(result,time.perf_counter());self.cv.notify_all()

    def close(self):
        with self.cv:self.stop=True;self.cv.notify_all()
        self.thread.join();self.engine.close()
        if self.error:raise RuntimeError(self.error)
