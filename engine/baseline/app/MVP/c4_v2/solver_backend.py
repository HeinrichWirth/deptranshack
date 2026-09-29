"""Solver-only control: fixed C4 geometry with native candidate fitting."""
from .native_api import startup
from MVP.performance_final.engine import Engine as Base, clone
from MVP.performance_final2.batching import make_infer
import tracker
import numpy as np
from types import MethodType


def tracker_type(solver):
    class NativeTracker(tracker.TemplateTracker):
        def fit(self, p, center, gate, fixed=False):
            return self.fit_batch([(p,center,gate,fixed)])[0]

        def fit_batch(self, packs):
            values=solver.fit_batch([(np.asarray(p),np.asarray(c,dtype=float),float(g),bool(rest[0]) if rest else False)
                                    for p,c,g,*rest in packs], self.template, self.cfg['min_support'])
            for candidates,_ in values:
                for candidate in candidates: candidate['anchor']=np.asarray(candidate['anchor'])
            return values
    return NativeTracker


class SolverEngine(Base):
    def __init__(self, threads=1, iterations=8, grid=0):
        super().__init__('spatial')
        self.solver=startup().Solver(threads,iterations,grid)
        self.process_frame=MethodType(clone(self.process_frame.__func__,infer=make_infer(tracker_type(self.solver))),self)

    def close(self):
        pass
