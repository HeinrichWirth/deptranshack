"""Dispatch gate only; inherited arrival still indexes EVERY raw frame."""
from MVP.realtime_final.live import Scheduler


class StrideScheduler(Scheduler):
    def __init__(self, directory, workers, threads, stride=1, offset=0):
        if stride < 1 or not 0 <= offset < stride:
            raise ValueError('Expected stride >= 1 and 0 <= offset < stride')
        self.solve_stride = stride
        self.solve_offset = offset
        super().__init__(directory, workers=workers, threads=threads)

    def eligible(self, i):
        return i % self.solve_stride == self.solve_offset and super().eligible(i)
