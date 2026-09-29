"""Rebuild offline figures with separately measured wall/CPU time."""
from . import ROOT
import time
import long_common as lc


def main():
    started=time.perf_counter();cpu=time.process_time()
    from .visualizations import main as cases
    from .overviews import main as overviews
    cases();overviews()
    lc.save(ROOT/'results_final_pipeline/diagnostics_timing.json',dict(
        T_DIAGNOSTICS=time.perf_counter()-started,CPU_DIAGNOSTICS=time.process_time()-cpu,
        scope='Offline full case gallery, two run overviews and performance figure; excludes inference and LAS export'))


if __name__=='__main__':main()
