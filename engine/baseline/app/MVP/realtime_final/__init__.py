from MVP.final_pipeline import ROOT
from pathlib import Path
import os
OUT=Path(os.environ.get('REALTIME_OUTPUT',str(ROOT/'results_realtime_final')))
if os.environ.get('REALTIME_DATA'):
    import long_common as lc
    lc.dataset=lambda:Path(os.environ['REALTIME_DATA'])
