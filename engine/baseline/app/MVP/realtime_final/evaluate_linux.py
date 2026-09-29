"""Use the identical frozen GT caches after all Linux inference is complete."""
from . import ROOT,OUT
from MVP.full_native_final import evaluate as ev
import long_common as lc
import shutil

def main():
    root=OUT/'linux_baseline';assert (root/'INFERENCE_COMPLETE.json').exists()
    dest=root/'contact_gt';dest.mkdir(exist_ok=True)
    for p in (ROOT/'results_full_native_final/contact_gt').glob('*.npz'):shutil.copy2(p,dest/p.name)
    ev.OUT=root;ev.main()
    shutil.copy2(root/'range_quality.csv',OUT/'linux_quality_240.csv')
    lc.save(OUT/'LINUX_BASELINE_QUALITY.json',lc.load(root/'QUALITY.json'))

if __name__=='__main__':main()
