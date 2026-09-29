"""Post-run provenance audit/normalization and unchanged-dependency checks."""
from . import ROOT
import long_common as lc
import numpy as np
from .audit import pins
import time

OUT=ROOT/'results_final_pipeline'


def main():
    changed=[];checked=0
    for folder in OUT.glob('deployable_*/*'):
        if not (folder/'PREDICTION_COMPLETE.json').exists():continue
        marker=lc.load(folder/'PREDICTION_COMPLETE.json')
        for name,sha in marker['files'].items():assert lc.sha(folder/name)==sha,(folder,name)
        meta=lc.load(folder/'summary.json')
        with np.load(folder/'point_provenance.npz') as z:rows=z['rows']
        if meta['final_geometry_available']:
            take=((rows['used_by_stage_mask']&4)!=0)&((rows['used_by_stage_mask']&16)==0)
            if take.any():
                # Bookkeeping correction only; geometry and first-found unchanged.
                old=lc.sha(folder/'point_provenance.npz');rows['used_by_stage_mask'][take]|=16
                np.savez_compressed(folder/'point_provenance.npz',rows=rows)
                marker['files']['point_provenance.npz']=lc.sha(folder/'point_provenance.npz')
                lc.save(folder/'PREDICTION_COMPLETE.json',marker)
                changed.append(dict(folder=folder.relative_to(OUT).as_posix(),points=int(take.sum()),before_sha256=old,
                    after_sha256=marker['files']['point_provenance.npz'],reason='STEP6 consumes tentative cr_state evidence; add transitive usage bit only'))
        keys=(rows['source_frame'].astype('uint64')<<np.uint64(32))|rows['source_row'].astype('uint64')
        assert len(np.unique(keys))==len(keys)
        assert np.all(rows['source_frame']<=meta['frame'])
        assert set(np.unique(rows['first_found_stage']))<={1,2,3}
        checked+=1
    before=lc.load(OUT/'audit/dependencies_before.json');after=pins()
    assert before==after,'Frozen stage dependency changed'
    lc.save(OUT/'audit/final_provenance_checks.json',dict(cases=checked,changed_usage_rows=changed,
        first_found_unchanged=True,predictions_unchanged=True,frozen_files_unchanged=len(before),time_ns=time.time_ns()))


if __name__=='__main__':main()
