"""Create an auditable local derivative; never edit/import-patch frozen STEP5."""
from fusion_common import *

def main():
    text=(PREVIOUS/'src/tracker.py').read_text(encoding='utf-8')
    replacements={
      'from geometry import binned,pca,transport,refine,scatter,angles':'from geometry import binned,pca,transport,refine,scatter,angles\nfrom weighting import balanced_weights,weighted_pca,weighted_binned,WeightedTemplateTracker',
      'def _engine(xyz,seed,template,cfg,oracle_orientation=None):':'def _engine(xyz,seed,template,cfg,oracle_orientation=None,source_frames=None,age_distance=None,tau=None):',
      "tt=TemplateTracker(template,seed['side'],cfg)":"tt=WeightedTemplateTracker(template,seed['side'],cfg)",
      "recent=support[su>=end-cfg['tail']]; clock=perf_counter()":"recent_ids=np.flatnonzero(seen)[su>=end-cfg['tail']]; recent=xyz[recent_ids]; clock=perf_counter()\n        weights=balanced_weights(source_frames[recent_ids],age_distance[recent_ids],tau)",
      "fitpoints=recent if cfg['method']=='M1' else binned(recent,B[:,0],cfg['bin_size'])\n            t,center=pca(fitpoints,B[:,0],robust=cfg['method']!='M1')":"fitpoints=recent if cfg['method']=='M1' else weighted_binned(recent,B[:,0],cfg['bin_size'],weights)\n            t,center=weighted_pca(fitpoints,weights,B[:,0]) if cfg['method']=='M1' else pca(fitpoints,B[:,0],robust=True)",
      'oc,od=tt.fit(overlap,np.zeros(2),.12)':'oc,od=tt.fit(overlap,np.zeros(2),.12,source_frames=source_frames[overlap_ids],age_distance=age_distance[overlap_ids],tau=tau)',
      "nc,nd=tt.fit(uvw[new_ids,1:],oa,cfg['gate'],fixed=cfg['anchor_mode']=='fixed')":"nc,nd=tt.fit(uvw[new_ids,1:],oa,cfg['gate'],fixed=cfg['anchor_mode']=='fixed',source_frames=source_frames[new_ids],age_distance=age_distance[new_ids],tau=tau)"
    }
    for a,b in replacements.items():
        assert text.count(a)==1,a;text=text.replace(a,b)
    text+='\n\ndef predict_weighted(xyz,seed,template,cfg,source_frames,age_distance,tau=None):\n    return _engine(xyz,seed,template,cfg,None,source_frames,age_distance,tau)\n'
    (STAGE/'src/weighted_tracker.py').write_text(text,encoding='utf-8')
    save(OUT/'audit/weighted_adapter.json',dict(source_sha256=sha(PREVIOUS/'src/tracker.py'),replacements=replacements,note='Only weighted PCA/centroids and template fit statistics differ. Acceptance gates, source-point selection, overlap quantile thresholds and forward-only marching remain identical.'))
if __name__=='__main__':main()
