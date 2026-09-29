"""Final artifact, causal provenance, baseline equivalence and freeze audit."""
from fusion_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
from PIL import Image
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import ast

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ('href','src'):self.links.append(value)

def main():
    lock=load(OUT/'research_lock.json');assert not lock['production_release'];assert sha(OUT/'configs.json')==lock['config_sha256']
    for path,digest in lock['inference_sha256'].items():assert sha(STAGE/path)==digest,path
    assert dependency_hashes()==lock['baseline_dependencies'],'STEP5 modified'
    from common import locks
    assert locks()==load(BASE_OUT/'freeze.json')['dependencies'],'Frozen STEP1/STEP2 dependencies modified'
    for p in (STAGE/'src').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    s=load(OUT/'SUMMARY.json');assert s['complete'];total=0;available=0;points=0
    for name in lock['heldout_variants']:
        folders=[p.parent for p in (OUT/'heldout'/name).glob('PREDICTION_COMPLETE.json')]
        folders=[p.parent for p in (OUT/'heldout'/name).glob('*/PREDICTION_COMPLETE.json')]
        assert len(folders)==1422,(name,len(folders))
        for folder in folders:
            p=load_prediction(folder);m=load(folder/'PREDICTION_COMPLETE.json');e=load(folder/'evaluation.json');assert m['time_ns']>lock['time_ns'];assert e['evaluated_after_prediction_ns']>=m['time_ns']
            assert sha(folder/'prediction.json')==m['prediction_sha256'];assert sha(folder/'points.npz')==m['points_sha256'];assert sha(folder/'provenance.npz')==m['provenance_sha256']
            assert all(k<=p['start_frame'] for k in p['fusion']['source_frames']);assert all(k<=p['start_frame'] for k in p['fusion']['causal_cloud_access'])
            ids=p['indices'];assert len(np.unique(ids))==len(ids);total+=1;points+=len(ids)
            with np.load(folder/'provenance.npz') as z:
                assert np.all(z['source_frame']<=p['start_frame']);assert np.all(z['age_frames']==p['start_frame']-z['source_frame']);assert np.all(z['age_seconds']>=0)
                assert len(np.unique(np.column_stack((z['source_frame'],z['source_row'])),axis=0))==len(ids)
                assert len(z['xyz'])==len(ids)
            if p['seed']['status']=='AVAILABLE':
                available+=1
                if name!='FUSED_BOOTSTRAP':assert (folder/'oracle_frame_evaluation.json').exists()
                if name=='F0':assert load(folder/'BASELINE_METRICS_IDENTICAL.json')['exact']
            if name=='F0':assert (folder/'BASELINE_IDENTICAL.json').exists()
            expected=dict(CFG,method='M2' if name in ('M2_F0','M2_BEST') else 'M1');assert p['config']==expected
            for st in p['steps']:
                if st.get('basis') is not None:
                    B=np.asarray(st['basis']);np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-9);assert np.linalg.det(B)>.99999
                if st['status']=='ACCEPTED':assert not set(st['support_indices'])&set(st['overlap_indices'])
        print('VERIFIED',name,len(folders),flush=True)
    exports=load(OUT/'las/index.json');assert len(exports)>=20
    for x in exports:
        assert x['original_fields_bitwise_verified']
        for field in ('fused_input','prediction_overlay','current_only_comparison'):assert (OUT/x[field]).exists()
    gifs=list((OUT/'animations').glob('*.gif'));assert len(gifs)>=8
    for p in gifs:
        with Image.open(p) as im:assert im.n_frames>=2;im.verify()
    png=list((OUT/'gallery').glob('*.png'))
    for p in png:Image.open(p).verify()
    parser=Links()
    for name in ('gallery.html','report.html'):parser.feed((OUT/name).read_text(encoding='utf-8'))
    for link in parser.links:
        if link.startswith('#') or urlparse(link).scheme:continue
        assert (OUT/unquote(link.split('#')[0])).exists(),link
    required=('per_start_variant','per_step_variant','source_contribution','fusion_density','registration_blur','failure_rescue','orientation_comparison','range_bins_comparison','runtime','memory','method_ablation','las_exports')
    for name in required:assert (OUT/(name+'.csv')).exists()
    source=load(OUT/'audit/source_integrity.json');assert source['all_source_point_hashes_match']
    result=dict(status='PASS',predictions=total,available_predictions=available,predicted_point_records=points,F0_arrays_and_steps_identical=True,F0_metrics_identical=True,previous_stages_unchanged=True,inference_config_unchanged_after_lock=True,causal_source_provenance=True,unique_source_rows=True,source_files_verified=source['files'],las_sets=len(exports),pngs=len(png),gifs=len(gifs),html_links=len(parser.links),production_release=False)
    save(OUT/'VERIFICATION.json',result)
    save(STAGE/'MANIFEST.json',dict(files=[dict(path=p.relative_to(STAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in STAGE.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]))
    files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json' and 'weighted_preliminary' not in p.parts]
    save(OUT/'ARTIFACT_MANIFEST.json',dict(files=files));print('FINAL PASS',result,flush=True)
if __name__=='__main__':main()
