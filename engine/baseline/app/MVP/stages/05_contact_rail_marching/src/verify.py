"""Final delivery verification: frozen inputs, provenance, completeness and artifacts."""
from common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
from PIL import Image
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
from evaluation import load_prediction
import ast

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href'):self.links.append(v)

def main():
    frozen=load(OUT/'freeze.json');assert frozen['production_release'] is False
    assert sha(OUT/'config.json')==frozen['config_sha256'];assert sha(OUT/'protocol.json')==frozen['protocol_sha256']
    for path,digest in frozen['inference_sha256'].items():assert sha(STAGE/path)==digest,path
    assert locks()==frozen['dependencies'];assert str(dataset())==load(OUT/'protocol.json')['data_root']
    source=load(OUT/'audit/source_integrity.json');assert source['all_original_point_hashes_match']
    for p in (STAGE/'src').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    rows=load(OUT/'analysis_rows.json');assert len(rows)==1422;total_points=0;available=0;not_before=True;bad=[]
    for r in rows:
        run=r['run'];i=r['start_frame'];folder=OUT/'heldout'/key(run,i);pred=load_prediction(folder);marker=load(folder/'PREDICTION_COMPLETE.json')
        assert pred['config']==load(OUT/'config.json');assert marker['time_ns']>frozen['time_ns']
        assert marker['prediction_sha256']==sha(folder/'prediction.json');assert marker['points_sha256']==sha(folder/'points.npz')
        ids=pred['indices'];assert len(ids)==len(np.unique(ids));assert np.all(ids>=0);total_points+=len(ids)
        if not r['seed_available']:continue
        available+=1;xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');assert np.all(ids<len(xyz))
        np.testing.assert_array_equal(ids[:len(pred['seed']['indices'])],pred['seed']['indices'])
        assert (OUT/'failures'/(key(run,i)+'.json')).exists()
        for suffix in ('_without_gt.png','_with_gt.png'):
            p=OUT/'failures'/(key(run,i)+suffix);assert p.exists();Image.open(p).verify()
        for step in pred['steps']:
            if step['status']!='ACCEPTED':continue
            idx=np.array(step['support_indices'],dtype=int);B=np.array(step['basis']);C=np.array(step['plane_origin']);u=(xyz[idx]-C)@B[:,0]
            assert np.all((u>=-1e-8)&(u<=pred['config']['window']+1e-8))
            np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-9);assert np.linalg.det(B)>.99999
            assert not np.intersect1d(idx,np.array(step['overlap_indices'],dtype=int)).size
        for name in ('ORACLE_SEED_8','ORACLE_SEED_10','ORACLE_SEED_12','ORACLE_SEED_15','ORACLE_FRAME','ORACLE_SEED_FRAME','ORACLE_SEED_TEMPLATE8','ORACLE_SEED_TEMPLATE8_FRAME'):
            path=OUT/'oracles'/name/key(run,i)/'evaluation.json';assert path.exists(),str(path)
            e=load(path)
            if name in ('ORACLE_SEED_10','ORACLE_SEED_12','ORACLE_SEED_15'):assert e.get('oracle_version')==3,str(path)
    assert total_points==load(OUT/'audit/point_csv.json')['rows']
    exports=load(OUT/'las/index.json');assert len(exports)>=15
    for r in exports:
        for field in ('overlay','predicted_map','predicted_sensorT'):assert (OUT/r[field]).exists()
    animations=list((OUT/'animations').glob('*.gif'));assert len(animations)>=8
    for p in animations:
        with Image.open(p) as im:im.verify()
    parser=Links();parser.feed((OUT/'gallery.html').read_text(encoding='utf-8'))
    for url in parser.links:
        if url.startswith('#') or urlparse(url).scheme:continue
        p=OUT/unquote(url.split('#')[0]);assert p.exists(),url
    for p in (OUT/'gallery').glob('*.png'):Image.open(p).verify()
    required=('per_start_frame.csv','per_step.csv','per_predicted_point.csv.gz','reach_summary.csv','distance_bins.csv','failure_summary.csv','method_ablation.csv','orientation_metrics.csv','compactness_metrics.csv','future_gt_alignment.csv','oracle_comparison.csv','runtime.csv','las_exports.csv','REPORT_CONTACT_MARCHING.md')
    for name in required:assert (OUT/name).exists(),name
    result=dict(status='PASS',starts=len(rows),available_starts=available,source_points_exported=total_points,terminal_pngs=available*2,
        las_sets=len(exports),animations=len(animations),html_local_links=len(parser.links),frozen_inference_and_config_unchanged=True,
        previous_frozen_dependencies_unchanged=True,all_original_source_point_hashes_match=source,
        point_provenance_unique=True,no_overlap_readded=True,bases_orthonormal=True,all_oracle_variants_complete=True,
        note='LAS original-field bit equality and sensor-coordinate rounding are verified during each export; unit test transcript in audit/unit_tests.txt.')
    save(OUT/'VERIFICATION.json',result)
    manifest=[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in STAGE.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]
    save(STAGE/'MANIFEST.json',dict(kind='research_delivery_not_production_freeze',files=manifest))
    delivery=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('cache','future_gt','ablation','development','pilot','oracles')) or p.name=='ARTIFACT_MANIFEST.json':continue
        delivery.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    save(OUT/'ARTIFACT_MANIFEST.json',dict(files=delivery));print('DELIVERY',result,flush=True)
if __name__=='__main__':main()
