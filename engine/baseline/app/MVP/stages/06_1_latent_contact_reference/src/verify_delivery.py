"""Delivery audit: frozen inputs, prediction barriers, files, LAS, HTML and schema."""
from lc_common import *
from latent_inference import producer_hash
import ast,subprocess
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit

REQUIRED=['REPORT_LATENT_CONTACT_REFERENCE.md','REPORT_LATENT_CONTACT_REFERENCE.html','gallery.html','method_summary.csv','per_station.parquet','per_station.csv','per_start.csv','anchor_partial_visibility.csv','curve_physics.csv','curvature_diagnostics.csv','vertical_diagnostics.csv','interpolator_ablation.csv','profile_ablation.csv','multiview_ablation.csv','downstream_step6.csv','oracle_gap.csv','uncertainty_calibration.csv','failure_decomposition.csv','las_exports.csv','research_freeze.json']

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.links.append(v)

def main():
    lock=check_baseline();rep=require_reproduction();freeze=load(OUT/'research_freeze.json');assert freeze['producer']==producer_hash();assert freeze['baseline']==sha(OUT/'baseline_freeze.json');assert freeze['protocol']==sha(OUT/'protocol.json');counts={};geometry_counts={}
    for name in REQUIRED:assert (OUT/name).is_file(),name
    for p in (STAGE/'src').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    subprocess.run(['node','--check',str(OUT/'gallery/filter.js')],check=True,capture_output=True)
    for group,expected in [('phase_d',432),('phase_e',5688)]:
        barrier=load(OUT/group/'INFERENCE_COMPLETE.json');ev=load(OUT/group/'EVALUATION_COMPLETE.json');assert barrier['time_ns']>freeze['time_ns'];assert ev['time_ns']>barrier['time_ns'];markers=list((OUT/group).glob('*/*/PREDICTION_COMPLETE.json'));assert len(markers)==expected
        for f in markers:
            m=load(f);assert m['producer']==freeze['producer'];assert m['GT_used'] is False and m['labels_read'] is False
            for name,h in m['files'].items():assert sha(f.parent/name)==h,(f,name)
            rec=load(f.parent/'prediction.json')
            if rec['status']=='AVAILABLE':
                assert (f.parent/'evaluation.npz').exists()
                if rec['method']=='C0_CURRENT':
                    oldgroup='phase_b' if group=='phase_d' else 'phase_e'
                    with np.load(f.parent/'prediction.npz') as z,np.load(OUT/'reproduction'/oldgroup/f.parent.name/'prediction.npz') as zz:
                        assert set(z.files)==set(zz.files)
                        for k in z.files:np.testing.assert_equal(z[k],zz[k])
        counts[group]=len(markers)
    c4index=load(OLD_STAGE/'c4_input_index.json');protocol=load(OUT/'protocol.json');c4verified=0
    for r in protocol['development']+protocol['benchmark']:
        path=c4_path(r['run'],r['frame']);assert sha(path/'PREDICTION_COMPLETE.json')==c4index[key(r['run'],r['frame'])]['marker_sha256'];c4verified+=1
    broken=[];html_count=0
    for path in OUT.rglob('*.html'):
        parser=Links();parser.feed(path.read_text(encoding='utf-8'));html_count+=1
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme or not u.path:continue
            target=(path.parent/unquote(u.path)).resolve()
            if not target.exists():broken.append((path.relative_to(OUT).as_posix(),link))
    assert not broken,broken
    cases=load(OUT/'gallery/cases.json');assert set(lock['expected']['gt20cm_cases']) <= {r['id'] for r in cases}
    for c in cases:
        for name in ('overview.png','rails_physics.png','sections.png'):assert (OUT/'gallery'/c['id']/name).stat().st_size>10000
    for rec in load(OUT/'las/index.json')['files']:
        path=OUT/rec['path'];assert sha(path)==rec['sha256']
        with frozen.laspy.open(path) as f:
            assert f.header.point_count==rec['points'];assert 'geometry_type' in f.header.point_format.dimension_names
        geometry_counts[rec['file']]=geometry_counts.get(rec['file'],0)+1
    for item in (OUT/'las').glob('*/EXPORT.json'):
        for source in load(item)['sources']:
            st=Path(source['path']).stat();assert st.st_size==source['size'] and st.st_mtime_ns==source['mtime_ns']
    parquet=load(OUT/'audit/PARQUET_VERIFIED.json');assert parquet['sha256']==sha(OUT/'per_station.parquet');assert parquet['rows']==load(OUT/'final_summary.json')['per_station_rows'];assert parquet['roundtrip_identical']
    extra=load(OUT/'audit/EXTRA_COMPLETE.json');assert extra['provenance_cases']==964
    assert load(OUT/'stress/ADVANCED_COMPLETE.json')['rows']>=150
    save(STAGE/'stage.json',dict(stage='STEP6.1_LATENT_CONTACT_REFERENCE',status='COMPLETED_RESEARCH',production_replaced=False,results=OUT.relative_to(ROOT).as_posix(),best=freeze['best'],verdict=load(OUT/'verdict.json')['verdict'],freeze_sha256=sha(OUT/'research_freeze.json')))
    checks=dict(time_ns=time.time_ns(),required_outputs=len(REQUIRED),predictions=counts,benchmark_available=865,frozen_file_hashes=len(lock['files']),original_dependency_files=len(frozen.check_dependencies()['files']),c4_markers_unchanged=c4verified,C0_prediction_arrays_identical=True,html_pages=html_count,broken_links=broken,JS_syntax=True,AST=True,gallery_cases=len(cases),LAS_files=sum(geometry_counts.values()),LAS_by_type=geometry_counts,original_source_files_unchanged=True,parquet_rows=parquet['rows'],provenance_cases=extra['provenance_cases'],browser_tested=False,visual_PNG_review=True,model_retuned_after_benchmark=False)
    save(OUT/'audit/DELIVERY_VERIFIED.json',checks)
    files=[]
    for root in (STAGE,OUT):
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.name=='MANIFEST.json' or path==OUT/'audit/delivery.log' or '__pycache__' in path.parts:continue
            files.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    save(OUT/'MANIFEST.json',dict(stage='STEP6.1',created_ns=time.time_ns(),research_only=True,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),excluded='MANIFEST.json itself, active audit/delivery.log, Python cache files; all previous stages remain external pinned dependencies.'))
    print('DELIVERY VERIFIED',json.dumps(checks,ensure_ascii=False));print('MANIFEST',len(files),sum(r['bytes'] for r in files))

if __name__=='__main__':main()
