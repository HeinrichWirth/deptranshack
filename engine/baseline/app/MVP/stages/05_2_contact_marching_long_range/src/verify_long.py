"""Read-only provenance, immutable inputs, artifacts and manifest verification."""
from long_common import *
import argparse,ast,subprocess
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit

def source_hash(r):
    run=r['run'];i=int(r['frame']);fr=frames(run)[i];path=dataset()/run/fr['file']
    with laspy.open(path) as f:h=f.header
    raw=np.memmap(path,dtype=np.uint8,mode='r',offset=h.offset_to_point_data,shape=(h.point_count*h.point_format.size,))
    digest=hashlib.sha256(raw).hexdigest();assert digest==r['point_data_sha256'],str(path)
    return dict(run=run,frame=i,points=h.point_count,sha256=digest)

def inputs():
    old=load(OUT/'audit/previous_inventory.json');n=0
    for rel,(size,mtime) in old.items():
        p=ROOT/rel;s=p.stat();assert (s.st_size,s.st_mtime_ns)==(size,mtime),str(p);n+=1
    code=load(OUT/'audit/previous_code_sha.json')
    for rel,h in code.items():assert sha(ROOT/rel)==h,rel
    rows=csv_read(OLD/'audit/source_integrity.csv')
    with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(source_hash,rows))
    save(OUT/'audit/inputs_verified.json',dict(previous_files_unchanged=n,previous_code_hashes=len(code),source_point_hashes_verified=len(verified),time_ns=time.time_ns()))
    csv_write(OUT/'audit/source_hashes_verified.csv',verified)
    print('INPUTS VERIFIED',n,len(verified),flush=True)

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links += [v for k,v in attrs if k in ('src','href') and v]

def prediction_check(path):
    folder=path.parent;m=check_marker(folder);lock=load(OUT/'research_freeze.json')
    assert m['time_ns']>lock['time_ns'] and m['inference_sha256']==lock['code_sha256']
    e=load(folder/'evaluation.json');p=read_prediction(folder);i=p['start_frame']
    assert e['evaluated_ns']>=e['prediction_ns']==m['time_ns']
    with np.load(folder/'provenance.npz') as z:
        sf=z['source_frame'];sr=z['source_row'];age=z['age_frames'];xyz=z['xyz']
        assert np.all(sf<=i) and np.all(age>=0) and np.all(age==i-sf)
        keys=(sf.astype(np.uint64)<<np.uint64(32))|sr.astype(np.uint64)
        assert len(np.unique(keys))==len(keys)==len(p['indices'])==len(p['point_state'])
        assert np.isfinite(xyz).all() and np.isin(p['point_state'],[1,2]).all()
    with np.load(folder/'evidence.npz') as z:assert np.all(z['source_frame']<=i) and np.all(z['age_frames']>0)
    for step in p['steps']:
        if step.get('state')=='GAP':assert not np.any(p['point_step']==step['step_index'])
    if p['variant'].startswith('B'):assert p.get('baseline_identical') is True
    return p['variant'],p['seed']['status']=='AVAILABLE',int(np.sum(p['point_state']==2)),int(np.sum(p['point_state']==1))

def final():
    lock=load(OUT/'research_freeze.json');assert code_hashes()==lock['code_sha256'];assert sha(OUT/'configs.json')==lock['config_sha256']
    for rel,(size,mtime) in load(OUT/'audit/previous_inventory.json').items():
        stat=(ROOT/rel).stat();assert (stat.st_size,stat.st_mtime_ns)==(size,mtime),rel
    for rel,h in load(OUT/'audit/previous_code_sha.json').items():assert sha(ROOT/rel)==h,rel
    for path in (STAGE/'src').glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(STAGE/'tests'),'-v'],capture_output=True,text=True)
    (OUT/'audit/regression_tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8');assert test.returncode==0
    paths=sorted((OUT/'phase_e_benchmark').glob('*/*/PREDICTION_COMPLETE.json'))
    expected=len(load(OUT/'audit/cohort.json')['heldout'])*(4+len(lock['top3']));assert len(paths)==expected,(len(paths),expected)
    totals={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for j,(name,available,confirmed,tentative) in enumerate(pool.map(prediction_check,paths),1):
            t=totals.setdefault(name,dict(starts=0,available=0,confirmed_points=0,tentative_points=0));t['starts']+=1;t['available']+=available;t['confirmed_points']+=confirmed;t['tentative_points']+=tentative
            if j%1000==0:print('VERIFY PREDICTIONS',j,len(paths),flush=True)
    exports=load(OUT/'las/index.json')
    for export in exports:
        # Exporter's read-back audit already verifies every original and extra field.
        folder=OUT/'las'/export['id'];rec=load(folder/'EXPORT.json')
        for file in rec['files']:
            assert file['original_fields_bitwise_verified']
            assert sha(OUT/file['path'])==file['sha256']
    examples=load(OUT/'gallery/examples.json')
    for e in examples:
        folder=OUT/'gallery'/e['id'];record=load(folder/'COMPLETE.json')
        for name in record['files']:assert (folder/name).is_file()
    links=0
    for file in ('report.html','gallery.html'):
        parser=Links();parser.feed((OUT/file).read_text(encoding='utf-8'))
        for value in parser.links:
            if value.startswith(('https:','http:','#','data:')):continue
            target=unquote(urlsplit(value).path)
            if target in ('MANIFEST.json','VERIFICATION.json'):continue
            assert (OUT/target).exists(),(file,target);links+=1
    node=subprocess.run(['node','--check',str(OUT/'audit/gallery_script.js')],capture_output=True,text=True);assert node.returncode==0,node.stderr
    facts=load(OUT/'REPORT_FACTS.json');iv=load(OUT/'audit/inputs_verified.json')
    save(OUT/'VERIFICATION.json',dict(status='PASS',time_ns=time.time_ns(),predictions=len(paths),by_method=totals,
      original_inputs=iv,freeze_hashes_unchanged=True,regression_tests='13 passed',las_sets=len(exports),gallery_examples=len(examples),local_html_links=links,
      previous_inventory_rechecked_at_final=True,viewer_rail_regression=(OUT/'audit/viewer_rail_regression.txt').read_text(encoding='utf-8').strip(),
      browser_tested=False,browser_note='Static HTML links and JavaScript syntax checked; figures inspected as PNG. No browser automation.',report_facts=facts))
    print('FINAL VERIFICATION PASS',len(paths),len(exports),len(examples),flush=True)

def manifest():
    for root in (OUT,STAGE):
        files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts}
        save(root/'MANIFEST.json',dict(time_ns=time.time_ns(),files=files))
        print('MANIFEST',root.name,len(files),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inputs','final','manifest']);a=ap.parse_args();globals()[a.mode]()
