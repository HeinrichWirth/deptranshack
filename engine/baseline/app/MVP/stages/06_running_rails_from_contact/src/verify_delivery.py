"""Integrity, causal provenance and local-artifact checks; no model mutation."""
from rr_common import *
from run_inference import check_freeze
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import unquote
import argparse,ast

class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.urls.append(v)

def marker_check(folder):
    mark=load(folder/'PREDICTION_COMPLETE.json');record=load(folder/'prediction.json')
    for n,h in mark['files'].items():assert sha(folder/n)==h,(str(folder),n)
    assert mark['future_class1_used'] is False and mark['future_class2_used'] is False and mark['post_seed_raw_rail_points_used'] is False
    ev=load(folder/'evaluation.json');assert ev['prediction_ns']==mark['time_ns'] and ev['evaluated_ns']>=mark['time_ns']
    return mark['time_ns'],record['input_folder']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',action='store_true');args=ap.parse_args();lock=check_freeze();check_dependencies();barrier=load(OUT/'phase_e/INFERENCE_COMPLETE.json');folders=[OUT/'phase_e'/n/key(r['run'],r['frame']) for n in lock['methods'] for r in barrier['starts']]
    with ThreadPoolExecutor(max_workers=8) as pool:checks=list(pool.map(marker_check,folders))
    assert min(r[0] for r in checks)>lock['time_ns'];assert max(r[0] for r in checks)<=barrier['time_ns'];inputs=sorted(set(r[1] for r in checks));labelmax=[]
    for path in inputs:
        folder=OUT/path;mark=load(folder/'COMPLETE.json');meta=load(folder/'input.json')
        for n,h in mark['files'].items():assert sha(folder/n)==h
        assert not mark['future_labels_used'] and not mark['post_seed_raw_rail_points_read']
        if meta['status']=='AVAILABLE':
            assert meta['source_max_frame']<=meta['current_frame'];assert meta['label_audit']['label_u_max']<=8.000001;assert meta['label_audit']['label_u_min']>=0;labelmax.append(meta['label_audit']['label_u_max'])
    for run in lock['split']['validation']:
        ref=load(OUT/'audit/future_class1'/run/'anchors.json');assert ref['created_ns']>barrier['time_ns']
    idx=load(STAGE/'c4_input_index.json')
    for k,r in idx.items():
        # Every pinned C4 marker is checked; its files were checked on input read.
        run=k.rsplit('__',1)[0];i=int(k.rsplit('__',1)[1]);assert sha(c4_path(run,i)/'PREDICTION_COMPLETE.json')==r['marker_sha256']
    htmlfiles=list((OUT/'gallery').rglob('*.html'))+[OUT/'gallery.html',OUT/'REPORT_RUNNING_FROM_CR.html'];bad=[]
    for path in htmlfiles:
        parser=Links();parser.feed(path.read_text(encoding='utf-8'))
        for url in parser.urls:
            cleanurl=unquote(url.split('#')[0].split('?')[0])
            if not cleanurl or '://' in cleanurl:continue
            dest=path.parent/cleanurl
            if not dest.exists() and dest.name!='MANIFEST.json':bad.append((str(path),url))
    assert not bad,bad[:10]
    for p in (STAGE/'src').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    required=['per_station.csv','per_start.csv','per_method.csv','seed_geometry.csv','alpha_gt_pred.csv','alpha_sensitivity.csv','offset_stability.csv','range_metrics.csv','corridor_metrics.csv','oracle_decomposition.csv','seed_stress.csv','cr_stress.csv','side_switch.csv','runtime.csv','las_exports.csv','research_freeze.json','REPORT_RUNNING_FROM_CR.md']
    for f in required:assert (OUT/f).exists(),f
    las=load(OUT/'las/index.json');assert las['cases']>=20
    for row in las['files']:assert sha(OUT/row['path'])==row['sha256']
    save(OUT/'audit/DELIVERY_VERIFIED.json',dict(time_ns=time.time_ns(),frozen_prediction_files_checked=len(folders),seed_inputs_checked=len(inputs),C4_markers_unchanged=len(idx),dependency_files_unchanged=len(load(STAGE/'dependency_lock.json')['files']),max_seed_label_u=max(labelmax),future_GT_read_after_benchmark_prediction_barrier=True,html_pages_checked=len(htmlfiles),broken_local_links=bad,LAS_files=len(las['files']),browser_tested=False,config_modified_after_freeze=False))
    if args.manifest:
        paths=[p for base in (STAGE,OUT) for p in base.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and p.suffix!='.tmp' and '__pycache__' not in p.parts]
        def record(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(record,paths))
        save(OUT/'MANIFEST.json',dict(stage='STEP6_RUNNING_RAILS_FROM_CONTACT',files=records,files_count=len(records),total_bytes=sum(r['bytes'] for r in records),freeze_sha256=sha(OUT/'research_freeze.json'),C4_unchanged=True,source_data_untouched=True,time_ns=time.time_ns(),scope='New stage code and outputs. Manifest excludes itself and temporary files.'))
    print('DELIVERY VERIFIED',len(folders),'predictions,',len(htmlfiles),'HTML pages,',len(las['files']),'LAS',flush=True)

if __name__=='__main__':main()
