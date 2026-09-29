"""Pin the old study BEFORE any new method is run."""
from lc_common import *
import shutil

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'reproduction').mkdir(exist_ok=True);(OUT/'audit').mkdir(exist_ok=True)
    target=OUT/'baseline_freeze.json'
    if target.exists():check_baseline();print('Existing baseline freeze verified');return
    from run_inference import check_freeze
    old=check_freeze();frozen.check_dependencies();paths=list((OLD_STAGE/'src').glob('*.py'))+list((OLD_STAGE/'tests').glob('*.py'))+[OLD_STAGE/'dependency_lock.json',OLD_STAGE/'c4_input_index.json',OLD_STAGE/'C4_CONFIG_LOCKED.json',OLD_OUT/'research_freeze.json',OLD_OUT/'configs.json',OLD_OUT/'protocol.json',OLD_OUT/'models/global_prior.json',OLD_OUT/'MANIFEST.json',OLD_OUT/'gallery/cases.json',OLD_OUT/'gallery/selection.json',OLD_OUT/'audit/interpolation_diagnostic.json',OLD_OUT/'audit/posthoc_diagnostics.json']
    for group in ('phase_b','phase_e'):paths.extend([OLD_OUT/group/'range_metrics.json',OLD_OUT/group/'per_start.json',OLD_OUT/group/'INFERENCE_COMPLETE.json'])
    assets=load(OLD_STAGE/'dependency_lock.json')['files'];paths.extend(ROOT/p for p in assets if '04_contact_rail_final' in p)
    files={p.relative_to(ROOT).as_posix():sha(p) for p in paths};cases=load(OLD_OUT/'gallery/cases.json');rows=load(OLD_OUT/'phase_e/per_start.json');name=old['best_development'];rr=[r for r in rows if r['method']==name]
    save(target,dict(name='STEP6_BASELINE_FROZEN',created_ns=time.time_ns(),files=files,step6_freeze=old,step6_manifest_sha256=sha(OLD_OUT/'MANIFEST.json'),C4_dependency_sha256=sha(OLD_STAGE/'dependency_lock.json'),template_assets={p:h for p,h in assets.items() if p.endswith('.npz') and '04_contact_rail_final' in p},expected=dict(method=name,benchmark_starts=len(rr),benchmark_available=sum(r['status']=='AVAILABLE' for r in rr),max_range=max(r.get('max_predicted_range') or 0 for r in rr),gt20cm_cases=sorted(r['id'] for r in cases if 'all_gt20cm' in r['tags']),range_metrics=[r for r in load(OLD_OUT/'phase_e/range_metrics.json') if r['method']==name],overshoot=load(OLD_OUT/'audit/interpolation_diagnostic.json')),mutation_permitted=False))
    source=Path(r'C:\Users\heinrich.wirth\.codex\attachments\e0a268e7-13e5-415a-9b3b-72d3c00c2dd2\Вставленный текст.txt');shutil.copyfile(source,STAGE/'REQUEST_RU.txt');save(STAGE/'stage.json',dict(stage='STEP6.1_LATENT_CONTACT_REFERENCE',output=str(OUT),baseline=str(target),previous_stages_read_only=True))
    print('BASELINE PINNED',len(files),'files',flush=True)

if __name__=='__main__':main()
