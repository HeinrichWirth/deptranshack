"""Initialize once; pin the user-selected C4 without changing previous stages."""
from rr_common import *
import shutil

def main():
    if (STAGE/'dependency_lock.json').exists():
        lock=check_dependencies()
        # Pin frozen template/config assets as well as Python modules.
        for stage in ('01_rail2d','04_contact_rail_final'):
            folder=ROOT/'MVP/stages'/stage
            for row in load(folder/'MANIFEST.json')['files']:
                p=folder/row['path'];assert sha(p)==row['sha256']
                lock['files'][p.relative_to(ROOT).as_posix()]=row['sha256']
            lock['files'][(folder/'MANIFEST.json').relative_to(ROOT).as_posix()]=sha(folder/'MANIFEST.json')
        save(STAGE/'dependency_lock.json',lock)
        print('Already initialized; frozen dependencies verified and assets pinned.',len(lock['files']));return
    for name in ('baseline','seed_calibration','oracle_cr','c4_cr','alpha_observability','cr_profile_roll','curvature_roll','models','uncertainty','corridors','oracle_decomposition','side_switch','stress','failures','gallery','las','runtime','audit','phase_a','phase_b','phase_e'):(OUT/name).mkdir(parents=True,exist_ok=True)
    freeze=load(C4ROOT/'research_freeze.json');assert 'C4' in freeze['top3']
    hashes={}
    for name,h in freeze['code_sha256'].items():
        p=C4STAGE/'src'/name;assert sha(p)==h;hashes[p.relative_to(ROOT).as_posix()]=h
    assert sha(C4ROOT/'configs.json')==freeze['config_sha256']
    previous=load(C4ROOT/'audit/previous_code_sha.json')
    for rel,h in previous.items():assert sha(ROOT/rel)==h;hashes[rel]=h
    for stage in ('01_rail2d','04_contact_rail_final'):
        folder=ROOT/'MVP/stages'/stage
        for p in folder.rglob('*.py'):
            if '__pycache__' not in p.parts:hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (C4ROOT/'research_freeze.json',C4ROOT/'configs.json',C4ROOT/'audit/cohort.json',C4ROOT/'audit/screen.json',C4STAGE/'MANIFEST.json'):
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    cfg=load(C4ROOT/'configs.json')['C4'];cohort=load(C4ROOT/'audit/cohort.json')
    split=load(ROOT/'MVP/stages/03_contact_generalization/results_contact_generalization/protocol.json')['outer_split']
    protocol=dict(stage='STEP6_RUNNING_RAILS_FROM_CONTACT',created_ns=time.time_ns(),split=split,cohort=cohort,screen=load(C4ROOT/'audit/screen.json'),primary_cr='C4',primary_seed_length_m=8,
        class1_policy='Current T labels only inside seed; future class1 only after persisted rail prediction.',
        primary_policy='No raw rail points after seed. CR0 and oracle roll/offsets are separately labelled diagnostics.',
        station_policy='Common CR stations and transverse planes, not independent arc lengths or unconstrained nearest points.',
        evaluation_label='reused research benchmark, not fresh blind validation',
        optional_step53='Excluded: user explicitly fixed C4; STEP5.3 unfinished and not a dependency.',
        phase_order=['observability','A: all families on 36 stratified development starts','B: top ~8 on 108 development starts','C: leave-one-run-out','D: freeze top3','E: one reused benchmark, 1422 starts'],
        synthetic_geometry='Allowed only as explicitly marked predicted rail curves; never relabel real measurements as synthetic or vice versa.',
        range_bins=[0,10,20,30,40,50,60,75,100,125,150],selection_bins=[[30,50],[50,75],[75,100]],source_dataset=str(dataset()),production_freeze=False)
    save(OUT/'protocol.json',protocol);save(STAGE/'C4_CONFIG_LOCKED.json',cfg)
    indices={}
    for group,rows in cohort.items():
        if group not in ('development','heldout'):continue
        for r in rows:
            folder=c4_path(r['run'],r['frame']);path=folder/'PREDICTION_COMPLETE.json';m=load(path)
            assert m['inference_sha256']==freeze['code_sha256']
            indices[key(r['run'],r['frame'])]=dict(folder=folder.relative_to(ROOT).as_posix(),marker_sha256=sha(path),files=m['files'],split=group)
    save(STAGE/'c4_input_index.json',indices)
    hashes[(STAGE/'C4_CONFIG_LOCKED.json').relative_to(ROOT).as_posix()]=sha(STAGE/'C4_CONFIG_LOCKED.json')
    hashes[(STAGE/'c4_input_index.json').relative_to(ROOT).as_posix()]=sha(STAGE/'c4_input_index.json')
    save(STAGE/'dependency_lock.json',dict(created_ns=time.time_ns(),primary='C4',mutations_permitted=False,files=hashes,c4_starts=len(indices),note='STEP6 only; does not modify STEP5.2 research/production status.'))
    source=Path('C:/Users/heinrich.wirth/.codex/attachments/5ecd04e4-2440-4f32-ab25-ccda99da1dc6/Вставленный текст.txt')
    shutil.copyfile(source,STAGE/'REQUEST_RU.txt')
    save(OUT/'audit/initialization.json',dict(status='PASS',dependencies=len(hashes),c4_inputs=len(indices),previous_stages_modified=False))
    print('STAGE6 INITIALIZED',len(indices),'frozen C4 inputs;',len(hashes),'dependencies',flush=True)

if __name__=='__main__':main()
