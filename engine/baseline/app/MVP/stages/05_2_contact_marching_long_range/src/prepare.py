"""Record protocol and fixed development screen before search."""
from long_common import *
from configurations import configurations
import shutil

def main():
    initialize()
    assert not (OUT/'protocol.json').exists(), 'Protocol already exists'
    cohort=load(OLD/'audit/cohort.json');save(OUT/'audit/cohort.json',cohort)
    candidates=[]
    for r in cohort['development']:
        p=OLD/'development/F0'/key(r['run'],r['frame'])/'evaluation.json'
        e=load(p);v=e['summary']
        if v.get('evaluation_eligible'):
            candidates.append(dict(r,reach=v['continuous_reach_5cm'],reason=v['stop_reason'],
                curvature=max([abs(s.get('gt_curvature_per_m') or 0) for s in e['steps']]+[0]),
                grade=max([abs(s.get('gt_pitch_deg') or 0) for s in e['steps']]+[0])))
    chosen=[]
    for run in SPLIT['development']:
        rr=[r for r in candidates if r['run']==run]
        for f in (lambda x:x['reach'],lambda x:-x['reach'],lambda x:-x['curvature'],lambda x:-x['grade']):
            pool=[r for r in rr if r not in chosen]
            if pool:chosen.append(sorted(pool,key=f)[0])
    remaining=sorted([r for r in candidates if r not in chosen],key=lambda r:(r['reason'],r['reach'],r['run']))
    if len(chosen)<36:
        inds=np.linspace(0,len(remaining)-1,min(36-len(chosen),len(remaining)),dtype=int)
        chosen += [remaining[j] for j in inds]
    save(OUT/'audit/screen.json',chosen)
    save(OUT/'configs.json',configurations())
    inventory={};code={}
    for root in (ROOT/'MVP/stages/05_contact_rail_marching',OLD_STAGE,BASE,OLD):
        for p in root.rglob('*'):
            if not p.is_file():continue
            s=p.stat(); rel=p.relative_to(ROOT).as_posix();inventory[rel]=[s.st_size,s.st_mtime_ns]
            if root in (ROOT/'MVP/stages/05_contact_rail_marching',OLD_STAGE) and '__pycache__' not in p.parts:code[rel]=sha(p)
    save(OUT/'audit/previous_inventory.json',inventory);save(OUT/'audit/previous_code_sha.json',code)
    save(OUT/'protocol.json',dict(created_ns=time.time_ns(),stage='STEP5.2 long-range research',production_release=False,
      split=SPLIT,development_n=len(cohort['development']),screen_n=len(chosen),benchmark_n=len(cohort['heldout']),
      benchmark_warning='reused benchmark, not fresh external validation',
      selection='Phase A: 36 fixed stratified development starts; top 15 nonbaseline by equal-run long-range utility. Phase B: all 108 development starts. Phase C: rank top 8 using leave-one-run-out reselection, median run utility, worst-run P50/P75, confirmed catastrophic transverse errors and paired losses. Phase D: 5 prespecified combinations. Lock 3 candidates before any new benchmark inference/evaluation.',
      utility='Per-run 2*P50 + 3*P75 + 5*P100 + mean(min(reach,120))/120 - 10*confirmed_error20_rate - 2*loss4_rate; run mean primary, worst run secondary. Reach is unchanged exact STEP5 continuous 5cm metric; transverse results reported separately.',
      causality='Current and contiguous valid past only; sanitized T+1 pose only for unchanged seed. No future clouds/labels/trajectory in inference. Evaluation strictly after prediction marker.',
      geometry='AF mode A: current T confirmed source records only in PCA and old overlap; mode B: accepted past may enter subsequent PCA, never old overlap. If current overlap disappears, strict AF stops; stateful variants may use short prior as tentative search.',
      gaps='Explicit empty spatial blocks, never synthetic observations. Strict continuous metric additionally stops at a gap; legacy evaluator also retained verbatim for comparability.',
      tentative='At most lookahead pending segments; single-real-position support cannot be immediately confirmed. Promotion requires subsequent distinct spatial observation and joint continuity. Unresolved pending returns stay tentative.',
      registration='Estimated saved registration; original XYZ and source records never modified. Optional dv/dw correction affects candidate matching only and is audited.',
      range_limit_m=128,gt_uncertainty='Existing imperfect class2 and registered future surface, not independent survey truth',self_mask='Deferred by user; no invented train geometry',
      runtime='Sequential fresh process, warm/uncontrolled OS file cache; report processing and I/O separately; no competing study jobs during final timing.'))
    shutil.copyfile(r'C:\Users\heinrich.wirth\.codex\attachments\92fbaa26-d1d8-4099-8fd6-0369c5b21b57\Вставленный текст.txt',STAGE/'REQUEST_RU.txt')
    print('Prepared',len(chosen),'screen',len(configurations()),'configs; immutable inventory',len(inventory))

if __name__=='__main__':main()
