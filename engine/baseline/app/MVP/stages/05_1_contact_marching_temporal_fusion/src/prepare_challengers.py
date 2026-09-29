from fusion_common import *
if __name__=='__main__':
    assert not (OUT/'research_lock.json').exists()
    cfg=load(OUT/'configs.json')
    # Corrected weight conservation at the deterministic 2mm fitting cells.
    # Preserve the preliminary files separately; never mix their statistics.
    for name in ('F4_BAL','F4_TAU0.5','F4_TAU1','F4_TAU2'):
        p=OUT/'development'/name;target=OUT/'calibration'/'weighted_preliminary'/name
        if p.exists():target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
    for b in (.02,.05):cfg['F4_ALIGN'+str(int(b*100))]=dict(frames=4,alignment_bound=b,diagnostic=True)
    cfg['F4_BOOT']=dict(frames=4,fused_bootstrap=True,diagnostic=True)
    save(OUT/'configs.json',cfg)
