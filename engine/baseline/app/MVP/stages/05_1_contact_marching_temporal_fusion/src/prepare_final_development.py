from fusion_common import *
from select_development import summaries,choose
if __name__=='__main__':
    rows=summaries();selected=choose(rows);cfg=load(OUT/'configs.json');name=selected['variant']
    cfg['M2_F0']=dict(frames=1,method='M2',diagnostic=True);cfg['M2_BEST']=dict(cfg[name],method='M2',diagnostic=True)
    cfg['FUSED_BOOTSTRAP']=dict(cfg[name],fused_bootstrap=True,diagnostic=True)
    for age in (1,2,3):cfg[f'BEST_DROP{age}']=dict(cfg[name],exclude_ages=[age],diagnostic=True)
    f4=next(r for r in rows if r['variant']=='F4')
    if f4['wins']>f4['losses'] and f4['mean_delta']>0:
        for n in (6,8):cfg[f'F{n}']=dict(frames=n,diagnostic=True)
    save(OUT/'configs.json',cfg);save(OUT/'calibration/selection_before_challengers.json',selected)
    print('SELECTED',selected,'F4 CONDITION',f4['mean_delta'],f4['wins'],f4['losses'],flush=True)
