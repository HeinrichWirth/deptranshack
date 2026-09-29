"""Finite staged ablations, specified before any new benchmark evaluation."""
from long_common import *

DEFAULT = dict(history='fixed',frames=1,geometry='T',overlap='hard',overlap_limit=.03,
    support=3,tentative=False,lookahead=1,gaps=0,beam=1,prior='P0',window_mode='fixed',window=8.,advance=4.,
    shape_limit=.025,anchor_gate=.03,relaxation='none',partial='full',evidence='count',consensus=1,
    current_veto=False,reg_gate=None,alignment=0.,alpha=1.,smooth=0,backward=False,dewarp=False,
    max_steps=40,max_range=128.,max_history=24,score_weight=1.,graph=False,tube=False)

def configurations():
    c={f'B{n}':dict(baseline=True,frames=max(n,1)) for n in (0,2,4,8)}
    def add(name,**kw): c[name]=dict(DEFAULT,**kw)
    add('A0')
    for n in (2,4,6,8,12,16,24): add(f'AF{n}',frames=n)
    for n in (2,4,8,12,16): add(f'AF{n}_B',frames=n,geometry='accepted')
    for name,hs in [('SF24',[1,2,4]),('SF248',[1,2,4,8]),('SF48',[1,4,8]),('AFUSION',[1,2,4,8,12,16])]: add(name,history='fallback',levels=hs,frames=max(hs))
    add('RFUSION',history='range',frames=16)
    for n in (2,1): add(f'TENTATIVE{n}',support=n,tentative=True,lookahead=2)
    for n in (1,2): add(f'GAP{n}',gaps=n,tentative=True,lookahead=2)
    for n in (2,3): add(f'LOOK{n}',tentative=True,lookahead=n,overlap='hysteresis')
    for n in (2,3,5): add(f'BEAM{n}',beam=n,tentative=True,lookahead=2,gaps=1,anchor_gate=.08)
    add('GRAPH',beam=5,tentative=True,lookahead=3,gaps=2,anchor_gate=.12,graph=True)
    for p in ('P1','P2','P3'): add(p,prior=p)
    for w in (12,16,24):
        add(f'W{w}',window=float(w))
        add(f'DEWARP{w}',window=float(w),dewarp=True,prior='P1')
    for a in (4,6,8): add(f'DYNAMIC_A{a}',window_mode='range',advance=float(a))
    add('ADAPTIVE_W',window_mode='support')
    for mode in ('q80','trimmed','fraction','soft','regularized','hysteresis'): add('OVERLAP_'+mode,overlap=mode,tentative=mode in ('soft','hysteresis'),lookahead=2)
    for cm in (3.5,4,5,6): add(f'OVERLAP_{cm:g}',overlap_limit=cm/100)
    for cm in (2,3,4,5,6): add(f'SHAPE_{cm}',shape_limit=cm/100)
    for cm in (5,8,12): add(f'ANCHOR_{cm}',anchor_gate=cm/100)
    for mode in ('vertical','horizontal','corner','subset'): add('PARTIAL_'+mode,partial=mode,tentative=True,lookahead=2)
    for n in (1,2,3,4): add(f'CONSENSUS{n}',frames=8,consensus=n,tentative=True,lookahead=2)
    for mode in ('diversity','cells','balanced'): add('EVIDENCE_'+mode,frames=8,evidence=mode,tentative=True,lookahead=2,support=2)
    for n in (.75,.5,.25): add(f'ALPHA{n:g}',alpha=n)
    for n in (3,5,8,12): add(f'SMOOTH{n}',smooth=n,prior='P2')
    add('BACKWARD',backward=True,tentative=True,lookahead=2,prior='P1')
    add('REFIT',smooth=12,prior='P3',backward=True)
    add('TUBE',tube=True,prior='P2',smooth=8,anchor_gate=.08,tentative=True,lookahead=2)
    add('TRACKAGE',relaxation='age',tentative=True,lookahead=2,support=2)
    add('NEW_ONLY_RELAX',frames=8,relaxation='age',tentative=True,lookahead=2,support=2,strict_old=True)
    add('RANGE_RELAX',relaxation='range',tentative=True,lookahead=2,support=2)
    for w in (.75,1.,1.25): add(f'SCORE{w:g}',evidence='score',score_weight=w,tentative=True,lookahead=2,support=2)
    add('VETO',frames=8,current_veto=True)
    for n in (2,3,4,5): add(f'REG{n}',frames=16,reg_gate=n/100)
    for f in (8,12,16):
        for cm in (1,2,3): add(f'ALIGN{f}_{cm}',frames=f,alignment=cm/100)
    add('KF_DISP',history='keyframe',frames=8)
    add('KF_UNIFORM',history='uniform',frames=8)
    add('KF_COVERAGE',history='coverage',frames=8)
    return c

def combinations():
    return {
      'C1':dict(DEFAULT,frames=4,tentative=True,lookahead=2,overlap='soft'),
      'C2':dict(DEFAULT,history='fallback',levels=[1,2,4,8,12,16],frames=16,tentative=True,lookahead=2,gaps=1),
      'C3':dict(DEFAULT,frames=8,current_veto=True,relaxation='age',tentative=True,lookahead=2,support=2),
      'C4':dict(DEFAULT,frames=12,beam=3,partial='subset',tentative=True,lookahead=2,anchor_gate=.08),
      'C5':dict(DEFAULT,history='fallback',levels=[1,2,4,8,12,16],frames=16,window_mode='support',beam=3,tentative=True,lookahead=2,geometry='accepted'),
      'RING_BEAM05':dict(DEFAULT,beam=3,tentative=True,lookahead=2,gaps=1,anchor_gate=.08,ring_score=.05),
      'RING_BEAM10':dict(DEFAULT,beam=3,tentative=True,lookahead=2,gaps=1,anchor_gate=.08,ring_score=.10),
      'WEIGHTS075':dict(DEFAULT,beam=3,tentative=True,lookahead=2,gaps=1,anchor_gate=.08,score_balance=.75),
      'WEIGHTS125':dict(DEFAULT,beam=3,tentative=True,lookahead=2,gaps=1,anchor_gate=.08,score_balance=1.25),
      'CURVATURE_DELTA':dict(DEFAULT,beam=3,tentative=True,lookahead=2,gaps=1,anchor_gate=.08,curvature_change=True),
      'CURVATURE_DELTA_F8':dict(DEFAULT,frames=8,beam=3,tentative=True,lookahead=2,gaps=1,anchor_gate=.08,curvature_change=True),
    }
