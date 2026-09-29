"""Supported upper-head selection from a single unlabelled 2D cloud.

Candidate generation is inherited from the frozen baseline. Vertical alternatives
are resolved before density ranking; gauge couples both *observed* heads. This
module never reads labels, poses, filenames, frame identities, maps, or GT.
"""
import numpy as np
from scipy.spatial import cKDTree
from time import perf_counter
from rail2d_detector import candidates

HEAD_DEFAULT = dict(surface_half_width_m=.055, surface_half_height_m=.020,
    surface_cell_m=.010, surface_min_points=8, surface_min_cells=3,
    upper_lateral_tolerance_m=.090, upper_min_rise_m=.035, upper_max_rise_m=.28,
    upper_min_support_ratio=.15, support_cap=80., support_weight=.4,
    balance_weight=.15, surface_weight=.4, margin_scale=.25,
    head_confidence_threshold=.25)


def compatible(a, b, cfg):
    return (abs((a[0]+b[0])*.5) <= cfg['center_max'] and
            1.30 <= b[0]-a[0] <= 1.80 and abs(b[1]-a[1]) <= cfg['cant_max'])


def detect(points, cfg, details=False):
    start = perf_counter()
    cfg = dict(HEAD_DEFAULT, **cfg)
    p = np.asarray(points, dtype=float)
    left, right, density_ms, _ = candidates(p, cfg, 'density')
    roi = p[(p[:, 0]>=cfg['vmin']) & (p[:, 0]<=cfg['vmax']) &
            (p[:, 1]>=cfg['wmin']) & (p[:, 1]<=cfg['wmax'])]
    tree = cKDTree(roi) if len(roi) else None
    for side in (left, right):
        for c in side:
            center = c['center']
            near = roi[tree.query_ball_point(center, np.hypot(cfg['surface_half_width_m'], cfg['surface_half_height_m']))]
            band = near[(abs(near[:, 0]-center[0])<=cfg['surface_half_width_m']) &
                        (abs(near[:, 1]-center[1])<=cfg['surface_half_height_m'])]
            # Occupied neighbouring lateral cells reject isolated high outliers.
            cells, counts = np.unique(np.floor((band[:, 0]-center[0])/cfg['surface_cell_m']).astype(int), return_counts=True)
            cells = cells[counts >= 2]
            longest = 0; streak = 0; previous = -10000
            for cell in cells:
                streak = streak+1 if cell == previous+1 else 1
                longest = max(longest, streak); previous = cell
            c.update(surface_points=len(band), surface_cells=int(longest),
                     surface_ok=bool(len(band)>=cfg['surface_min_points'] and longest>=cfg['surface_min_cells']),
                     surface_quality=float(min(1., len(band)/20.)*min(1., longest/6.)))
    options = []
    for li, l in enumerate(left):
        for ri, r in enumerate(right):
            if not l['surface_ok'] or not r['surface_ok'] or not compatible(l['center'], r['center'], cfg): continue
            pair = np.array([l['center'], r['center']])
            replacements = []
            # A lower candidate is dominated only if a supported upper candidate
            # remains compatible with the SAME observed opposite rail.
            for k, side, chosen, other in [(0, left, l, r), (1, right, r, l)]:
                for index, upper in enumerate(side):
                    dv, dw = upper['center']-chosen['center']
                    supported = (upper['surface_ok'] and abs(dv)<=cfg['upper_lateral_tolerance_m'] and
                        cfg['upper_min_rise_m']<=dw<=cfg['upper_max_rise_m'] and
                        upper['support']>=max(cfg['min_support'],cfg['upper_min_support_ratio']*min(chosen['support'],cfg['support_cap'])))
                    fits = compatible(upper['center'], other['center'], cfg) if k==0 else compatible(other['center'], upper['center'], cfg)
                    if supported and fits:
                        replacements.append((k,index))
            a, b = min(l['support'],cfg['support_cap']), min(r['support'],cfg['support_cap'])
            g = pair[1,0]-pair[0,0]; c = pair[:,0].mean(); h = pair[:,1].mean(); dz = pair[1,1]-pair[0,1]
            outside = max(cfg['gauge_soft'][0]-g, 0, g-cfg['gauge_soft'][1])
            terms = dict(support=cfg['support_weight']*(np.log1p(a)+np.log1p(b)),
                balance=cfg['balance_weight']*min(a,b)/max(a,b),
                surface=cfg['surface_weight']*(l['surface_quality']+r['surface_quality']),
                gauge=-.12*((g-cfg['gauge_m'])/.07)**2-2*(outside/.07)**2,
                center=-.12*(c/.30)**2, height=-.08*((h-cfg['height_m'])/.35)**2,
                cant=-.10*(dz/.18)**2, shape=-.25*(l['scatter']+r['scatter'])/.05)
            options.append(dict(pair=pair,score=float(sum(terms.values())),left=li,right=ri,
                                upper_replacements=replacements,terms=terms))
    kept = sorted([o for o in options if not o['upper_replacements']],key=lambda o:o['score'],reverse=True)
    result = dict(pair=None,confidence=0.,status='refusal',reason='no_supported_upper_pair',
                  candidates_left=len(left),candidates_right=len(right),
                  surface_candidates_left=sum(c['surface_ok'] for c in left),
                  surface_candidates_right=sum(c['surface_ok'] for c in right),
                  lower_pairs_suppressed=sum(bool(o['upper_replacements']) for o in options),
                  head_pairs=len(kept),density_ms=density_ms)
    if kept:
        chosen = kept[0]; l=left[chosen['left']];r=right[chosen['right']]
        alt = next((o for o in kept[1:] if np.linalg.norm(o['pair']-chosen['pair'],axis=1).max()>.06), None)
        margin = max(0.,chosen['score']-alt['score']) if alt else 3.
        support = min(l['support'],r['support'])
        confidence = float((1-np.exp(-support/20.))*(1-np.exp(-margin/cfg['margin_scale'])))
        result.update(pair=chosen['pair'],confidence=confidence,status='ok',reason='',margin=margin,
            score=chosen['score'],support_left=l['support'],support_right=r['support'],
            surface_left=l['surface_points'],surface_right=r['surface_points'],
            surface_cells_left=l['surface_cells'],surface_cells_right=r['surface_cells'],
            selected_left_index=chosen['left'],selected_right_index=chosen['right'])
        if confidence<cfg['head_confidence_threshold']:
            result.update(status='refusal',reason='ambiguous_upper_pair')
    if details: result.update(left=left,right=right,options=options)
    result['detector_ms']=(perf_counter()-start)*1000
    result['search_ms']=result['detector_ms']-density_ms
    return result
