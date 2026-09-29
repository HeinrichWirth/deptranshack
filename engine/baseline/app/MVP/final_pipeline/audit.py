"""Integration gate: old-data dependency tests, never used by production inference."""
from . import ROOT
import sys
import platform
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import long_common as lc

OUT = ROOT / 'results_final_pipeline'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pins():
    stages = ['01_rail2d', '04_contact_rail_final', '05_contact_rail_marching',
              '05_1_contact_marching_temporal_fusion', '05_2_contact_marching_long_range',
              '06_running_rails_from_contact', '06_1_latent_contact_reference',
              '06_final_geometry']
    files = {}
    for name in stages:
        directory = ROOT / 'MVP/stages' / name
        candidates = list(directory.glob('*.json')) + list(directory.glob('*.md'))
        for sub in ('src', 'config', 'contact_rail_step2'):
            if (directory / sub).exists():
                candidates += [p for p in (directory / sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        for p in candidates:
            files[p.relative_to(ROOT).as_posix()] = lc.sha(p)
    for p in [ROOT/'results_contact_marching/config.json', ROOT/'results_contact_marching_long_range/configs.json',
              ROOT/'results_contact_marching_long_range/research_freeze.json',
              ROOT/'MVP/pipelines/frozen_classification_common.py',
              ROOT/'MVP/stages/06_3_final_geometry_rescue/FINAL_RECOMMENDATION.md']:
        files[p.relative_to(ROOT).as_posix()] = lc.sha(p)
    return files


def verify_freezes():
    rows = []
    for stage in ('01_rail2d', '04_contact_rail_final', '06_final_geometry'):
        path = ROOT/'MVP/stages'/stage
        manifest = lc.load(path/'MANIFEST.json')['files']
        items = [(r['path'], r['sha256']) for r in manifest] if isinstance(manifest, list) else manifest.items()
        for file, expected in items:
            assert lc.sha(path/file) == expected, (stage, file)
        rows.append(dict(stage=stage, verified=True, files=len(manifest)))
    freeze = lc.load(ROOT/'results_contact_marching_long_range/research_freeze.json')
    for name, expected in freeze['code_sha256'].items():
        assert lc.sha(ROOT/'MVP/stages/05_2_contact_marching_long_range/src'/name) == expected
    assert lc.sha(ROOT/'results_contact_marching_long_range/configs.json') == freeze['config_sha256']
    rows.append(dict(stage='C4', verified=True, files=len(freeze['code_sha256'])+1))
    return rows


def main():
    (OUT/'audit').mkdir(parents=True, exist_ok=True)
    before = pins()
    lc.save(OUT/'audit/dependencies_before.json', before)
    freezes = verify_freezes()
    # This module is an explicitly offline forensic test, permitted to inspect
    # labels in old saved inputs. The public RAW pipeline does not import it.
    sys.path[:0] = [str(ROOT/'MVP/stages/06_3_final_geometry_rescue/src'),
                    str(ROOT/'MVP/stages/06_final_geometry/src')]
    from rescue_common import read_causal, checked_base
    from seed_input import calibrate_seed
    from final_geometry import predict_geometry
    rows = []
    for r in lc.load(ROOT/'results_final_geometry_rescue/protocol.json')['screen']:
        run, i = r['run'], r['frame']
        m, a, p, pr = read_causal(run, i)
        directory = ROOT/'results_running_rails_from_contact/c4_cr/seed8'/lc.key(run, i)
        with np.load(directory/'input.npz') as z:
            xyz, labels = z['seed_points'], z['seed_labels']
        model = dict(s=a['s'], C=a['C'], B=a['B'], knots=a['anchor_knots'], anchors=a['anchor_xyz'])
        original = calibrate_seed(xyz, labels, model, a['initial_basis'], m['q'], 8.)
        no_labels = calibrate_seed(xyz, np.zeros_like(labels), model, a['initial_basis'], m['q'], 8.)
        assert original['status'] == m['seed']['status'] == 'AVAILABLE'
        np.testing.assert_array_equal(original['x0'], np.asarray(m['seed']['x0']))
        assert no_labels['status'] == 'SEED_TOO_SPARSE'
        started = time.perf_counter()
        prediction, curve = predict_geometry(m['seed'], a, m['q'], pr['source_frame'], i)
        milliseconds = (time.perf_counter()-started)*1000
        base, oldcurve = checked_base(run, i)
        assert set(prediction) == set(base)
        errors = []
        for name in prediction:
            np.testing.assert_array_equal(prediction[name], base[name], err_msg=name)
            d = abs(prediction[name].astype(float)-base[name].astype(float))
            errors.append(float(np.nanmax(d)) if np.isfinite(d).any() else 0.)
        np.testing.assert_array_equal(curve['covariance'], oldcurve['covariance'])
        rows.append(dict(run=run, frame=i, original_seed=original['status'],
                         zero_labels_seed=no_labels['status'], label1_points=int(np.sum(labels==1)),
                         saved_seed_exact=True, geometry_exact=True, max_difference=max(errors),
                         prepared_geometry_ms=milliseconds, scope='PREPARED_INPUT_ONLY_NOT_RAW_E2E'))
        print('SEED_AUDIT',len(rows), run, i, no_labels['status'], flush=True)
    after = pins()
    assert before == after, 'Frozen dependencies changed'
    lc.save(OUT/'audit/seed_contract.json', dict(cases=len(rows), rows=rows,
        original_seed_reproduced=True, all_zero_label_seeds_unavailable=True,
        geometry_with_old_seed_bitwise_equal=True, raw_quality_reproduced=False,
        dependency_files_unchanged=len(before), freezes=freezes))
    lc.save(OUT/'audit/dependencies_after.json', after)
    table = []
    for logical, stage, config, version in [
        ('STEP1', '01_rail2d', 'config/detector_v2.json', 'rail2d_head_v2'),
        ('STEP2', '04_contact_rail_final', 'contact_rail_step2/config.json', '1.0.0-confidence-pre-guard.1'),
        ('C4', '05_2_contact_marching_long_range', '../../../results_contact_marching_long_range/configs.json', 'research_freeze C4'),
        ('C4_SMOOTH + STEP6', '06_final_geometry', 'config.json', 'HACKATHON_FINAL_CANDIDATE')]:
        path = ROOT/'MVP/stages'/stage
        code = {k:v for k,v in before.items() if k.startswith('MVP/stages/'+stage+'/') and k.endswith('.py')}
        table.append(dict(logical_stage=logical, actual_path=str(path), config=config, version=version,
            code_sha256=digest(code), config_sha256=lc.sha(path/config), dependency_sha256=digest(before)))
    lc.csv_write(OUT/'audit/frozen_dependencies.csv', table)
    lc.save(OUT/'FINAL_PIPELINE_FREEZE.json', dict(status='INTEGRATION_FAILED_NOT_RELEASED',
        algorithmic_stages_changed=False, quality_reproduced=False,
        blockers=['STEP6 seed construction reads current class1 labels', 'Frozen bootstrap requires T+1 pose'],
        stages=table, dependency_files=before, dependency_sha256=digest(before),
        python=sys.version, os=platform.platform(), cpu=platform.processor(),
        thread_settings=dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1),
        successful_freeze_marker_created=False, time_ns=time.time_ns()))


if __name__ == '__main__':
    main()
