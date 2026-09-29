"""RAW pipeline with an authorized unlabeled seed adapter and one-pose delay.

Frozen two-pose bootstrap is called after T+1 pose arrives. No future cloud is
opened. strict_current_past remains an explicit audit of the old dependency.
"""
from . import ROOT
import json
import time
from pathlib import Path
import numpy as np
import long_common as lc
from fusion import CausalSource, sanitized
from roi_source import RoiSource
from bootstrap import initial_frame, pose_record
from rail2d_head_v2 import detect
from frozen_classification_common import surface_support
from contact_rail_step2 import ContactRailDetector
from long_tracker import infer
from .adapters.geometry_input import build_geometry
from .adapters.step1_to_step6_seed import build_seed
from .timing import geometry_call

SEED_BLOCKER = 'STEP6_GT_FREE_SEED_ADAPTER_UNAVAILABLE'


class TimedSource(CausalSource):
    def __init__(self, run, index, records, cache=None, allow_disk=True):
        # Supplying a prefix prevents the frozen constructor reading T+1.
        super().__init__(run, index, records[:index + 1])
        self.read_times = []
        self.cache=cache;self.allow_disk=allow_disk

    def read(self, k):
        if self.cache is None:part,timing=super().read(k)
        else:
            self.accessed.append(k)
            part,timing=self.cache.read(k,self.i,self.allow_disk)
        self.read_times.append(timing)
        for value in part.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return part, timing


class RawRoi(RoiSource):
    def __init__(self, run, index, records, cache=None, allow_disk=True):
        self.source = TimedSource(run, index, records,cache,allow_disk)
        self.i, self.run = index, run
        self.parts, self.trees, self.timers, self.used = {}, {}, {}, {}
        self.audit, self.processed = [], 0
        self.current = self.read(index)


def provenance(source, detail, pred, cloud):
    """Compact unique point provenance for this invocation, exact source IDs.

Stages 4/5 are not executed, so their mask bits are never asserted.
"""
    dtype = np.dtype([('source_frame', '<i4'), ('source_row', '<u4'),
                      ('source_point_index', '<i8'), ('first_found_stage', 'u1'),
                      ('used_by_stage_mask', '<u2'), ('pipeline_component', 'u1'),
                      ('confidence', '<f8'), ('march_step', '<i4')])
    rail = detail.get('rail_support', np.empty(0, dtype=int))
    contact = detail.get('contact_support', np.empty(0, dtype=int))
    selected = pred['indices']
    keys = np.concatenate(((np.uint64(source.i) << np.uint64(32)) | rail.astype('uint64'),
                           (np.uint64(source.i) << np.uint64(32)) | contact.astype('uint64'),
                           cloud['keys'][selected]))
    unique, inverse = np.unique(keys, return_inverse=True)
    out = np.zeros(len(unique), dtype=dtype)
    out['source_frame'] = unique >> np.uint64(32)
    out['source_row'] = unique & np.uint64(0xffffffff)
    out['confidence'] = np.nan
    out['march_step'] = -1
    cursor = 0
    groups = [(1, rail, source.current, rail),
              (2, contact, source.current, contact),
              (3, selected, cloud, selected)]
    for stage, ids, data, data_ids in groups:
        if not len(ids):
            continue
        target = inverse[cursor:cursor + len(ids)]
        cursor += len(ids)
        fresh = out['first_found_stage'][target] == 0
        new = target[fresh]
        out['first_found_stage'][new] = stage
        np.bitwise_or.at(out['used_by_stage_mask'], target, 1 << (stage - 1))
        out['source_point_index'][target] = data['source_point_index'][data_ids]
        if stage == 1:
            # Near/far requires CR side, otherwise leave UNKNOWN rather than guess.
            if pred['seed'].get('status') == 'AVAILABLE':
                vw = source.current['xyz'][ids] @ pred['seed']['basis'][:, 1:]
                pair = np.asarray(detail['step1']['pair'])
                which = np.argmin(np.linalg.norm(vw[:, None] - pair[None], axis=2), axis=1)
                near = 1 if pred['seed']['side'] > 0 else 0
                out['pipeline_component'][new] = np.where(which[fresh] == near, 1, 2)
            out['confidence'][new] = detail['step1'].get('confidence', np.nan)
        elif stage == 2:
            out['pipeline_component'][new] = 3
            out['confidence'][new] = detail['step2']['confidence']
        else:
            out['pipeline_component'][new] = 3
            out['confidence'][new] = pred['confidence'][fresh]
            out['march_step'][new] = pred['point_step'][fresh]
    return out


class Pipeline:
    def __init__(self, config=None):
        initialize=time.perf_counter()
        self.config = dict(config or {})
        self.config.setdefault('engine','cache')
        self.contact = ContactRailDetector()
        self.rail_config = lc.load(ROOT / 'MVP/stages/01_rail2d/config/detector_v2.json')
        self.c4 = lc.load(ROOT / 'results_contact_marching_long_range/configs.json')['C4']
        self.records = None
        self.startup_times={'T_CONFIG_LOAD':time.perf_counter()-initialize}

    def start_run(self, run_dir):
        begin=time.perf_counter()
        directory = Path(run_dir).resolve()
        # The frozen file source is tied to the pinned dataset; refuse substitution.
        if directory.parent != lc.dataset().resolve() and self.config['engine']=='baseline':
            raise ValueError('Legacy baseline reader is pinned to its dataset; use engine=cache for another registered LAS run')
        indexes = list(directory.glob('*.frames.json'))
        if len(indexes) != 1:
            raise ValueError('Expected exactly one frame metadata index')
        self.startup_times['T_FILE_DISCOVERY']=time.perf_counter()-begin
        begin=time.perf_counter()
        self.run = directory.name
        self.records = [sanitized(r) for r in lc.load(indexes[0])['frames']]
        self.startup_times['T_METADATA']=time.perf_counter()-begin
        self.last_index = -1
        self.results = []
        from .optimization.cache import HistoryCache
        self.cache=None if self.config.get('engine','baseline')=='baseline' else HistoryCache(directory,self.records,metadata=self.config.get('engine')!='raw_cache')
        return dict(run=self.run, frames=len(self.records), status='RUN_STARTED',pose_latency_frames=1)

    def process_frame(self, index):
        if self.records is None:
            raise RuntimeError('Call start_run first')
        if not 0 <= index < len(self.records) or index <= self.last_index:
            raise ValueError('Frames must be processed in increasing causal order')
        self.last_index = index
        started, cpu = time.perf_counter(), time.process_time()
        timing = {}
        if self.cache is not None:self.cache.advance(index)
        source = RawRoi(self.run, index, self.records,self.cache,allow_disk=not self.config.get('preloaded',False))
        xyz = source.current['xyz']
        now = time.perf_counter()
        delayed = self.config.get('pose_policy', 'frozen_delayed_pose_audit') == 'frozen_delayed_pose_audit'
        b = pose_record(self.records[index + 1]) if delayed and index + 1 < len(self.records) else None
        basis, reason = initial_frame(pose_record(self.records[index]), b)
        timing['T_POSES']=time.perf_counter()-now
        now=time.perf_counter()
        seed = dict(status='START_UNAVAILABLE', reason=reason, seed_ms=0., basis=basis,
                    indices=np.empty(0, dtype=np.int64))
        detail = {}
        if basis is not None:
            u = xyz @ basis[:, 0]
            ids = np.flatnonzero((u >= 0) & (u <= 8) & np.isfinite(xyz).all(axis=1))
            vw = xyz[ids] @ basis[:, 1:]
            r = detect(vw, self.rail_config, details=False)
            detail.update(step1=r, rail_support=ids[surface_support(vw, r, self.rail_config)])
            seed.update(step1_status=r['status'], step1_reason=r['reason'])
        timing['T_STEP1'] = time.perf_counter() - now
        now = time.perf_counter()
        if detail.get('step1', {}).get('status') == 'ok':
            uvw = xyz @ basis
            ids = np.flatnonzero((uvw[:, 0] >= 0) & (uvw[:, 0] <= 8) & np.isfinite(uvw).all(axis=1))
            q = self.contact.detect({'uvw': uvw[ids]}, r['pair'])
            detail['step2'] = q
            seed.update(step2_status=q['status'], confidence=q['confidence'], diagnostics=q['diagnostics'])
            if q['status'] in ('LEFT', 'RIGHT'):
                detail['contact_support'] = ids[q['support_indices']]
                seed.update(status='AVAILABLE', reason='', indices=detail['contact_support'],
                            side=-1 if q['status'] == 'LEFT' else 1,
                            anchor=np.array([0., q['anchor_v'], q['anchor_w']]) @ basis.T, seed_end=8.)
            else:
                seed['reason'] = 'STEP2_' + q['status'] + ':' + q['diagnostics']['reason']
        elif basis is not None:
            seed['reason'] = 'STEP1_' + r['reason']
        timing['T_STEP2'] = time.perf_counter() - now
        seed['seed_ms'] = 1000 * (timing['T_POSES']+timing['T_STEP1'] + timing['T_STEP2'])
        now = time.perf_counter()
        pred, cloud = infer(source, seed, self.contact.template, self.c4)
        timing['T_C4_MARCHING_INCLUSIVE_HISTORY_IO'] = time.perf_counter() - now
        final, curve, geometry, running_seed = None, None, None, None
        seed_members=np.empty(0,dtype=np.int64)
        geometry_reason='C4_SEED_UNAVAILABLE'
        if seed['status']=='AVAILABLE' and not self.config.get('partial_audit',False):
            now=time.perf_counter()
            model, geometry=build_geometry(pred,cloud,self.records[index]['lidar_pose_in_folder'])
            timing['T_GEOMETRY_CONTRACT']=time.perf_counter()-now
            geometry_reason='C4_SHORT_HORIZON'
            if model is not None:
                now=time.perf_counter()
                running_seed,seed_members=build_seed(xyz,detail['rail_support'],r['pair'],model,basis,seed['side'])
                timing['T_SEED_ADAPTER']=time.perf_counter()-now
                geometry_reason=running_seed['status']
                if running_seed['status']=='AVAILABLE':
                    (final,curve),times=geometry_call(running_seed,geometry,seed['side'],
                        cloud['source_frame'][pred['indices']],index)
                    timing.update(times)
                    geometry_reason='FINAL_GEOMETRY_AVAILABLE'
        now = time.perf_counter()
        prov = provenance(source, detail, pred, cloud)
        if final is not None:
            keys=(prov['source_frame'].astype('uint64')<<np.uint64(32))|prov['source_row'].astype('uint64')
            crkeys=cloud['keys'][pred['indices'][pred['point_state']==2]]
            # Transitive evidence use by smoothing and reconstruction is explicit.
            prov['used_by_stage_mask'][np.searchsorted(keys,crkeys)] |= 8|16
            # Tentative C4 evidence also contributes to the frozen cr_state
            # contract consumed by STEP6, though it does not anchor smoothing.
            prov['used_by_stage_mask'][(prov['used_by_stage_mask']&4)!=0] |= 16
            railkeys=(np.uint64(index)<<np.uint64(32))|seed_members.astype('uint64')
            prov['used_by_stage_mask'][np.searchsorted(keys,railkeys)] |= 16
        timing['T_PROVENANCE'] = time.perf_counter() - now
        timing['T_LAS_IO'] = sum(x['read_ms'] for x in source.source.read_times) / 1000
        timing['T_TRANSFORMS'] = sum(x['transform_ms'] for x in source.source.read_times) / 1000
        timing['T_C4_MARCHING']=timing['T_C4_MARCHING_INCLUSIVE_HISTORY_IO']-sum((x['read_ms']+x['transform_ms'])/1000 for x in source.source.read_times[1:])
        timing['T_TOTAL'] = time.perf_counter() - started
        timing['CPU_TOTAL'] = time.process_time() - cpu
        status = geometry_reason if seed['status'] == 'AVAILABLE' else (
            'STEP2_CONTACT_NOT_FOUND' if detail.get('step1', {}).get('status') == 'ok' else 'STEP1_UNAVAILABLE')
        row = dict(run=self.run, frame=index, status=status, original_reason=pred['reason'],
                   final_geometry_available=final is not None,
                   pipeline_component='DEPLOYABLE_PIPELINE',
                   pose_policy=self.config.get('pose_policy', 'frozen_delayed_pose_audit'),
                   pose_wait_seconds=None if b is None else (b['time_ns']-int(self.records[index]['header_time_ns']))/1e9,
                   source_points=len(xyz), source_frames=sorted(set(source.source.accessed)),
                   available_c4_observed_range=pred.get('max_confirmed_observed_range', 0),
                   timing=timing, selected_real_points=len(prov))
        row['startup_times']=dict(self.startup_times) if not self.results else {}
        if self.cache is not None:row.update(cache_bytes=self.cache.memory_bytes(),physical_reads_total=self.cache.reads,
                                            physical_bytes_read=self.cache.bytes_read)
        self.results.append(row)
        near=1 if seed.get('side')==1 else 0
        return dict(summary=row, step1=detail.get('step1'), step2=detail.get('step2'),
                    c4=pred, provenance=prov, cloud=cloud, c4_smooth=curve,
                    seed=running_seed,geometry_input=geometry,prediction=final,
                    left_rail=None if final is None else final['pair'][:,near],
                    right_rail=None if final is None else final['pair'][:,1-near],
                    centerline=None if final is None else final['pair'].mean(axis=1))

    def finish_run(self):
        return dict(status='RUN_COMPLETE',frames=self.results,
                    final_geometry_frames=sum(r['final_geometry_available'] for r in self.results))


def write_frame(directory, result, mode='research'):
    directory.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    lc.save(directory / 'summary.json', result['summary'])
    p = result['c4']
    if mode=='research':
        lc.save(directory / 'short_stages.json', dict(step1=result['step1'], step2=result['step2']))
        lc.save(directory / 'c4.json', {k:v for k,v in p.items() if not isinstance(v, np.ndarray)})
        np.savez_compressed(directory / 'c4.npz', **{k:v for k,v in p.items() if isinstance(v, np.ndarray)})
    else:
        lc.save(directory/'stage_statuses.json',dict(step1=None if result['step1'] is None else result['step1']['status'],
            step2=None if result['step2'] is None else result['step2']['status'],c4_status=p['status'],c4_reason=p['reason']))
    np.savez_compressed(directory / 'point_provenance.npz', rows=result['provenance'])
    lc.save(directory/'seed.json',result['seed'])
    for name,value in (('geometry_input',result['geometry_input']),('prediction',result['prediction']),('curve',result['c4_smooth'])):
        if name=='geometry_input' and mode!='research':continue
        if value is not None:np.savez_compressed(directory/(name+'.npz'),**value)
    names=['summary.json','point_provenance.npz','seed.json']
    names+=['short_stages.json','c4.json','c4.npz'] if mode=='research' else ['stage_statuses.json']
    names += [name+'.npz' for name,value in (('geometry_input',result['geometry_input']),('prediction',result['prediction']),('curve',result['c4_smooth'])) if value is not None and (name!='geometry_input' or mode=='research')]
    result['summary']['timing']['T_SERIALIZATION']=time.perf_counter()-started
    lc.save(directory/'summary.json',result['summary'])
    lc.save(directory/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),
        files={name:lc.sha(directory/name) for name in names},
        pipeline='DEPLOYABLE_PIPELINE',uses_annotation_labels=False,uses_future_clouds=False))


def process_run(run_dir, config=None):
    config = dict(config or {})
    pipeline = Pipeline(config)
    pipeline.start_run(run_dir)
    output = Path(config.get('output', ROOT / 'results_final_pipeline/partial_runs'))
    output = output.resolve()
    output.relative_to(ROOT.resolve())
    if output == Path(run_dir).resolve() or output.is_relative_to(Path(run_dir).resolve()):
        raise ValueError('Output must not be inside the source run')
    # Never permit frozen-stage writes through the public CLI.
    if output.is_relative_to((ROOT / 'MVP/stages').resolve()):
        raise ValueError('Frozen stage directories are read-only')
    indices = config.get('indices', range(len(pipeline.records)))
    for index in indices:
        result = pipeline.process_frame(index)
        write_frame(output / pipeline.run / f'{index:06d}', result,config.get('mode','online'))
        print(json.dumps(result['summary']), flush=True)
    summary = pipeline.finish_run()
    lc.save(output / pipeline.run / 'RUN_RESULT.json', summary)
    return summary
