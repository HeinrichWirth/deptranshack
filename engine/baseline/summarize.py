"""Every reported duration is milliseconds; inclusive timers are identified."""
import json,sys,collections
from pathlib import Path
import numpy as np

def stats(a):
    return dict(n=len(a),median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)),max_ms=float(np.max(a))) if a else None

for directory in map(Path,sys.argv[1:]):
    rows=[json.loads(s) for s in (directory/'arrivals.jsonl').read_text().splitlines()];summary=json.loads((directory/'SUMMARY.json').read_text());solves=json.loads((directory/'solves.json').read_text());columns=collections.defaultdict(list)
    for row in rows:
        for name in ('read_ms','decode_ms','payload_hash_ms','registration_ms','frame_data_ms','producer_ms','core_arrival_ms','consumer_ms'):
            if name in row:columns[name].append(row[name])
        for name,value in row['registration_detail_ms'].items():columns['registration_'+name+'_ms'].append(value)
    for row in solves:
        for name,value in row['timing'].items():columns[name+'_ms'].append(value*1000)
        columns['solve_wall_ms'].append(row['wall_ms'])
    if (directory/'direction_poses.json').exists():
        for row in json.loads((directory/'direction_poses.json').read_text()):
            for name in ('read_ms','decode_ms','payload_hash_ms','registration_ms','producer_ms'):
                columns['direction_'+name].append(row[name])
            for name in ('prepare_ms','producer_service_ms','worker_queue_ms'):
                if name in row:columns['direction_'+name].append(row[name])
            for name,value in row['registration_detail_ms'].items():columns['direction_registration_'+name+'_ms'].append(value)
    if (directory/'COPY_IO_TIMINGS.json').exists():
        for row in json.loads((directory/'COPY_IO_TIMINGS.json').read_text()):columns['io_'+row['operation']+'_ms'].append(row['ms'])
    result=dict(total_wall_ms=summary['wall_seconds']*1000,source_duration_ms=summary['recording_seconds']*1000,frames=summary['frames'],solves=len(solves),publications=summary['published'],timings={k:stats(v) for k,v in columns.items()},notes=['All durations in ms. Stage medians must not be added across concurrent workers.', 'T_TOTAL, solve_wall, producer, registration_total and C4 inclusive timers overlap children.', 'Normals and recovery statistics include only frames where that operation ran.'])
    (directory/'STAGES_MS.json').write_text(json.dumps(result,indent=2)+'\n');print(directory.name,json.dumps({k:result[k] for k in ('total_wall_ms','frames','solves','publications')}))
    print({k:round(v['median_ms'],2) for k,v in result['timings'].items()})
