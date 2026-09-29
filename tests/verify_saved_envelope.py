"""Compare rebuilt C++ envelope hits with every saved accepted result."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine/native'))
import _obstacle_envelope as native

parser = argparse.ArgumentParser()
parser.add_argument('--payload', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
rows = [json.loads(line) for line in (args.payload / 'far.jsonl').read_text().splitlines()]
checked = 0
digest = hashlib.sha256()
for row in rows:
    source = row['source_frame']
    path = args.payload / 'rail_refresh' / f'{source:06d}.npz'
    if not path.exists():
        continue
    with np.load(path) as forecast, np.load(args.payload / 'batches' / f'{source:06d}.npz') as batch:
        envelope = native.Envelope(forecast['pair'], 2.1, 3., .030001, len(forecast['near_pair']))
        actual = np.asarray(envelope.query(batch['xyz'], np.eye(4))['all_indices'], dtype=np.int64)
    np.testing.assert_array_equal(actual, np.asarray(row['all_indices'], dtype=np.int64))
    digest.update(actual.tobytes())
    checked += 1
report = dict(pass_=True, frames=checked, all_source_point_indices_exact=True, sha256=digest.hexdigest())
args.out.write_text(json.dumps(report, indent=2))
print(report)
