"""Register an already completed result directory for read-only review.

Copy the payload to /data/jobs/<32-hex-id>/result before invoking this script.
Object grouping uses the same automatic spatial rule as a newly computed run.
"""

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.storage import job_path, write_json, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()
path = job_path(args.job)
complete = read_json(path / "result/COMPLETE.json")
if not complete or not complete.get("pass_"):
    raise SystemExit("A completed payload is required")
write_json(
    path / "job.json",
    dict(
        id=args.job,
        name=args.name,
        status="COMPLETE",
        created=time.time(),
        topics=[],
        topic=dict(count=complete["received"]),
        imported=True,
        temporary_input_removed=True,
    ),
)
print(args.job)
