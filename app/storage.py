"""Small, atomic JSON records and append-only worker log readers."""

import json, os
from pathlib import Path

DATA = Path(os.environ.get("RAIL_DATA", "/data"))
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_log(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The producer may still be writing the final line.
            continue
    return result


def job_path(job_id):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Некорректный номер прогона")
    path = JOBS / job_id
    if not path.is_dir():
        raise ValueError("Прогон не найден")
    return path
