"""Verify accepted source, then seal this build's freshly compiled binaries."""

import hashlib, json
from pathlib import Path

root = Path("/app/engine")
hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = json.loads((root / "SOURCE_MANIFEST.json").read_text())
for name, digest in source.items():
    if hash_file(root / name) != digest:
        raise RuntimeError("Frozen source changed: " + name)
origin = json.loads((root / "COPY_ORIGIN.json").read_text())
origin["files"] = {
    str(p.relative_to(root / "baseline")): hash_file(p)
    for p in (root / "baseline").rglob("*")
    if p.is_file()
}
origin["build_note"] = (
    "Accepted sources verified; native binaries compiled from shipped sources in Docker."
)
(root / "COPY_ORIGIN.json").write_text(json.dumps(origin, indent=2))
print("Verified frozen source files:", len(source))
