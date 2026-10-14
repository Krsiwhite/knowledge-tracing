"""Verify the vendored pyKT source against its committed content inventory."""
import hashlib
import json
import argparse
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "third_party/pykt.lock.json").read_text(encoding="utf-8"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, help="Also prove inventory against the original pinned GitHub archive")
args = parser.parse_args()
failures = []
for name, digest in lock["files_sha256"].items():
    path = root / "third_party/pykt" / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
unexpected = [p.relative_to(root / "third_party/pykt").as_posix() for p in (root / "third_party/pykt").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.relative_to(root / "third_party/pykt").as_posix() not in lock["files_sha256"]]
result = {"commit": lock["commit"], "checked": len(lock["files_sha256"]), "mismatched": failures, "unexpected": unexpected}
if args.archive:
    archive_digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if archive_digest != lock["archive_sha256"]:
        failures.append("archive_sha256")
    with tarfile.open(args.archive, "r:gz") as archive:
        prefix = "pykt-toolkit-" + lock["commit"] + "/"
        for name, expected in lock["files_sha256"].items():
            with archive.extractfile(prefix + name) as stream:
                if hashlib.sha256(stream.read()).hexdigest() != expected:
                    failures.append("archive:" + name)
    result["archive_verified"] = not failures
print(json.dumps(result, indent=2))
raise SystemExit(1 if failures or unexpected else 0)
