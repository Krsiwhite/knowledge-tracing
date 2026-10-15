"""One pure-file repair check for the four S010 error-message regressions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ktlab.provenance import canonical_hash, sha256_file, source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/checks/S010-R1-ktst-history-reader-message.json"
AUTHORITY = ROOT / "approvals/S010-R1-ktst-history-reader-message.json"


def main():
    manifest = json.loads(MANIFEST.read_text())
    authority = json.loads(AUTHORITY.read_text())
    if authority.get("status") != "approved" or not authority.get("user_confirmation") or authority.get("manifest_sha256") != canonical_hash(manifest):
        raise ValueError("Concrete S010-R1 authorization binding is required")

    def verify():
        if source_fingerprint(ROOT) != manifest["implementation_sha256"]:
            raise ValueError("Training implementation changed")
        for name, digest in manifest["files"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f"Bound file changed: {name}")
        prior = json.loads((ROOT / "configs/checks/S010-ktst-history-training-cpu.json").read_text())
        for name, digest in prior["files"].items():
            if name != "scripts/analyze_shared_predictions.py" and sha256_file(ROOT / name) != digest:
                raise ValueError(f"Previously checked file changed: {name}")
        body = (ROOT / "scripts/analyze_shared_predictions.py").read_bytes()
        old, new = (manifest["message_only_change"][key].encode() for key in ("old", "new"))
        if body.count(new) != 1:
            raise ValueError("Expected exactly one repaired error message")
        import hashlib
        if hashlib.sha256(body.replace(new, old)).hexdigest() != prior["files"]["scripts/analyze_shared_predictions.py"]:
            raise ValueError("Reader repair is not exactly the declared message-only change")

    verify()
    runtime = ROOT / ".runtime"
    started = datetime.now(timezone.utc).isoformat()
    identity = {"check_id": manifest["check_id"], "manifest_sha256": canonical_hash(manifest),
                "manifest_file_sha256": sha256_file(MANIFEST), "authority_file_sha256": sha256_file(AUTHORITY),
                "implementation_sha256": manifest["implementation_sha256"]}
    with (runtime / "S010-R1-attempt.json").open("x") as handle:
        json.dump({**identity, "status": "started", "started_at": started}, handle, indent=2)
    start = time.monotonic()
    timed_out = False
    names = ("S010-R1.stdout.log", "S010-R1.stderr.log")
    with (runtime / names[0]).open("x") as out, (runtime / names[1]).open("x") as err:
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", *manifest["pytest_arguments"]],
                cwd=ROOT, env=dict(os.environ, **manifest["environment"]), stdout=out, stderr=err,
                timeout=manifest["limits"]["subprocess_wall_seconds"], check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, code = True, 124
    elapsed = time.monotonic() - start
    verify()
    if sha256_file(MANIFEST) != identity["manifest_file_sha256"] or sha256_file(AUTHORITY) != identity["authority_file_sha256"]:
        raise ValueError("S010-R1 freeze changed during execution")
    output = (runtime / names[0]).read_text()
    passed = code == 0 and not timed_out and "4 passed" in output and "skipped" not in output
    record = {**identity, "status": "passed" if passed else "failed", "exit_code": code,
              "timed_out": timed_out, "started_at": started,
              "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
              "source_and_bound_files_unchanged": True, "message_only_change_verified": True,
              "tests_passed": 4 if passed else None,
              "models": 0, "forwards": 0, "gradients": 0, "optimizer_steps": 0,
              "count_basis": "Bound pure-file tests and runner statically contain no model or gradient calls",
              "logs": {name: sha256_file(runtime / name) for name in names}}
    with (runtime / "S010-R1-result.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record))
    return 0 if passed else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
