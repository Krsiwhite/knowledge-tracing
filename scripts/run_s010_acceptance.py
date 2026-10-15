"""Run the separately frozen S010 synthetic CPU check exactly once.

This entry point reads no research data or trained weights.  The check manifest
binds the source, test file, this runner, limits and the standing authorization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ktlab.provenance import canonical_hash, sha256_file, source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/checks/S010-ktst-history-training-cpu.json"
AUTHORITY = ROOT / "approvals/S010-ktst-history-training-cpu.json"


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if (authority.get("status") != "approved" or not authority.get("user_confirmation")
            or authority.get("manifest_sha256") != canonical_hash(manifest)):
        raise ValueError("S010 needs the concrete manifest bound to the standing authority")

    def verify():
        if source_fingerprint(ROOT) != manifest["implementation_sha256"]:
            raise ValueError("S010 implementation differs from the frozen manifest")
        for name, expected in manifest["files"].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f"S010 bound file changed: {name}")

    verify()
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    identity = {"check_id": manifest["check_id"], "manifest_sha256": canonical_hash(manifest),
                "manifest_file_sha256": sha256_file(MANIFEST),
                "authority_file_sha256": sha256_file(AUTHORITY),
                "implementation_sha256": manifest["implementation_sha256"]}
    with (runtime / "S010-attempt.json").open("x", encoding="utf-8") as stream:
        json.dump({**identity, "status": "started", "started_at": started_at}, stream, indent=2)
        stream.write("\n")
    start = time.monotonic()
    timed_out, error, code = False, None, 1
    log_names = ("S010.stdout.log", "S010.stderr.log")
    with (runtime / log_names[0]).open("x", encoding="utf-8") as out, \
            (runtime / log_names[1]).open("x", encoding="utf-8") as err:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *manifest["pytest_arguments"]], cwd=ROOT,
                env=dict(os.environ, **manifest["environment"]), stdout=out, stderr=err,
                timeout=manifest["limits"]["subprocess_wall_seconds"], check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, code = True, 124
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
    elapsed = time.monotonic() - start
    unchanged = False
    try:
        verify()
        if sha256_file(MANIFEST) != identity["manifest_file_sha256"] or sha256_file(AUTHORITY) != identity["authority_file_sha256"]:
            raise ValueError("S010 manifest or authority changed during execution")
        unchanged = True
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    stdout = (runtime / log_names[0]).read_text(encoding="utf-8")
    counts_matches = re.findall(r"S010_COUNTS=(\{[^\n]*\})", stdout)
    counts = json.loads(counts_matches[0]) if len(counts_matches) == 1 else None
    expected = manifest["expected_counts"]
    passed_matches = re.findall(r"\b(\d+) passed\b", stdout)
    passed = int(passed_matches[-1]) if passed_matches else 0
    skipped = bool(re.search(r"\b\d+ skipped\b", stdout))
    accepted = code == 0 and not timed_out and error is None and unchanged and counts == expected and passed > 0 and not skipped
    record = {**identity, "status": "passed" if accepted else "failed", "exit_code": code,
              "timed_out": timed_out, "error": error, "started_at": started_at,
              "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
              "source_and_bound_files_unchanged": unchanged, "actual_counts": counts,
              "expected_counts_match": counts == expected, "tests_passed": passed, "tests_skipped": skipped,
              "logs": {name: sha256_file(runtime / name) for name in log_names}}
    with (runtime / "S010-result.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False))
    return 0 if accepted else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
