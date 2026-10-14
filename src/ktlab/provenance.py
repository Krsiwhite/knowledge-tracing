"""Content-addressed run provenance and explicit human approval gates."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ApprovalError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_fingerprint(root: Path | None = None) -> str:
    """Hash actual implementation bytes, even before the first Git commit."""
    root = root or project_root()
    paths = []
    for folder in (root / "src", root / "third_party" / "pykt" / "pykt"):
        paths.extend(folder.rglob("*.py"))
    paths.append(root / "pyproject.toml")
    inventory = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths) if p.is_file()}
    return canonical_hash(inventory)


def git_revision(root: Path | None = None) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root or project_root(), capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def config_fingerprint(config: dict) -> str:
    # Placement of the approval record and runtime authorization are not model choices.
    return canonical_hash({k: v for k, v in config.items() if k not in {"_batch_approved", "test_release_file"}})


def verify_manifest_files(manifest_file: Path) -> None:
    manifest = read_json(manifest_file)
    base = Path(manifest["data_dir"])
    base = (base if base.is_absolute() else manifest_file.parent / base).resolve()
    if not manifest.get("files"):
        raise ApprovalError("Dataset manifest has no file integrity records.")
    for name, expected in manifest["files"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base):
            raise ApprovalError("Dataset manifest contains a path outside its data directory.")
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ApprovalError(f"Prepared data file changed: {name}")


def validate_batch_approval(batch_file: Path, approval_file: Path) -> dict:
    batch = read_json(batch_file)
    if not approval_file.is_file():
        raise ApprovalError("This training batch is pending explicit user approval; no optimizer steps were started.")
    approval = read_json(approval_file)
    if approval.get("status") != "approved" or not approval.get("user_confirmation") or not approval.get("approved_at"):
        raise ApprovalError("Approval must record the user's explicit confirmation and its time.")
    if approval.get("batch_sha256") != canonical_hash(batch):
        raise ApprovalError("Batch configuration changed since approval. Re-align this exact batch with the user.")
    if batch.get("source_sha256") != source_fingerprint():
        raise ApprovalError("Implementation changed since the batch was prepared. Rebuild and re-align the batch.")
    checked = set()
    for entry in batch.get("runs", []):
        manifest = Path(entry["config"]["dataset_manifest"])
        if sha256_file(manifest) != entry.get("dataset_manifest_sha256"):
            raise ApprovalError(f"Prepared data manifest changed: {manifest}")
        if manifest.resolve() not in checked:
            verify_manifest_files(manifest)
            checked.add(manifest.resolve())
    if not batch.get("runs"):
        raise ApprovalError("Empty batch cannot be approved.")
    return batch


def validate_test_release(run_dir: Path, config: dict, release_file: Path) -> None:
    """Called before constructing a test dataset or reading its labels."""
    if not release_file.is_file():
        raise ApprovalError("Final test is sealed: no approved test release record exists.")
    release = read_json(release_file)
    if release.get("status") != "approved" or not release.get("user_confirmation") or not release.get("analysis_plan_sha256"):
        raise ApprovalError("Final test release needs explicit user approval and a frozen analysis plan.")
    analysis_file = Path(release.get("analysis_plan_file", ""))
    if not analysis_file.is_file() or sha256_file(analysis_file) != release["analysis_plan_sha256"]:
        raise ApprovalError("Analysis plan is missing or differs from the frozen test release.")
    record = release.get("runs", {}).get(run_dir.name)
    if not record:
        raise ApprovalError(f"Run {run_dir.name!r} is not included in the frozen test release.")
    if record.get("config_sha256") != config_fingerprint(config):
        raise ApprovalError("Run configuration differs from the frozen test release.")
    if record.get("dataset_manifest_sha256") != sha256_file(config["dataset_manifest"]):
        raise ApprovalError("Dataset manifest differs from the frozen test release.")
    if record.get("checkpoint_sha256") != sha256_file(run_dir / "best.pt"):
        raise ApprovalError("Checkpoint differs from the frozen test release.")
    if release.get("source_sha256") != source_fingerprint():
        raise ApprovalError("Evaluation implementation differs from the frozen test release.")
