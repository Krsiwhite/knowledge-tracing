import json
from pathlib import Path

import pytest

from ktlab import provenance as p


def test_canonical_hash_is_order_independent_but_value_sensitive():
    assert p.canonical_hash({"a": 1, "b": 2}) == p.canonical_hash({"b": 2, "a": 1})
    assert p.canonical_hash({"a": 1}) != p.canonical_hash({"a": 2})


def test_source_fingerprint_covers_ktst_upstream_bytes(tmp_path):
    vendor = tmp_path / "third_party/ktst/pkg/model"
    vendor.mkdir(parents=True)
    source = vendor / "fixture.py"
    source.write_text("fixture = 1\n")
    before = p.source_fingerprint(tmp_path)
    source.write_text("fixture = 2\n")
    assert p.source_fingerprint(tmp_path) != before


def test_batch_is_sealed_until_specific_user_confirmation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    data = tmp_path / "fixture.txt"
    data.write_text("fixture data", encoding="utf-8")
    p.write_json(manifest, {"dataset": "fixture", "data_dir": str(tmp_path), "files": {data.name: {"sha256": p.sha256_file(data), "bytes": data.stat().st_size}}})
    monkeypatch.setattr(p, "source_fingerprint", lambda: "source")
    batch = {"source_sha256": "source", "runs": [{"config": {"dataset_manifest": str(manifest)}, "dataset_manifest_sha256": p.sha256_file(manifest)}]}
    batch_file, approval_file = tmp_path / "batch.json", tmp_path / "approval.json"
    p.write_json(batch_file, batch)
    with pytest.raises(p.ApprovalError, match="pending"):
        p.validate_batch_approval(batch_file, approval_file)
    p.write_json(approval_file, {"status": "approved", "user_confirmation": "Approved fixture", "approved_at": "fixture", "batch_sha256": p.canonical_hash(batch)})
    assert p.validate_batch_approval(batch_file, approval_file) == batch
    data.write_text("changed data", encoding="utf-8")
    with pytest.raises(p.ApprovalError, match="data file changed"):
        p.validate_batch_approval(batch_file, approval_file)
    data.write_text("fixture data", encoding="utf-8")
    p.write_json(manifest, {"dataset": "changed"})
    with pytest.raises(p.ApprovalError, match="manifest changed"):
        p.validate_batch_approval(batch_file, approval_file)


def test_test_release_requires_frozen_checkpoint_and_config(tmp_path, monkeypatch):
    run = tmp_path / "fixture-run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"fixture only, no training")
    manifest = tmp_path / "data.json"
    p.write_json(manifest, {"split": "fixed"})
    config = {"dataset_manifest": str(manifest), "model": "fixture"}
    analysis = tmp_path / "analysis.json"
    p.write_json(analysis, {"primary_metric": "auc"})
    monkeypatch.setattr(p, "source_fingerprint", lambda: "source")
    release_file = tmp_path / "release.json"
    with pytest.raises(p.ApprovalError, match="sealed"):
        p.validate_test_release(run, config, release_file)
    p.write_json(release_file, {"status": "approved", "user_confirmation": "Fixture approval", "analysis_plan_file": str(analysis), "analysis_plan_sha256": p.sha256_file(analysis), "source_sha256": "source", "runs": {run.name: {"config_sha256": p.config_fingerprint(config), "dataset_manifest_sha256": p.sha256_file(manifest), "checkpoint_sha256": p.sha256_file(run / "best.pt")}}})
    p.validate_test_release(run, config, release_file)
    (run / "best.pt").write_bytes(b"replaced")
    with pytest.raises(p.ApprovalError, match="Checkpoint"):
        p.validate_test_release(run, config, release_file)
