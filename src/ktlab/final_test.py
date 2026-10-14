"""Separately released, one-attempt evaluation of frozen shared-dev candidates.

No data-file reader or model builder runs during preflight. Receipts establish
consistency of unsigned local records, not independent proof of training or of
method equivalence. A compatibility review supplies the latter human judgment.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .provenance import (ApprovalError, canonical_hash, config_fingerprint, project_root,
                         read_json, sha256_file, source_fingerprint, utc_now,
                         write_json)
from .shared_events import FINAL_PROTOCOL, PROTOCOL, SharedTestEventDataset

_ISSUER = object()
_IDENTITY_KEYS = ("config_sha256", "implementation_sha256", "model_source_sha256",
                  "validation_source_sha256", "validation_target_sha256", "validation_context_sha256",
                  "checkpoint_selection_policy")
_METRICS = ["auc", "nll", "brier", "accuracy", "ece_15bins"]


def _require(condition, message):
    if not condition:
        raise ApprovalError(message)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value, low=0):
    return type(value) is int and value >= low


def _current_model_source(name):
    root = project_root() / "third_party"
    path = root / "ktst.lock.json" if name in {"ktst_mean", "ktst_mhsa"} else root / "pykt/pykt/models" / f"{name}.py"
    return sha256_file(path)


def _load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _build_model(config, data_config):
    from .adapters import build_model, resolve_device
    device = resolve_device(config.get("device", "auto"))
    if config["model"] in {"ktst_mean", "ktst_mhsa"}:
        from .ktst_model import build_ktst_model as build_model
    return build_model(config["model"], data_config, config["model_config"], device), device


def _preflight(run_dir, release_file):
    """Only frozen JSON and checkpoint bytes; NEVER verify_manifest_files here."""
    receipts = {}
    data_base = None

    def capture(path, expected=None):
        path = Path(path).resolve()
        _require(path.suffix == ".json" and (data_base is None or not path.is_relative_to(data_base)),
                 "Control receipts cannot point to prepared data files")
        before = sha256_file(path)
        _require(expected is None or before == expected, f"Frozen receipt changed: {path.name}")
        value = read_json(path)
        _require(sha256_file(path) == before, f"Receipt changed while reading: {path.name}")
        receipts[path] = before
        return value

    if not release_file.is_file():
        raise PermissionError("Shared final test is sealed: an approved release is required")
    # Discover the prepared-data boundary from the historical config/manifest
    # before opening ANY caller-supplied release/plan/review JSON.
    config = capture(run_dir / "resolved_config.json")
    manifest_path = Path(config["dataset_manifest"]).resolve()
    _require(manifest_path.name == "manifest.json", "A canonical prepared manifest is required")
    manifest = capture(manifest_path)
    base = Path(manifest["data_dir"])
    data_base = base = (base if base.is_absolute() else manifest_path.parent / base).resolve()
    manifest["data_dir"] = str(base)
    release = capture(release_file)
    _require(release.get("schema_version") == 1 and release.get("protocol") == FINAL_PROTOCOL,
             "A separate shared final-test release protocol is required")
    _require(release.get("status") == "approved" and release.get("user_confirmation") and release.get("approved_at"),
             "Release must record approval, its authorization basis, and time")
    record = release.get("runs", {}).get(run_dir.name)
    _require(isinstance(record, dict), "Run is absent from the frozen candidate set")
    metadata = capture(run_dir / "metadata.json", record.get("metadata_sha256"))
    _require(_hash(record.get("metadata_sha256")), "Original metadata receipt is required")
    _require(receipts[manifest_path] == record.get("dataset_manifest_sha256"), "Dataset manifest differs from release")
    checkpoint_path = run_dir / "best.pt"
    _require(checkpoint_path.resolve().parent == run_dir and not checkpoint_path.resolve().is_relative_to(base),
             "Checkpoint must stay in the original run directory, outside prepared data")
    evaluation_source = source_fingerprint()
    _require(release.get("source_sha256") == evaluation_source, "Evaluation source changed")
    plan = capture(release["analysis_plan_file"], release["analysis_plan_sha256"])
    review = capture(release["compatibility_review_file"], release["compatibility_review_sha256"])
    for key in ("analysis_plan_sha256", "compatibility_review_sha256"):
        _require(_hash(release.get(key)), f"Missing frozen {key}")
    # Use the captured release, not a second parser that could follow a changed
    # analysis_plan_file into prepared data before the attempt is registered.
    _require(plan.get("schema_version") == 1 and plan.get("protocol") == FINAL_PROTOCOL,
             "Analysis plan must declare the final-test protocol")
    run_ids = plan.get("run_ids")
    _require(isinstance(run_ids, list) and all(isinstance(r, str) for r in run_ids)
             and len(set(run_ids)) == len(run_ids) and set(run_ids) == set(release["runs"]),
             "Analysis plan must freeze the exact candidate set")
    _require(plan.get("metrics") == _METRICS and plan.get("accuracy_threshold") == .5
             and plan.get("ece_bins") == 15 and plan.get("kc_budget") == 200
             and plan.get("test_selection") == "all_eligible_complete_event_kc200"
             and plan.get("max_targets") is None,
             "Analysis plan must use the complete fixed selection and metric definitions")
    for key in ("batch_size", "max_rows_per_forward"):
        _require(_integer(plan.get(key), 1), f"Invalid frozen {key}")
    _require(review.get("schema_version") == 1 and review.get("status") == "approved"
             and review.get("reviewed_by") and review.get("statement")
             and review.get("training_protocol") == PROTOCOL
             and review.get("evaluation_protocol") == FINAL_PROTOCOL
             and review.get("evaluation_source_sha256") == evaluation_source
             and review.get("candidates") == release["runs"],
             "Compatibility review must explicitly bind all original candidates to this evaluator")
    _require(config.get("protocol") == PROTOCOL and record.get("training_protocol") == PROTOCOL,
             "Only original shared-development candidates are eligible; no relabeling")
    _require(config.get("model") in {"akt", "simplekt", "mockt", "ukt", "ktst_mean", "ktst_mhsa"},
             "Unsupported shared model")
    _require(_integer(config.get("fold")) and config["fold"] in range(5)
             and _integer(config.get("seed")), "Training fold/seed must remain explicit integers")
    _require(metadata.get("run_id") == run_dir.name and metadata.get("status") == "completed"
             and metadata.get("test_accessed") is False and metadata.get("partial_debug") is False,
             "Only completed, untested, full-validation runs may be released")
    _require(not config.get("debug_smoke") and config.get("max_validation_targets") is None
             and config.get("max_validation_batches") is None and config.get("initial_validation") is True
             and metadata.get("initial_validation") is True and metadata.get("seed") == config["seed"],
             "Candidate is debug, incomplete, or inconsistent with epoch-zero selection")
    _require(isinstance(config.get("model_config"), dict), "Resolved model configuration is required")
    identity = {key: record.get(key) for key in _IDENTITY_KEYS}
    from .engine import SHARED_SELECTION, _validate_shared_checkpoint
    _require(identity["checkpoint_selection_policy"] == SHARED_SELECTION,
             "Original shared checkpoint-selection policy is required")
    for key in _IDENTITY_KEYS[:-1]:
        _require(_hash(identity[key]), f"Missing original {key}")
    _require(identity["config_sha256"] == config_fingerprint(config), "Original configuration changed")
    _require(identity["model_source_sha256"] == _current_model_source(config["model"]),
             "Original model source changed; this entry does not migrate model mathematics")
    _require(all(metadata.get(k) == v for k, v in identity.items()), "Original training identity mismatch")
    audit = metadata.get("valid_event_audit", {})
    _require(audit.get("protocol") == PROTOCOL and audit.get("split") == "valid"
             and audit.get("fold") == config["fold"] and audit.get("dataset") == config.get("dataset")
             and audit.get("kc_budget") == 200 and audit.get("max_targets") is None
             and audit.get("selection_limited") is False and audit.get("selected_for_debug") is False
             and _integer(audit.get("selected_targets"), 1)
             and audit["selected_targets"] == audit.get("eligible_total") == audit.get("expected_total"),
             "Original validation must cover the complete shared selection")
    for name in ("source", "target", "context"):
        _require(audit.get(f"{name}_sha256") == identity[f"validation_{name}_sha256"],
                 f"Original validation {name} identity mismatch")
    epoch, auc = record.get("best_epoch"), record.get("best_valid_auc")
    _require(_integer(epoch) and type(auc) in (int, float) and math.isfinite(auc) and 0 <= auc <= 1
             and type(record.get("best_epoch_completed")) is bool,
             "Invalid recorded best validation candidate")
    for key in ("best_epoch", "best_valid_auc", "best_epoch_completed", "checkpoint_sha256", "dataset_manifest_sha256"):
        _require(metadata.get(key) == record.get(key), f"Frozen candidate {key} mismatch")
    _require(manifest.get("dataset") == config.get("dataset")
             and manifest.get("split_audit", {}).get("train_test_disjoint") is True,
             "Prepared dataset/split audit does not match the candidate")
    data_config = manifest["data_config"]
    _require(data_config.get("test_original_file") == "test.csv" and data_config.get("maxlen") == 200
             and _integer(data_config.get("num_c"), 1) and _integer(data_config.get("num_q"))
             and isinstance(data_config.get("folds"), list)
             and all(_integer(f) and f in range(5) for f in data_config["folds"])
             and config["fold"] in data_config["folds"], "Invalid canonical held-out data configuration")
    data_receipts = {}
    for name in ("test.csv", "keyid2idx.json"):
        receipt = manifest.get("files", {}).get(name, {})
        path = (base / name).resolve()
        _require(path.parent == base and _hash(receipt.get("sha256")) and _integer(receipt.get("bytes"), 1),
                 f"Missing canonical data/mapping receipt: {name}")
        data_receipts[path] = receipt
    receipts[checkpoint_path] = record["checkpoint_sha256"]
    _require(sha256_file(checkpoint_path) == record["checkpoint_sha256"], "Checkpoint differs from release")
    checkpoint = _load_checkpoint(checkpoint_path)
    _validate_shared_checkpoint(checkpoint, metadata, identity, record["dataset_manifest_sha256"], config["model_config"])
    _require("state_dict" in checkpoint, "Checkpoint has no parameter state")
    test_identity = {"run_id": run_dir.name, "checkpoint_sha256": record["checkpoint_sha256"],
                     "dataset_manifest_sha256": record["dataset_manifest_sha256"],
                     "test_source_sha256": manifest["files"]["test.csv"]["sha256"]}
    # Tied to canonical manifest and candidate, not a caller-selected output or release ID.
    registry = base / ".final-test-attempts"
    attempt_path = registry / (canonical_hash(test_identity) + ".json")
    dataset_identity = {"dataset": manifest["dataset"], "test_source_sha256": test_identity["test_source_sha256"],
                        "mapping_sha256": manifest["files"]["keyid2idx.json"]["sha256"]}
    release_lock = registry / ("dataset-" + canonical_hash(dataset_identity) + ".json")
    frozen_release = {"schema_version": 1, "dataset_identity": dataset_identity,
                      "release_sha256": receipts[release_file], "analysis_plan_sha256": release["analysis_plan_sha256"],
                      "compatibility_review_sha256": release["compatibility_review_sha256"],
                      "evaluation_source_sha256": evaluation_source, "candidates_sha256": canonical_hash(release["runs"])}
    if release_lock.exists():
        _require(read_json(release_lock) == frozen_release,
                 "This test source already has a different frozen release/candidate set/analysis plan")
    return dict(run_dir=run_dir, release=release, record=record, config=config, metadata=metadata,
                manifest=manifest, checkpoint=checkpoint, plan=plan, receipts=receipts,
                data_receipts=data_receipts, evaluation_source=evaluation_source,
                attempt_path=attempt_path, test_identity=test_identity, output=run_dir / "final_test",
                release_path=release_file, release_lock=release_lock, frozen_release=frozen_release)


def _exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class FinalTestAccess:
    """Process-local capability issued only after the durable exclusive attempt."""

    def __init__(self, bundle, start_sha256, *, issuer):
        _require(issuer is _ISSUER, "Test access can only be issued by the released evaluator")
        self.bundle, self.start_sha256 = bundle, start_sha256
        self.manifest_sha256 = canonical_hash(bundle["manifest"])

    def verify_receipts(self):
        b = self.bundle
        _require(source_fingerprint() == b["evaluation_source"], "Evaluation source changed during final test")
        for path, expected in b["receipts"].items():
            _require(sha256_file(path) == expected, f"Frozen receipt changed: {path.name}")
        _require(sha256_file(b["attempt_path"]) == self.start_sha256,
                 "Final attempt is no longer the registered active attempt")
        _require(read_json(b["release_lock"]) == b["frozen_release"], "Dataset release lock changed")

    def check_data_access(self, manifest):
        self.verify_receipts()
        _require(canonical_hash(manifest) == self.manifest_sha256, "Released manifest changed before data access")

    def check_dataset(self, dataset):
        self.check_data_access(self.bundle["manifest"])
        audit = dataset.audit
        _require(isinstance(dataset, SharedTestEventDataset) and dataset.path ==
                 Path(self.bundle["manifest"]["data_dir"]) / "test.csv"
                 and audit.get("protocol") == FINAL_PROTOCOL and audit.get("split") == "test"
                 and audit.get("fold") == -1 and audit.get("test_access") is True
                 and audit.get("max_targets") is None and not audit.get("selection_limited")
                 and len(dataset) == audit.get("expected_total") == audit.get("eligible_total")
                 and len(dataset) > 0, "Final evaluation requires the entire canonical held-out selection")
        self.verify_data_receipts()

    def verify_data_receipts(self):
        for path, expected in self.bundle["data_receipts"].items():
            _require(path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"],
                     f"Released data/mapping changed: {path.name}")

    def output_identity(self):
        b = self.bundle
        return {"run_id": b["run_dir"].name, "training_identity": b["record"],
                "training_fold": b["config"]["fold"], "training_seed": b["config"]["seed"],
                "evaluation_protocol": FINAL_PROTOCOL, "evaluation_implementation_sha256": b["evaluation_source"],
                "test_identity": b["test_identity"], "test_release_sha256": b["receipts"][b["release_path"]],
                "analysis_plan_sha256": b["release"]["analysis_plan_sha256"],
                "compatibility_review_sha256": b["release"]["compatibility_review_sha256"],
                "attempt_file": str(b["attempt_path"]), "attempt_start_sha256": self.start_sha256,
                "dataset_release_lock_file": str(b["release_lock"]),
                "dataset_release_lock_sha256": sha256_file(b["release_lock"]),
                "original_candidate_receipts_verified": True,
                "compatibility_review_required_not_independently_proven": True}


def evaluate_shared_test(run_dir, release_file, *, allow_test=False):
    """Evaluate a frozen shared-dev checkpoint once, into run_dir/final_test.

    A failed/crashed attempt remains consumed. No implicit recovery/retry API is
    provided. Passing a different release filename does not reset the attempt.
    """
    if allow_test is not True:
        raise PermissionError("Test is sealed; explicit frozen-candidate release is required")
    b = _preflight(Path(run_dir).resolve(), Path(release_file).resolve())
    _require(not b["output"].exists(), "Final output already exists; no automatic retry")
    # Recheck non-data receipts before the irreversible first-access registration.
    for path, expected in b["receipts"].items():
        _require(sha256_file(path) == expected, f"Frozen receipt changed: {path.name}")
    _require(source_fingerprint() == b["evaluation_source"], "Evaluation source changed before attempt")
    start = {"schema_version": 1, "status": "started", "created_at": utc_now(),
             "test_identity": b["test_identity"], "release_sha256": b["receipts"][b["release_path"]],
             "evaluation_source_sha256": b["evaluation_source"], "output_directory": str(b["output"]),
             "test_access_attempted": True, "automatic_retry_allowed": False}
    b["attempt_path"].parent.mkdir(parents=True, exist_ok=True)
    try:
        _exclusive_json(b["release_lock"], b["frozen_release"])
    except FileExistsError:
        _require(read_json(b["release_lock"]) == b["frozen_release"],
                 "This test source already has a different frozen release/candidate set/analysis plan")
    b["receipts"][b["release_lock"]] = sha256_file(b["release_lock"])
    _exclusive_json(b["attempt_path"], start)
    access = FinalTestAccess(b, sha256_file(b["attempt_path"]), issuer=_ISSUER)
    dataset = None
    try:
        access.check_data_access(b["manifest"])
        access.verify_data_receipts()  # First possible test-file read is AFTER fsync.
        dataset = SharedTestEventDataset(b["manifest"], access=access)
        access.check_dataset(dataset)
        model, device = _build_model(b["config"], b["manifest"]["data_config"])
        _require(model.model_name == b["config"]["model"]
                 and model.ktlab_source_sha256 == b["record"]["model_source_sha256"]
                 and model.ktlab_model_config == b["config"]["model_config"],
                 "Constructed model differs from the original frozen candidate")
        model.load_state_dict(b["checkpoint"]["state_dict"])
        from .shared_evaluation import evaluate_shared
        path = b["output"] / "predictions.csv"
        result = evaluate_shared(model, dataset, device, path, batch_size=b["plan"]["batch_size"],
                                 max_rows_per_forward=b["plan"]["max_rows_per_forward"], final_access=access)
        access.verify_receipts()
        access.verify_data_receipts()
        _require(result["implementation_sha256"] == b["evaluation_source"], "Export source differs from release")
        result.update(prediction_file=str(path), prediction_sha256=sha256_file(path),
                      prediction_manifest_sha256=sha256_file(path.with_suffix(".csv.manifest.json")),
                      auc_available=result["auc"] is not None,
                      auc_unavailable_reason="single_response_class" if result["auc"] is None else None)
        write_json(b["output"] / "result.json", result)
        access.verify_receipts()
        write_json(b["attempt_path"], {**start, "status": "completed", "finished_at": utc_now(),
                   "attempt_start_sha256": access.start_sha256,
                   "result_sha256": sha256_file(b["output"] / "result.json"),
                   "prediction_sha256": result["prediction_sha256"],
                   "prediction_manifest_sha256": result["prediction_manifest_sha256"]})
        return result
    except BaseException as error:
        write_json(b["attempt_path"], {**start, "status": "failed", "finished_at": utc_now(),
                   "attempt_start_sha256": access.start_sha256,
                   "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if dataset is not None:
            dataset.close()
