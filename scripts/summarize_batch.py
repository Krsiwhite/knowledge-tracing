#!/usr/bin/env python3
"""Verify saved calibration artifacts and emit an allowlisted public report.

Only Python's standard library is imported. Checkpoints/predictions are hashed
as opaque bytes; models, optimizers, prediction records and prepared data files
are never loaded. The small dataset manifest is read for provenance receipts.
Approval text and absolute paths are deliberately excluded from the report.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


V1 = "ktlab-v1-causal-valid-full-precision-repeat-kc-train"
KTST = "ktlab-ktst-reference-dev-v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class ReportError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReportError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {Path(path).name}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ReportError(f"Non-finite JSON number in {Path(path).name}")
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=reject_constant)
    require(isinstance(value, dict), f"Expected JSON object in {Path(path).name}")
    return value


def checked_hash(value, name):
    require(isinstance(value, str) and bool(_SHA.fullmatch(value)), f"Invalid {name} SHA256")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer {name}")
    return value


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"Invalid finite numeric {name}")
    return value


def timestamp(value, name):
    require(isinstance(value, str), f"Missing {name}")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, f"Naive timestamp {name}")
    return parsed.astimezone(timezone.utc)


def host_path(value, root):
    value = str(value)
    if os.name == "nt" and re.match(r"^/mnt/[a-z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def source_fingerprint(root):
    paths = []
    for folder in (root / "src", root / "third_party/pykt/pykt", root / "third_party/ktst"):
        paths.extend(folder.rglob("*.py"))
    paths.append(root / "pyproject.toml")
    paths.extend((root / "third_party").glob("*.lock.json"))
    return canonical_hash({p.relative_to(root).as_posix(): file_hash(p)
                           for p in sorted(paths) if p.is_file()})


def literal_assignment(path, name, function=None):
    """Read constructor defaults without importing or executing model code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if function:
        nodes = next(node.body for node in nodes if isinstance(node, ast.FunctionDef) and node.name == function)
    for node in nodes:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ReportError(f"Cannot statically verify model default {name}")


def expected_model_config(root, config, data):
    model = config["model"]
    if model in {"akt", "simplekt", "mockt"}:
        result = literal_assignment(root / "src/ktlab/adapters.py", "settings", "build_model")
        if model == "mockt":
            result.update(emb_type="qid_conv_ker_noexp", kernel_size=4)
        result.update(config.get("model_config", {}))
        if model in {"simplekt", "mockt"}:
            result.setdefault("seq_len", int(data.get("maxlen", 200)))
    elif model in {"ktst_mean", "ktst_mhsa"}:
        result = literal_assignment(root / "src/ktlab/ktst_model.py", "_DEFAULT_MODEL")
        result.update(num_questions=max(1, int(data["num_q"])), num_concepts=int(data["num_c"]),
                      max_concepts=int(data["max_concepts"]), max_len=int(data.get("maxlen", 200)),
                      aggregation="q_mean_c" if model == "ktst_mean" else "self_attn_all")
        result.update(config.get("model_config", {}))
        require(result["num_questions"] == max(1, data["num_q"]) and result["num_concepts"] == data["num_c"]
                and result["max_concepts"] == data["max_concepts"] and result["max_len"] == data.get("maxlen", 200),
                "KTST data dimensions differ from approved manifest")
    else:
        raise ReportError("Unsupported model for this artifact-only calibration report")
    return result


def public_metrics(metrics):
    result = {key: metrics[key] for key in (
        "n", "auc", "accuracy", "brier", "nll", "ece_15bins", "concept_predictions",
        "aggregation", "prediction_unit", "evaluated_batches", "partial_debug", "dropped_partial_groups"
    ) if key in metrics}
    integer(result.get("n"), "validation.n", 1)
    for key in ("auc", "accuracy", "brier", "nll", "ece_15bins"):
        number(result.get(key), "validation." + key)
        if key != "nll":
            require(result[key] <= 1, f"Validation {key} outside [0,1]")
    integer(result.get("evaluated_batches"), "validation.evaluated_batches", 1)
    require(type(result.get("partial_debug")) is bool, "Missing partial validation flag")
    require(result.get("prediction_unit") == "question_event", "Unsupported prediction unit")
    require(result.get("aggregation") in {"late_mean_full_precision", "native_set_event"}, "Unsupported aggregation")
    return result


def summarize(repo_root, batch_path):
    root = Path(repo_root).resolve()
    batch_file = host_path(batch_path, root)
    batch = read_json(batch_file)
    batch_id = batch.get("batch_id")
    require(isinstance(batch_id, str) and bool(_SAFE_ID.fullmatch(batch_id)), "Unsafe/missing batch ID")
    require(batch.get("test_access") is False, "Only a test-sealed batch can be summarized")
    batch_sha = canonical_hash(batch)
    approval_path = root / "approvals" / (batch_id + ".json")
    approval = read_json(approval_path)
    require(approval.get("status") == "approved" and bool(approval.get("user_confirmation")), "Explicit approval evidence is missing")
    approved_at = timestamp(approval.get("approved_at"), "approval time")
    require(approval.get("batch_sha256") == batch_sha, "Canonical approved batch SHA mismatch")
    source_sha = checked_hash(batch.get("source_sha256"), "batch source")
    require(source_fingerprint(root) == source_sha, "Repository source differs from approved implementation")
    entries = batch.get("runs")
    require(isinstance(entries, list) and bool(entries), "Empty/missing run manifest")
    run_ids = [entry.get("run_id") for entry in entries]
    require(all(isinstance(value, str) and _SAFE_ID.fullmatch(value) for value in run_ids), "Unsafe run ID")
    require(len(set(run_ids)) == len(run_ids), "Run IDs must be unique in the batch manifest")
    for meta_path in (root / "runs").glob("*/metadata.json"):
        candidate = read_json(meta_path)
        if candidate.get("batch_id") == batch_id:
            require(meta_path.parent.name in run_ids, "An extra unmanifested run claims this batch ID")
    total_limit = integer(batch["limits"].get("total_optimizer_steps_upper_bound",
                                           batch["limits"].get("total_optimizer_steps")), "batch step limit", 1)
    runs, snapshots, manifest_cache = [], {}, {}
    declared_steps = 0
    for entry in entries:
        run_id, approved_config = entry["run_id"], entry["config"]
        run_dir = (root / "runs" / run_id).resolve()
        require(run_dir.parent == (root / "runs").resolve(), "Run directory escaped repository runs")
        metadata_path, config_path, metrics_path = (run_dir / name for name in
                                                   ("metadata.json", "resolved_config.json", "valid_metrics.json"))
        metadata, config, metrics = read_json(metadata_path), read_json(config_path), read_json(metrics_path)
        require(metadata.get("status") == "completed", f"Run is not completed: {run_id}")
        require(metadata.get("run_id") == run_id and metadata.get("batch_id") == batch_id, "Run identity mismatch")
        require(metadata.get("approved_batch_sha256") == batch_sha, "Run approval SHA mismatch")
        require(metadata.get("implementation_sha256") == source_sha, "Run implementation SHA mismatch")
        require(metadata.get("test_accessed") is False, "Run test access is not sealed")
        require(not any(run_dir.glob("test*")), "Run contains unexpected test artifacts")
        manifest_path = host_path(approved_config["dataset_manifest"], root)
        if manifest_path not in manifest_cache:
            manifest_cache[manifest_path] = (read_json(manifest_path), file_hash(manifest_path))
        manifest, manifest_sha = manifest_cache[manifest_path]
        require(entry.get("dataset_manifest_sha256") == manifest_sha == metadata.get("dataset_manifest_sha256"), "Dataset manifest SHA mismatch")
        require(manifest.get("dataset") == approved_config.get("dataset"), "Dataset identity mismatch")
        require(host_path(config.get("dataset_manifest", ""), root) == manifest_path, "Resolved manifest location mismatch")
        expected = dict(approved_config, batch_id=batch_id, approved_batch_sha256=batch_sha)
        expected["dataset_manifest"] = config["dataset_manifest"]
        expected["model_config"] = expected_model_config(root, approved_config, manifest["data_config"])
        require(config == expected, "Resolved config differs from approved config and static constructor defaults")
        config_sha = canonical_hash({k: v for k, v in config.items() if k not in {"_batch_approved", "test_release_file"}})
        require(config_sha == metadata.get("config_sha256"), "Resolved config SHA mismatch")
        model, dataset = config["model"], config["dataset"]
        protocol = KTST if model.startswith("ktst_") else V1
        require(metadata.get("protocol") == protocol, "Run protocol mismatch")
        require(metrics.get("protocol", protocol) == protocol, "Validation protocol mismatch")
        require(config.get("protocol", protocol) == protocol, "Config protocol mismatch")
        model_source = root / ("third_party/ktst.lock.json" if model.startswith("ktst_")
                               else f"third_party/pykt/pykt/models/{model}.py")
        require(file_hash(model_source) == metadata.get("model_source_sha256"), "Model source SHA mismatch")
        train_file = manifest["data_config"]["train_valid_file"]
        valid_file = (manifest["validation_question_files"][str(config["fold"])]
                      if not model.startswith("ktst_") and manifest["data_config"]["num_q"] > 0 else train_file)
        for key, name in (("train_file_sha256", train_file), ("valid_file_sha256", valid_file)):
            require(metadata.get(key) == manifest["files"][name]["sha256"], "Recorded development file receipt mismatch")
        steps = integer(metadata.get("optimizer_steps"), "optimizer_steps")
        step_limit = integer(config.get("max_train_steps"), "run step limit", 1)
        require(steps <= step_limit, "Run optimizer-step limit exceeded")
        declared_steps += step_limit
        epoch = integer(metadata.get("best_epoch"), "best epoch")
        require(epoch <= integer(config.get("epochs"), "epoch limit", 1), "Selected epoch exceeds approved limit")
        require(epoch > 0 or model.startswith("ktst_"), "Unexpected initial checkpoint selection")
        validation = public_metrics(metrics)
        require(metadata.get("partial_debug") is validation["partial_debug"], "Partial validation evidence mismatch")
        require(validation["partial_debug"] == (config.get("max_validation_batches") is not None), "Validation scope differs from approved config")
        best_auc = number(metadata.get("best_valid_auc"), "best validation AUC")
        require(abs(best_auc - validation["auc"]) <= 1e-12, "Saved validation AUC differs from selected checkpoint")
        history_path = run_dir / "history.jsonl"
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(history and history[-1].get("optimizer_steps") == steps, "Final optimizer count differs from epoch history")
        previous_epoch = previous_steps = 0
        for record in history:
            current_epoch = integer(record.get("epoch"), "history epoch", 1)
            current_steps = integer(record.get("optimizer_steps"), "history steps", 1)
            require(previous_epoch < current_epoch <= config["epochs"] and previous_steps < current_steps <= steps,
                    "Epoch/optimizer history is not strictly increasing within approved limits")
            previous_epoch, previous_steps = current_epoch, current_steps
        selected_metrics = (read_json(run_dir / "initial_valid_metrics.json") if epoch == 0 else
                            next((record["valid"] for record in history if record["epoch"] == epoch), {}))
        require(abs(number(selected_metrics.get("auc"), "selected history AUC") - best_auc) <= 1e-12,
                "Selected epoch does not match saved metric history")
        checkpoint_sha = file_hash(run_dir / "best.pt")
        require(checkpoint_sha == metadata.get("checkpoint_sha256"), "Saved checkpoint SHA mismatch")
        prediction_sha = file_hash(run_dir / "valid_predictions.csv")
        require(metadata.get("prediction_sha256", prediction_sha) == prediction_sha, "Saved prediction SHA mismatch")
        started = timestamp(metadata.get("started_at"), "run start")
        finished = timestamp(metadata.get("finished_at"), "run finish")
        ended = timestamp(metadata.get("ended_at", metadata["finished_at"]), "run end")
        require(approved_at <= started <= finished <= ended, "Invalid approval/run time ordering")
        elapsed = number(metadata.get("elapsed_seconds"), "elapsed_seconds")
        allocated = integer(metadata.get("peak_gpu_allocated_bytes"), "peak allocation")
        reserved = metadata.get("peak_gpu_reserved_bytes")
        if reserved is not None:
            integer(reserved, "peak reservation")
            require(reserved >= allocated, "Peak reserved GPU memory is below allocated memory")
        runtime_limit = number(config.get("max_wall_seconds"), "wall limit")
        artifact_paths = [metadata_path, config_path, metrics_path, history_path]
        snapshots.update({path: file_hash(path) for path in artifact_paths})
        runs.append({
            "run_id": run_id, "model": model, "dataset": dataset, "fold": config["fold"], "seed": config["seed"],
            "status": "completed", "protocol": protocol, "test_accessed": False,
            "optimizer_steps": steps, "optimizer_steps_upper_bound": step_limit,
            "epochs_recorded": len(history), "selected_epoch": epoch, "selected_valid_auc": best_auc,
            "validation": validation, "elapsed_seconds": elapsed, "wall_seconds_upper_bound": runtime_limit,
            "soft_elapsed_limit_satisfied": elapsed <= runtime_limit,
            "started_at": started.isoformat(), "finished_at": finished.isoformat(), "ended_at": ended.isoformat(),
            "recorded_time_window_seconds": (ended - started).total_seconds(),
            "peak_gpu_allocated_bytes": allocated, "peak_gpu_reserved_bytes": reserved,
            "train_rows": integer(metadata.get("train_rows"), "train rows", 1),
            "valid_rows": integer(metadata.get("valid_rows"), "validation rows", 1),
            "stop_reason": metadata.get("stop_reason") if metadata.get("stop_reason") in
                           {"epochs_completed", "max_train_steps", "early_stopping"} else "unrecognized",
            "hashes": {"approved_batch_sha256": batch_sha, "implementation_sha256": source_sha,
                       "config_sha256": config_sha, "dataset_manifest_sha256": manifest_sha,
                       "model_source_sha256": metadata["model_source_sha256"],
                       "train_file_sha256": metadata["train_file_sha256"], "valid_file_sha256": metadata["valid_file_sha256"],
                       "checkpoint_sha256": checkpoint_sha, "prediction_sha256": prediction_sha,
                       "metadata_sha256": snapshots[metadata_path], "valid_metrics_sha256": snapshots[metrics_path],
                       "history_sha256": snapshots[history_path]},
        })
    total_steps = sum(run["optimizer_steps"] for run in runs)
    require(declared_steps <= total_limit and total_steps <= total_limit, "Batch optimizer-step limit exceeded")
    for path, expected_hash in snapshots.items():
        require(file_hash(path) == expected_hash, "Saved run evidence changed during report construction")
    require(canonical_hash(read_json(batch_file)) == batch_sha and read_json(approval_path) == approval,
            "Batch or approval changed during report construction")
    per_model = defaultdict(lambda: {"runs": 0, "optimizer_steps": 0, "elapsed_seconds": 0., "peak_gpu_allocated_bytes": 0})
    for run in runs:
        aggregate = per_model[run["model"]]
        aggregate["runs"] += 1
        aggregate["optimizer_steps"] += run["optimizer_steps"]
        aggregate["elapsed_seconds"] += run["elapsed_seconds"]
        aggregate["peak_gpu_allocated_bytes"] = max(aggregate["peak_gpu_allocated_bytes"], run["peak_gpu_allocated_bytes"])
    started = min(timestamp(run["started_at"], "start") for run in runs)
    ended = max(timestamp(run["ended_at"], "end") for run in runs)
    return {
        "schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id, "status": "completed", "test_accessed": False,
        "approval_evidence": {"status": "approved", "approved_at": approved_at.isoformat(), "canonical_batch_sha256": batch_sha},
        "implementation_sha256": source_sha, "runs_expected": len(entries), "runs_completed": len(runs),
        "total_optimizer_steps": total_steps, "total_optimizer_steps_upper_bound": total_limit,
        "sum_elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
        "started_at": started.isoformat(), "ended_at": ended.isoformat(),
        "recorded_batch_time_window_seconds": (ended - started).total_seconds(),
        "peak_gpu_allocated_bytes": max(run["peak_gpu_allocated_bytes"] for run in runs),
        "per_model": dict(per_model), "runs": runs,
        "evidence_scope": {"saved_metrics_only": True, "new_model_forward_calls": 0, "new_optimizer_steps": 0,
                           "prepared_data_records_read": False, "checkpoint_deserialized": False,
                           "prediction_records_parsed": False, "approval_confirmation_text_published": False,
                           "metrics_recomputed": False, "data_files_rehashed": False},
        "interpretation": "Calibration results, not converged quality or SOTA evidence. Different protocols must not be ranked together.",
        "timing_note": "Recorded engine elapsed time excludes preceding model/data setup. Time windows use saved metadata timestamps; wall limits are soft checks.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), "Output already exists; refusing to replace a published report")
        report = summarize(args.repo_root, args.batch)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pending = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=args.output.name + ".", suffix=".tmp", delete=False) as stream:
                pending = Path(stream.name)
                stream.write(encoded)
            require(not args.output.exists(), "Output appeared during report construction")
            pending.replace(args.output)
        finally:
            if pending is not None and pending.exists():
                pending.unlink()
        print(json.dumps({"status": "completed", "batch_id": report["batch_id"], "runs": report["runs_completed"],
                          "optimizer_steps": report["total_optimizer_steps"], "report_sha256": file_hash(args.output)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(f"summarize_batch: verification failed ({type(error).__name__}): {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
