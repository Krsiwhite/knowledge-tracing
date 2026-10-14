#!/usr/bin/env python3
"""Summarize a completed full-v2 development batch using saved receipts only.

Reads the public manifest and each listed run's saved validation CSV, sidecar,
config, metrics, metadata, initial metrics and history. No approval records,
source tree, raw datasets or checkpoint bytes are read; no model is imported.
The report is an allowlist of aggregates, not independent proof of training or
of which checkpoint produced a CSV. Single-seed development is not SOTA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


HELPER = Path(__file__).with_name("analyze_shared_predictions.py")
_spec = importlib.util.spec_from_file_location("_shared_batch_receipt_analysis", HELPER)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)
require, integer = analysis.require, analysis.integer
METRICS = ("auc", "nll", "brier", "accuracy", "ece_15bins")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, f"Invalid {name}")
    return value


def same_number(left, right, name):
    require(math.isclose(number(left, name), number(right, name), rel_tol=1e-10, abs_tol=1e-12),
            f"Inconsistent {name}")


def identifier(value, name):
    require(isinstance(value, str) and SAFE_ID.fullmatch(value), f"Unsafe/missing {name}")
    return value


def timestamp(value, name):
    require(isinstance(value, str), f"Missing {name}")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None, f"Missing timezone in {name}")
    return result


def metric_receipt(value, run):
    identity, metadata, config = run["identity"], run["metadata"], run["config"]
    require(isinstance(value, dict), "Missing validation metrics")
    required = {"protocol": analysis.PROTOCOL, "split": "valid", "model": config["model"],
                "n": identity["selected_targets"], "selection_limited": False,
                "test_accessed": False, "quality_metrics_computed": True, "partial_debug": False,
                "checkpoint_selection_policy": analysis.SELECTION,
                "implementation_sha256": metadata["implementation_sha256"],
                "model_source_sha256": metadata["model_source_sha256"]}
    required.update({key: identity[key] for key in ("source_sha256", "target_sha256", "context_sha256")})
    require(analysis._matches_subset(required, value), "History/initial validation identity mismatch")
    for key in METRICS:
        number(value.get(key), key)
        if key != "nll":
            require(value[key] <= 1, f"Invalid {key}")
    return {key: value[key] for key in METRICS}


def read_history(path, receipts):
    def nonfinite(value):
        raise ValueError("Nonfinite history JSON")
    records = [json.loads(line, object_pairs_hook=analysis._unique_object, parse_constant=nonfinite)
               for line in receipts.read(path).decode("utf-8").splitlines() if line.strip()]
    require(records and all(isinstance(record, dict) for record in records), "Missing/malformed epoch history")
    return records


def replay_history(run, directory, receipts):
    config, metadata = run["config"], run["metadata"]
    epochs = integer(config.get("epochs"), "epochs", 1)
    patience = integer(config.get("patience"), "patience", 1)
    require(config.get("early_stop_min_delta", 0.0) == 0.0, "Only strict-AUC min_delta=0 is supported")
    step_limit = integer(config.get("max_train_steps"), "step limit", 1)
    initial_path, history_path = directory / "initial_valid_metrics.json", directory / "history.jsonl"
    initial = receipts.json(initial_path)
    metric_receipt(initial, run)
    require(initial.get("candidate_epoch") == 0 and type(initial.get("candidate_epoch")) is int
            and initial.get("optimizer_steps") == 0 and type(initial.get("optimizer_steps")) is int
            and initial.get("training_started") is False and initial.get("epoch_completed") is False,
            "Initial candidate must be untrained epoch zero")
    initial_seconds = number(initial.get("validation_seconds"), "initial validation seconds")
    same_number(initial_seconds, metadata.get("initial_validation_seconds"), "initial validation seconds")
    history = read_history(history_path, receipts)
    require(len(history) <= epochs, "History exceeds epoch budget")
    best_epoch, best_auc, best_completed, selected = 0, initial["auc"], False, initial
    previous_steps, previous_elapsed, stale = 0, initial_seconds, 0
    public_history = []
    for index, record in enumerate(history, start=1):
        require(integer(record.get("epoch"), "history epoch", 1) == index, "Epoch history must be consecutive")
        steps = integer(record.get("optimizer_steps"), "history steps", 1)
        require(previous_steps < steps <= step_limit, "History steps are not increasing within budget")
        completed = record.get("epoch_completed")
        require(type(completed) is bool, "Missing epoch completion state")
        require(completed or (index == len(history) and steps == step_limit),
                "Partial epoch is only valid at the terminal step budget")
        elapsed = number(record.get("elapsed_seconds"), "history elapsed seconds")
        require(elapsed >= previous_elapsed, "History elapsed time goes backwards")
        validation = metric_receipt(record.get("valid"), run)
        number(record.get("train_loss"), "training loss")
        train_seconds = number(record.get("train_seconds"), "training seconds")
        validation_seconds = number(record.get("validation_seconds"), "validation seconds")
        if validation["auc"] > best_auc:
            best_epoch, best_auc, best_completed, selected = index, validation["auc"], completed, record["valid"]
            stale = 0
        else:
            stale += 1
        # Both engines give patience priority, including a simultaneous step cap.
        reason = ("early_stopping" if stale >= patience else "max_train_steps" if steps >= step_limit
                  else "epochs_completed" if index == epochs else None)
        require(reason is None or index == len(history), "Training continued after its recorded stop condition")
        public_history.append({"epoch": index, "epoch_completed": completed, "optimizer_steps": steps,
                               "valid_auc": validation["auc"], "train_seconds": train_seconds,
                               "validation_seconds": validation_seconds})
        previous_steps, previous_elapsed = steps, elapsed
    require(reason is not None and metadata.get("stop_reason") == reason, "Terminal stop reason disagrees with history")
    require(metadata.get("optimizer_steps") == previous_steps, "Terminal optimizer steps disagree with metadata")
    require(metadata.get("best_epoch") == best_epoch and metadata.get("best_epoch_completed") is best_completed,
            "Recorded checkpoint is not the earliest strict best validation candidate")
    same_number(metadata.get("best_valid_auc"), best_auc, "best validation AUC")
    # The selected boundary and final export are independent model forwards.
    # This tolerance is declared before inspecting completed batch results;
    # copied receipts and CSV-to-metrics calculations still use strict checks.
    export_deltas = {
        metric: analysis.selected_export_delta(selected[metric], run["metrics"].get(metric), metric=metric)
        for metric in METRICS
    }
    elapsed = number(metadata.get("elapsed_seconds"), "run elapsed seconds")
    require(elapsed >= previous_elapsed, "Final elapsed time precedes final history")
    limit = number(config.get("max_wall_seconds"), "soft wall limit")
    require(limit > 0, "Wall budget must be positive")
    started = timestamp(metadata.get("started_at"), "start time")
    finished = timestamp(metadata.get("finished_at"), "finish time")
    ended = timestamp(metadata.get("ended_at", metadata.get("finished_at")), "end time")
    require(started <= finished <= ended, "Run timestamps are out of order")
    allocated = integer(metadata.get("peak_gpu_allocated_bytes"), "peak allocated memory")
    reserved = metadata.get("peak_gpu_reserved_bytes")
    if reserved is not None:
        require(integer(reserved, "peak reserved memory") >= allocated, "Reserved peak is below allocated peak")
    return {"epochs_recorded": len(history), "epochs_completed": sum(record["epoch_completed"] for record in history),
            "epochs_upper_bound": epochs, "selected_epoch": best_epoch, "selected_epoch_completed": best_completed,
            "selected_checkpoint_stage": run["stage"], "optimizer_steps": previous_steps,
            "optimizer_steps_upper_bound": step_limit, "stop_reason": reason, "patience": patience,
            "terminal_stale_epochs": stale, "initial_valid_auc": initial["auc"],
            "selected_to_export_consistency": {
                "absolute_tolerance": analysis.SELECTED_EXPORT_ABS_TOL,
                "relative_tolerance": 0.0, "delta_direction": "exported_minus_selected_history",
                "metrics": {metric: {"selected_history": selected[metric], "exported": run["metrics"][metric],
                                      "delta": export_deltas[metric]} for metric in METRICS}},
            "history": public_history, "loop_elapsed_seconds": elapsed,
            "initial_validation_seconds": initial_seconds,
            "train_seconds": sum(row["train_seconds"] for row in public_history),
            "epoch_validation_seconds": sum(row["validation_seconds"] for row in public_history),
            "soft_wall_seconds_upper_bound": limit, "soft_wall_limit_exceeded": elapsed > limit,
            "started_at": started.isoformat(), "finished_at": finished.isoformat(), "ended_at": ended.isoformat(),
            "recorded_time_window_seconds": (ended - started).total_seconds(),
            "peak_gpu_allocated_bytes": allocated, "peak_gpu_reserved_bytes": reserved,
            "history_files": {"initial_metrics": receipts.files[initial_path.resolve()],
                              "history": receipts.files[history_path.resolve()]}}


def summarize(repo_root, batch_path):
    root, batch_path = Path(repo_root).resolve(), Path(batch_path)
    batch_path = (batch_path if batch_path.is_absolute() else root / batch_path).resolve()
    receipts = analysis.Receipts()
    batch = receipts.json(batch_path)
    batch_sha = analysis.canonical_hash(batch)
    batch_id = identifier(batch.get("batch_id"), "batch ID")
    require(batch.get("test_access") is False, "Batch test access must remain sealed")
    analysis.digest(batch.get("source_sha256"), "batch implementation")
    entries = batch.get("runs")
    require(isinstance(entries, list) and entries, "Missing batch runs")
    ids = [identifier(entry.get("run_id"), "run ID") for entry in entries]
    require(len(ids) == len(set(ids)), "Duplicate run IDs in batch")
    limits = batch.get("limits", {})
    total_limit = integer(limits.get("total_optimizer_steps_upper_bound"), "total step limit", 1)
    grouped = defaultdict(list)
    run_root = (root / "runs").resolve()
    # Reject any unfinished/missing run before reading large saved CSVs.
    for entry in entries:
        directory = (run_root / entry["run_id"]).resolve()
        require(directory.parent == run_root, "Run path escapes its runs directory")
        metadata = receipts.json(directory / "metadata.json")
        require(metadata.get("status") == "completed", "All manifest runs must be completed")
        config = entry["config"]
        require(config.get("protocol") == analysis.PROTOCOL, "Only the full v2 development protocol is supported")
        group_key = (identifier(config.get("dataset"), "dataset"), config["protocol"],
                     integer(config.get("fold"), "fold"), integer(config.get("seed"), "seed"))
        identifier(config.get("model"), "model")
        grouped[group_key].append(entry)
    declared = sum(integer(entry["config"].get("max_train_steps"), "declared step limit", 1) for entry in entries)
    require(declared <= total_limit, "Declared run budgets exceed batch step limit")
    runs, groups = [], []
    for (dataset, protocol, fold, seed), group_entries in sorted(grouped.items()):
        baseline_rows, baseline_identity, baseline_manifest = None, None, None
        sequence_verified = False
        group_ids, variants = [], set()
        for entry in group_entries:
            directory = run_root / entry["run_id"]
            run = analysis._read_run(directory / "valid_predictions.csv", batch, batch_sha, receipts)
            comparison = analysis.comparison_identity(run["config"])
            variant = analysis.comparison_key(comparison)
            require(variant not in variants, "Duplicate model/recipe for the same dataset/fold/seed")
            variants.add(variant)
            identities = {key: values[0] for key, values in run["rows"].items()}
            if baseline_rows is None:
                baseline_rows, baseline_identity = identities, run["identity"]
                baseline_manifest = run["metadata"]["dataset_manifest_sha256"]
            require(run["identity"] == baseline_identity and identities == baseline_rows,
                    "Unpaired target/history/label identities within dataset/fold/seed")
            require(run["metadata"]["dataset_manifest_sha256"] == baseline_manifest,
                    "Dataset manifest receipts differ within comparison group")
            matches = []
            for name, column, domain in (("target_sha256", 3, b"ktlab.shared.target-sequence.v1\0"),
                                        ("context_sha256", 4, b"ktlab.shared.context-sequence.v1\0")):
                hasher = hashlib.sha256(domain)
                for identity in identities.values():
                    hasher.update(bytes.fromhex(identity[column]))
                matches.append(hasher.hexdigest() == run["identity"][name])
            sequence_verified |= all(matches)
            values = list(run["rows"].values())
            computed = analysis.metrics_for(np.asarray([value[0][1] for value in values], dtype=np.float64),
                                            np.asarray([value[1] for value in values], dtype=np.float64))
            for metric in METRICS:
                same_number(run["metrics"].get(metric), computed[metric], f"saved CSV {metric}")
            history = replay_history(run, directory, receipts)
            metadata = run["metadata"]
            runs.append({"run_id": run["run_id"], **comparison,
                         "dataset": dataset, "protocol": protocol, "fold": fold, "seed": seed,
                         "status": "completed", "validation": {"n": computed["n"], **{k: computed[k] for k in METRICS}},
                         **history, "input_files": run["input_receipts"],
                         "hashes": {key: metadata[key] for key in ("config_sha256", "model_source_sha256",
                                                                  "dataset_manifest_sha256")},
                         "checkpoint_sha256_recorded": metadata["checkpoint_sha256"]})
            group_ids.append(run["run_id"])
        require(sequence_verified, "Aggregate target/context mismatch; no input retains frozen sequence order")
        groups.append({"dataset": dataset, "protocol": protocol, "fold": fold, "seed": seed,
                       "run_ids": group_ids, "selection": baseline_identity,
                       "paired_events": len(baseline_rows), "paired_students": len({key[0] for key in baseline_rows}),
                       "dataset_manifest_sha256_recorded": baseline_manifest,
                       "comparison_scope": "same_saved_development_targets_histories_labels; not SOTA or multiseed evidence"})
    total_steps = sum(run["optimizer_steps"] for run in runs)
    require(total_steps <= total_limit, "Actual batch optimizer steps exceed budget")
    started = min(timestamp(run["started_at"], "start") for run in runs)
    ended = max(timestamp(run["ended_at"], "end") for run in runs)
    receipts.verify()
    return {"schema_version": 1, "summary": "saved_full_shared_development_batch",
            "batch_id": batch_id, "status": "completed", "canonical_batch_sha256": batch_sha,
            "implementation_sha256_recorded": batch["source_sha256"],
            "batch_file": receipts.files[batch_path], "runs_expected": len(entries), "runs_completed": len(runs),
            "total_optimizer_steps": total_steps, "total_optimizer_steps_upper_bound": total_limit,
            "sum_loop_elapsed_seconds": sum(run["loop_elapsed_seconds"] for run in runs),
            "started_at": started.isoformat(), "ended_at": ended.isoformat(),
            "recorded_batch_time_window_seconds": (ended - started).total_seconds(),
            "peak_gpu_allocated_bytes": max(run["peak_gpu_allocated_bytes"] for run in runs),
            "groups": groups, "runs": runs, "ece_bins": 15,
            "receipt_scope": {"saved_receipts_consistent": True, "weights_read_or_verified": False,
                              "direct_checkpoint_to_prediction_hash_binding_available": False,
                              "training_identity_fully_proven": False, "approval_record_read_or_authenticated": False,
                              "source_tree_read_or_verified": False},
            "limitations": ["Validation also selected checkpoints; this descriptive summary is not final-test, statistical-significance or SOTA evidence.",
                            "Unsigned saved receipts do not independently prove training or which checkpoint bytes generated predictions.",
                            "Independent selected-boundary and final-export forwards use a predeclared absolute metric tolerance of 1e-6; strict-best history and CSV metric checks are unchanged.",
                            "Wall budgets use soft checks between operations; exceeded soft timing is reported without asserting unauthorized execution.",
                            "Loop elapsed totals and start/end calendar span have different scopes; no cross-dataset metric average is computed."],
            "new_optimizer_steps": 0, "model_loads": 0, "model_forward_calls": 0,
            "raw_dataset_accessed": False, "test_accessed": False, "automatic_publication": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Output already exists; choose a new report")
        report = summarize(args.repo_root, args.batch)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"runs": report["runs_completed"], "optimizer_steps": report["total_optimizer_steps"],
                          "test_accessed": False, "automatic_publication": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as exc:
        print(f"Summary refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
