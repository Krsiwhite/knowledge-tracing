#!/usr/bin/env python3
"""Describe saved full-v2 validation errors using canonical development data.

Fixed, independent strata: visible history events 1-5/6-20/21-50/>50;
training question-event frequency 0/1-5/6-20/21-100/>100; visible same-question
history 0/>=1; visible history events sharing any query KC 0/>=1. Training
counts include first events and exclude the held-out fold. No model, weights,
test data, significance test or automatic publication is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_RECEIPTS = {}


def _load_code(path, name):
    """Execute exactly the hashed bytes, avoiding any main-checkout .pth import."""
    path = path.resolve()
    content = path.read_bytes()
    CODE_RECEIPTS[path] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass resolves its defining module here.
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


A = _load_code(ROOT / "scripts/analyze_shared_predictions.py", "_shared_error_receipts")
E = _load_code(ROOT / "src/ktlab/shared_events.py", "_shared_error_events")
_script = Path(__file__).resolve()
_content = _script.read_bytes()
CODE_RECEIPTS[_script] = {"sha256": hashlib.sha256(_content).hexdigest(), "bytes": len(_content)}
require = A.require
BINS = {
    "visible_history_events": ["1-5", "6-20", "21-50", ">50"],
    "question_train_event_frequency": ["0", "1-5", "6-20", "21-100", ">100"],
    "visible_history_same_question": ["0", ">=1"],
    "query_kc_visible_history_support": ["0", ">=1"],
}


def _stream_receipt(path, receipts):
    path = path.resolve()
    hasher, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            hasher.update(chunk)
    receipt = {"sha256": hasher.hexdigest(), "bytes": size}
    require(path not in receipts.files or receipts.files[path] == receipt, "Input changed during diagnosis")
    receipts.files[path] = receipt
    return receipt


def training_question_counts(path, manifest, fold):
    """Count canonical events, including each training student's first event."""
    config = manifest["data_config"]
    folds = set(config["folds"])
    require(fold in folds and folds.issubset(set(range(5))), "Invalid development membership")
    frequencies, seen, count, students = Counter(), set(), 0, 0
    for _, _, row in E._csv_rows(path):
        part, uid = int(row["fold"]), row["uid"]
        require(part in folds and uid not in seen, "Duplicate student or non-development fold")
        seen.add(uid)
        if part == fold:
            continue
        events, _ = E._events_from_row(row, int(config["num_c"]), int(config["num_q"]))
        students += 1
        count += len(events)
        for event in events:
            if event.question is not None:
                frequencies[event.question] += 1
    return frequencies, {"folds": sorted(folds - {fold}), "students": students, "canonical_events": count,
                         "first_events_included": True, "counting_unit": "canonical_question_event_not_repeated_KC"}


def stratum_values(context, frequencies, has_questions):
    history, query = context["history"], context["query"]
    count = len(history)
    require(count >= 1, "Scored target has no visible history")
    history_bin = "1-5" if count <= 5 else "6-20" if count <= 20 else "21-50" if count <= 50 else ">50"
    shared_kcs = set(query["concepts"])
    # One history event counts once even when several KCs overlap.
    support = sum(bool(shared_kcs.intersection(event["concepts"])) for event in history)
    result = {"visible_history_events": history_bin,
              "query_kc_visible_history_support": "0" if support == 0 else ">=1"}
    if has_questions:
        frequency = frequencies[query["question"]]
        result["question_train_event_frequency"] = (
            "0" if frequency == 0 else "1-5" if frequency <= 5 else "6-20" if frequency <= 20
            else "21-100" if frequency <= 100 else ">100")
        result["visible_history_same_question"] = (
            ">=1" if any(event["question"] == query["question"] for event in history) else "0")
    return result


def _aggregate(labels, probabilities, students):
    metrics = A.metrics_for(np.asarray(labels, dtype=np.float64), np.asarray(probabilities, dtype=np.float64))
    return {"n": metrics["n"], "students": len(students),
            "positive_rate": float(np.mean(labels)) if labels else None,
            **{key: metrics[key] for key in ("auc", "accuracy", "nll", "brier", "ece_15bins", "auc_unavailable_reason")}}


def diagnose(run_dir, batch_path, *, dataset_manifest=None):
    directory, batch_path = Path(run_dir).resolve(), Path(batch_path).resolve()
    receipts = A.Receipts()
    receipts.files.update(CODE_RECEIPTS)
    batch = receipts.json(batch_path)
    batch_sha = A.canonical_hash(batch)
    require(batch.get("test_access") is False and isinstance(batch.get("runs"), list), "Invalid sealed public batch")
    require(isinstance(batch.get("batch_id"), str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", batch["batch_id"]), "Unsafe batch ID")
    A.digest(batch.get("source_sha256"), "batch source")
    ids = [entry.get("run_id") for entry in batch["runs"]]
    require(all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) for value in ids)
            and len(ids) == len(set(ids)), "Invalid/duplicate batch run IDs")
    run = A._read_run(directory / "valid_predictions.csv", batch, batch_sha, receipts)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run["config"]["dataset"]), "Unsafe dataset name")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run["config"]["model"]), "Unsafe model name")
    manifest_path = Path(dataset_manifest if dataset_manifest is not None else run["config"]["dataset_manifest"]).resolve()
    manifest = receipts.json(manifest_path)
    require(receipts.files[manifest_path]["sha256"] == run["metadata"]["dataset_manifest_sha256"],
            "Canonical source manifest byte receipt mismatch")
    require(manifest.get("dataset") == run["identity"]["dataset"], "Canonical dataset identity mismatch")
    base = Path(manifest["data_dir"])
    base = (base if base.is_absolute() else manifest_path.parent / base).resolve()
    manifest = {**manifest, "data_dir": str(base)}
    require(manifest["data_config"].get("train_valid_original_file") == "train_valid.csv",
            "Only the canonical train_valid.csv development source is allowed")
    source = (base / "train_valid.csv").resolve()
    require(source.parent == base, "Development source escapes its prepared directory")
    source_receipt = _stream_receipt(source, receipts)
    pinned = manifest.get("files", {}).get("train_valid.csv", {})
    require(source_receipt["sha256"] == pinned.get("sha256") == run["identity"]["source_sha256"]
            and source_receipt["bytes"] == pinned.get("bytes"), "Canonical development source receipt mismatch")
    fold = run["config"]["fold"]
    dataset = E.SharedEventDataset(manifest, fold, split="valid")
    try:
        require(A._audit(dataset.audit) == run["identity"], "Canonical validation targets/history differ from saved predictions")
        frequencies, training = training_question_counts(source, manifest, fold)
        has_questions = dataset.num_questions > 0
        holders = {name: {bucket: ([], [], set()) for bucket in bins} for name, bins in BINS.items()
                   if has_questions or name not in {"question_train_event_frequency", "visible_history_same_question"}}
        labels, probabilities, students = [], [], set()
        target_hasher = hashlib.sha256(b"ktlab.shared.target-sequence.v1\0")
        context_hasher = hashlib.sha256(b"ktlab.shared.context-sequence.v1\0")
        require(len(dataset) == len(run["rows"]), "Canonical target count differs from saved predictions")
        remaining = run["rows"]
        for context in dataset:
            key = str(context["uid"]), str(context["event_id"])
            require(key in remaining, "Canonical target identity is missing or duplicated in saved predictions")
            identity, probability = remaining.pop(key)
            digests = E.context_digests(context)
            require(all(context[name] == digests[name] for name in digests), "Canonical context digest changed during diagnosis")
            target_hasher.update(bytes.fromhex(digests["target_sha256"]))
            context_hasher.update(bytes.fromhex(digests["context_sha256"]))
            expected = ("" if context["query"]["question"] is None else str(context["query"]["question"]),
                        context["label"], len(context["query"]["concepts"]),
                        digests["target_sha256"], digests["context_sha256"])
            require(identity == expected, "Saved target question/KCs/history/label do not match canonical development data")
            label = context["label"]
            labels.append(label)
            probabilities.append(probability)
            students.add(key[0])
            for name, bucket in stratum_values(context, frequencies, has_questions).items():
                bucket_labels, bucket_probabilities, bucket_students = holders[name][bucket]
                bucket_labels.append(label)
                bucket_probabilities.append(probability)
                bucket_students.add(key[0])
        require(not remaining, "Saved predictions include noncanonical targets")
        require(target_hasher.hexdigest() == run["identity"]["target_sha256"]
                and context_hasher.hexdigest() == run["identity"]["context_sha256"], "Canonical scored sequence digest mismatch")
        overall = A.metrics_for(np.asarray(labels, dtype=np.float64), np.asarray(probabilities, dtype=np.float64))
        for name in ("auc", "nll", "brier", "accuracy", "ece_15bins"):
            value = run["metrics"].get(name)
            require(type(value) in (int, float) and math.isfinite(value) and overall[name] is not None
                    and math.isclose(value, overall[name], rel_tol=1e-10, abs_tol=1e-12),
                    f"Saved {name} does not match saved probabilities")
        selected_delta = A.selected_export_delta(run["metadata"]["best_valid_auc"], overall["auc"])
        strata = {}
        for name in BINS:
            if name not in holders:
                strata[name] = {"supported": False, "reason": "No question/item IDs in concept-only data (num_q=0)", "bins": None}
            else:
                strata[name] = {"supported": True,
                                "bins": {bucket: _aggregate(*holder) for bucket, holder in holders[name].items()}}
                require(sum(value["n"] for value in strata[name]["bins"].values()) == len(labels), "Incomplete stratum assignment")
    finally:
        dataset.close()
    receipts.verify()
    return {"schema_version": 1, "diagnostic": "saved_shared_validation_independent_error_strata",
            "protocol": A.PROTOCOL, "split": "valid", "run_id": run["run_id"],
            **A.comparison_identity(run["config"]), "dataset": run["identity"]["dataset"],
            "fold": fold, "seed": run["config"]["seed"], "selection": run["identity"],
            "batch_id": batch["batch_id"], "canonical_batch_sha256": batch_sha,
            "overall": _aggregate(labels, probabilities, students), "strata": strata,
            "stratification": {"independent_not_crossed": True, "bins_fixed_before_analysis": BINS,
                               "history": "exact complete-event suffix visible to this target under KC200",
                               "train_frequency": "all canonical events in non-held-out folds, including first events; repeated KCs count once",
                               "query_kc_support": "number of visible history events intersecting any query KC; each event counts once",
                               "ece_bins": 15, "metric_weighting": "events"},
            "training_frequency_scope": training,
            "provenance": {"batch": receipts.files[batch_path], "dataset_manifest": receipts.files[manifest_path],
                           "canonical_development_source": source_receipt, "run_files": run["input_receipts"],
                           "diagnostic_script": CODE_RECEIPTS[_script],
                           "receipt_helper": CODE_RECEIPTS[Path(A.__file__).resolve()],
                           "shared_events_source": CODE_RECEIPTS[Path(E.__file__).resolve()],
                           "training_implementation_sha256_recorded": run["metadata"]["implementation_sha256"],
                           "checkpoint_sha256_recorded": run["metadata"]["checkpoint_sha256"]},
            "selected_export_auc_check": {"delta": selected_delta, "absolute_tolerance": A.SELECTED_EXPORT_ABS_TOL},
            "limitations": ["Descriptive development diagnostics only; validation also selected the checkpoint. No significance tests or SOTA claim.",
                            "Strata are independent summaries, not causal effects. Training question frequency is not inferred from validation frequency.",
                            "Saved receipts do not independently prove training or a direct checkpoint-to-prediction binding. No checkpoint bytes were read.",
                            "The analysis worktree may differ from training; canonical event code is separately hashed and every target/history/label is checked."],
            "development_data_accessed": True, "test_accessed": False, "weights_read_or_verified": False,
            "model_loads": 0, "model_forward_calls": 0, "new_optimizer_steps": 0,
            "significance_tests": False, "automatic_publication": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Output exists; choose a new diagnostic report")
        report = diagnose(args.run_dir, args.batch, dataset_manifest=args.dataset_manifest)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"events": report["overall"]["n"], "test_accessed": False, "model_forward_calls": 0}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as exc:
        print(f"Diagnosis refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
