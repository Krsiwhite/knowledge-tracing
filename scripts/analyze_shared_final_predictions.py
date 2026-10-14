"""Frozen final-test statistics from complete saved receipts; no model/data loader.

Only the release selects inputs, comparisons and bootstrap settings. No weights,
raw test/development CSV, mapping file, torch or evaluator module is read/imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV = "ktlab-v2-shared-event-context-kc200-dev"
FINAL = "ktlab-v2-shared-event-context-kc200-test-v1"
METHOD = "paired-student-mean-run-auc-v1"
SELECTION = "epoch0_and_each_validation_boundary_strict_best_shared_valid_auc"
METRICS = ["auc", "nll", "brier", "accuracy", "ece_15bins"]
FIELDS = ["uid", "event_id", "question_id", "label", "probability", "kc_count", "target_sha256", "context_sha256"]
IDENTITY = ("config_sha256", "implementation_sha256", "model_source_sha256", "validation_source_sha256",
            "validation_target_sha256", "validation_context_sha256", "checkpoint_selection_policy")
RECORD_KEYS = (*IDENTITY, "dataset_manifest_sha256", "checkpoint_sha256", "best_epoch", "best_valid_auc", "best_epoch_completed")
TRAINING_RECIPE_EXCLUDED = {"run_id", "fold", "seed", "device", "dataset_manifest", "_batch_approved",
                            "batch_id", "approved_batch_sha256", "test_release_file", "max_train_steps"}
UKT_AUG = "ukt-author-first-two-from-last-86ad9d8-v1"
MOCKT_RECIPES = {"mockt-target-prefix-branches-v1", "mockt-three-pass-ground-seq-control-v1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer")
    return value


def nonempty(value):
    require(isinstance(value, str) and bool(value.strip()), "Missing text identity")
    return value


def object_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key")
        value[key] = item
    return value


class Receipts:
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        content = path.read_bytes()
        record = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        require(expected is None or record["sha256"] == expected, f"Receipt hash mismatch: {path.name}")
        require(path not in self.files or self.files[path] == record, "Input changed during analysis")
        self.files[path] = record
        return content

    def json(self, path, expected=None):
        path = Path(path).resolve()
        require(path.suffix == ".json" and path.name != "keyid2idx.json", "Expected a control/artifact JSON, never mapping or raw data")
        def invalid(_):
            raise ValueError("Nonfinite JSON")
        value = json.loads(self.read(path, expected), object_pairs_hook=object_pairs, parse_constant=invalid)
        require(isinstance(value, dict), "Expected JSON object")
        return value

    def verify(self):
        for path, expected in self.files.items():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            require(hasher.hexdigest() == expected["sha256"] and path.stat().st_size == expected["bytes"],
                    "Input changed during analysis")


def _path(value, *, name=None):
    path = Path(nonempty(value))
    require(path.is_absolute() and path == path.resolve(), "Frozen paths must be absolute and canonical")
    require(name is None or path.name == name, f"Expected canonical {name}")
    return path


def _fixed_file(directory, name):
    path = directory / name
    require(path.resolve() == path, "Artifact path redirects outside its canonical location")
    return path


def _load_pure(path, expected, receipts, name):
    content = receipts.read(path, digest(expected))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


def _recipe(config):
    model = config.get("model")
    require(model in {"akt", "simplekt", "mockt", "ukt", "ktst_mean", "ktst_mhsa"}, "Unsupported model identity")
    settings = config.get("model_config")
    require(isinstance(settings, dict) and settings, "Resolved model_config is required")
    ukt, moc = config.get("ukt_training_recipe"), config.get("mockt_training_recipe")
    if model == "ukt":
        flags = []
        for key in ("use_CL", "use_uncertainty_aug"):
            value = settings.get(key)
            require(type(value) is bool or (type(value) is int and value in (0, 1)), "UKT switches must be explicit")
            flags.append(bool(value))
        expected = "ukt-no-cl" if not flags[0] else UKT_AUG if flags[1] else "ukt-cl-same-responses"
        require(ukt == expected, "UKT recipe mismatch")
    else:
        require(ukt is None, "UKT recipe on another model")
    if moc is not None:
        require(model == "mockt" and moc in MOCKT_RECIPES, "Unsupported MoC recipe")
        require(settings.get("emb_type", "qid_conv_ker_noexp") == "qid_conv_ker_noexp", "MoC embedding mismatch")
        for key, value in (("kernel_size", 4), ("seq_len", 200), ("sequence1", 50), ("sequence2", 150)):
            require(type(settings.get(key, value)) is int and settings.get(key, value) == value, "MoC configuration mismatch")
    return {"model": model, "ukt_training_recipe": ukt, "mockt_training_recipe": moc,
            "model_config_sha256": canonical(settings),
            "training_recipe_sha256": canonical({key: value for key, value in config.items() if key not in TRAINING_RECIPE_EXCLUDED})}


def _plan(release, plan):
    require(plan.get("schema_version") == 1 and plan.get("protocol") == FINAL, "Wrong final analysis plan")
    run_ids = plan.get("run_ids")
    require(isinstance(run_ids, list) and all(isinstance(x, str) for x in run_ids)
            and len(run_ids) == len(set(run_ids)) and set(run_ids) == set(release["runs"]), "Plan/release candidate set mismatch")
    require(plan.get("metrics") == METRICS and plan.get("accuracy_threshold") == .5 and plan.get("ece_bins") == 15
            and plan.get("kc_budget") == 200 and plan.get("test_selection") == "all_eligible_complete_event_kc200"
            and plan.get("max_targets") is None, "Changed final selection or metrics")
    integer(plan.get("batch_size"), 1); integer(plan.get("max_rows_per_forward"), 1)
    s = plan.get("final_statistics", {})
    require(s.get("schema_version") == 1 and s.get("method") == METHOD, "Missing frozen final_statistics v1")
    require(s.get("folds") == [0, 1, 2, 3, 4] and s.get("seeds") == [42, 2026, 3407], "Final statistics require the full 5-fold x 3-seed grid")
    for value in (*s["folds"], *s["seeds"]): integer(value)
    require(s.get("metrics") == METRICS and s.get("auxiliary_intervals") == "none"
            and s.get("dispersion") == "sample_std_ddof1_descriptive"
            and s.get("multiple_comparisons") == "bonferroni_auc_family", "Unsupported final statistical definitions")
    bootstrap = s.get("bootstrap", {})
    for key in ("n_resamples", "seed", "family_size", "max_attempts"): integer(bootstrap.get(key))
    require(bootstrap == {"n_resamples": 2000, "seed": 42, "seed_scheme": "reset_42_per_dataset", "confidence": .95,
                           "family_size": len(s.get("comparisons", [])), "max_attempts": 20000,
                           "single_class_draws": "bounded_redraw_condition_on_two_classes"}, "Changed frozen bootstrap v1 scheme")
    methods, cohort, comparisons = s.get("methods"), s.get("cohort"), s.get("comparisons")
    require(isinstance(methods, dict) and methods and isinstance(cohort, list) and cohort
            and isinstance(comparisons, list) and comparisons, "Missing complete cohort/methods/comparisons")
    for key, method in methods.items():
        nonempty(key); nonempty(method.get("label")); digest(method.get("model_config_sha256"))
        digest(method.get("training_recipe_sha256"))
        require(set(method) == {"label", "model", "ukt_training_recipe", "mockt_training_recipe", "model_config_sha256", "training_recipe_sha256"},
                "Method declarations require explicit model, recipes and configuration hash")
    by_run, grids = {}, {}
    expected_grid = {(fold, seed) for fold in s["folds"] for seed in s["seeds"]}
    directories = set()
    for entry in cohort:
        run = nonempty(entry.get("run_id")); dataset = nonempty(entry.get("dataset")); method = entry.get("method_id")
        require(run not in by_run and method in methods, "Duplicate cohort run or unknown method")
        directory = _path(entry.get("run_dir"))
        require(directory.name == run and directory not in directories, "Cohort run directory mismatch or duplicate")
        directories.add(directory)
        pair = integer(entry.get("fold")), integer(entry.get("seed"))
        integer(entry.get("max_train_steps"), 1)
        grid = grids.setdefault((dataset, method), {})
        require(pair in expected_grid and pair not in grid, "Duplicate/invalid cohort fold-seed")
        grid[pair] = run; by_run[run] = entry
    require(set(by_run) == set(release["runs"]), "Cohort must contain exactly the complete release")
    require(all(set(grid) == expected_grid for grid in grids.values()), "Incomplete 15-fit method grid")
    for dataset in {key[0] for key in grids}:
        declarations = [canonical({k: v for k, v in methods[method].items() if k != "label"})
                        for name, method in grids if name == dataset]
        require(len(declarations) == len(set(declarations)), "Duplicate method identity disguised by labels")
    require(set(methods) == {entry["method_id"] for entry in cohort}, "Unused/omitted method declaration")
    ids, pairs_seen, used, datasets = set(), set(), set(), set()
    for comparison in comparisons:
        cid = nonempty(comparison.get("comparison_id")); dataset = nonempty(comparison.get("dataset"))
        candidate, reference = comparison.get("candidate_method"), comparison.get("reference_method")
        require(cid not in ids and candidate != reference and (dataset, candidate) in grids and (dataset, reference) in grids,
                "Duplicate/invalid comparison")
        key = (dataset, *sorted((candidate, reference)))
        require(key not in pairs_seen, "Duplicate or reversed comparison")
        ids.add(cid); pairs_seen.add(key); datasets.add(dataset)
        pairs = comparison.get("pairs")
        require(isinstance(pairs, list) and len(pairs) == 15, "Comparison requires exactly 15 matched pairs")
        found = set()
        for pair in pairs:
            grid_key = integer(pair.get("fold")), integer(pair.get("seed"))
            require(grid_key in expected_grid and grid_key not in found, "Duplicate/missing comparison grid")
            require(pair.get("candidate_run_id") == grids[(dataset, candidate)][grid_key]
                    and pair.get("reference_run_id") == grids[(dataset, reference)][grid_key], "Wrong candidate/reference pair")
            found.add(grid_key); used.update((pair["candidate_run_id"], pair["reference_run_id"]))
    require(used == set(by_run) and datasets == {entry["dataset"] for entry in cohort}, "Uncompared candidate or dataset")
    return s, by_run


def _audit(audit, protocol, split, dataset, fold):
    require(audit.get("protocol") == protocol and audit.get("split") == split and audit.get("dataset") == dataset
            and audit.get("fold") == fold and type(audit.get("fold")) is int, "Audit protocol/dataset/fold mismatch")
    require(audit.get("test_access") is (split == "test"), "Audit access scope mismatch")
    require(audit.get("kc_budget") == 200 and audit.get("max_targets") is None
            and audit.get("selection_limited") is False and audit.get("selected_for_debug") is False,
            "Partial/debug audit is forbidden")
    n = integer(audit.get("selected_targets"), 1)
    require(n == audit.get("expected_total") == audit.get("eligible_total"), "Incomplete event audit")
    for kind in ("source", "target", "context"):
        digest(audit.get(kind + "_sha256"))
    return n


def _candidate(entry, record, method, receipts):
    directory = _path(entry["run_dir"])
    config_path, metadata_path = _fixed_file(directory, "resolved_config.json"), _fixed_file(directory, "metadata.json")
    config, metadata = receipts.json(config_path), receipts.json(metadata_path, digest(record.get("metadata_sha256")))
    require(config.get("protocol") == metadata.get("protocol") == record.get("training_protocol") == DEV, "No training-protocol relabeling")
    require(metadata.get("run_id") == directory.name and metadata.get("status") == "completed"
            and metadata.get("test_accessed") is False and metadata.get("partial_debug") is False, "Training candidate not completed/full/untested")
    require(not config.get("debug_smoke") and config.get("max_validation_targets") is None
            and config.get("max_validation_batches") is None and config.get("initial_validation") is True
            and metadata.get("initial_validation") is True, "Debug or wrong checkpoint selection")
    for key in ("dataset", "fold", "seed"):
        require(type(config.get(key)) is type(entry[key]) and config[key] == entry[key], "Cohort configuration identity mismatch")
    require(type(config.get("max_train_steps")) is int and config["max_train_steps"] == entry["max_train_steps"],
            "Per-run frozen training step budget mismatch")
    require(metadata.get("seed") == config["seed"] and metadata.get("checkpoint_selection_policy") == SELECTION,
            "Seed/checkpoint-selection mismatch")
    config_sha = canonical({key: value for key, value in config.items() if key not in {"_batch_approved", "test_release_file"}})
    require(config_sha == record.get("config_sha256"), "Original config fingerprint mismatch")
    require(_recipe(config) == {k: v for k, v in method.items() if k != "label"}, "Frozen method/model/recipe configuration mismatch")
    for key in RECORD_KEYS:
        require(metadata.get(key) == record.get(key), f"Original training {key} mismatch")
        if key.endswith("sha256"): digest(record.get(key))
    require(record.get("checkpoint_selection_policy") == SELECTION, "Wrong original selection policy")
    epoch = integer(record.get("best_epoch")); completed = record.get("best_epoch_completed")
    auc = record.get("best_valid_auc")
    require(type(completed) is bool and not (epoch == 0 and completed) and type(auc) in (int, float)
            and math.isfinite(auc) and 0 <= auc <= 1, "Invalid selected epoch0/epoch/AUC state")
    integer(metadata.get("optimizer_steps"))  # Epoch0 can win after subsequent positive training steps.
    require(metadata["optimizer_steps"] <= config["max_train_steps"], "Recorded optimizer steps exceed frozen run budget")
    require(epoch <= integer(config.get("epochs"), 1), "Selected epoch exceeds frozen training budget")
    valid = metadata.get("valid_event_audit", {})
    _audit(valid, DEV, "valid", entry["dataset"], entry["fold"])
    for kind in ("source", "target", "context"):
        require(valid[kind + "_sha256"] == record["validation_" + kind + "_sha256"], "Original validation identity mismatch")
    manifest_path = _path(config.get("dataset_manifest"), name="manifest.json")
    manifest = receipts.json(manifest_path, record["dataset_manifest_sha256"])
    require(manifest.get("dataset") == entry["dataset"] and manifest.get("split_audit", {}).get("train_test_disjoint") is True,
            "Prepared manifest/split identity mismatch")
    base = Path(manifest["data_dir"])
    base = (base if base.is_absolute() else manifest_path.parent / base).resolve()
    require(not directory.is_relative_to(base), "Run artifacts cannot be prepared data")
    dc = manifest.get("data_config", {})
    require(dc.get("test_original_file") == "test.csv" and dc.get("maxlen") == 200
            and entry["fold"] in dc.get("folds", []), "Wrong held-out data configuration")
    integer(dc.get("num_c"), 1); integer(dc.get("num_q"))
    for name in ("test.csv", "keyid2idx.json"):
        digest(manifest.get("files", {}).get(name, {}).get("sha256")); integer(manifest["files"][name].get("bytes"), 1)
    return {"directory": directory, "config": config, "metadata": metadata, "manifest": manifest, "base": base,
            "manifest_path": manifest_path, "stage": "initial_untrained" if epoch == 0 else "after_training_epoch" if completed else "after_partial_training_epoch",
            "inputs": {"config": receipts.files[config_path], "metadata": receipts.files[metadata_path], "manifest": receipts.files[manifest_path]}}


def _read_final(entry, record, candidate, release, release_sha, receipts, reference_rows, metric_helpers):
    directory, base, manifest = candidate["directory"], candidate["base"], candidate["manifest"]
    output = _fixed_file(directory, "final_test")
    source_sha = manifest["files"]["test.csv"]["sha256"]
    test_identity = {"run_id": entry["run_id"], "checkpoint_sha256": record["checkpoint_sha256"],
                     "dataset_manifest_sha256": record["dataset_manifest_sha256"], "test_source_sha256": source_sha}
    dataset_identity = {"dataset": entry["dataset"], "test_source_sha256": source_sha,
                        "mapping_sha256": manifest["files"]["keyid2idx.json"]["sha256"]}
    registry = _fixed_file(base, ".final-test-attempts")
    attempt_path = _fixed_file(registry, canonical(test_identity) + ".json")
    lock_path = _fixed_file(registry, "dataset-" + canonical(dataset_identity) + ".json")
    attempt = receipts.json(attempt_path)
    require(attempt.get("schema_version") == 1 and attempt.get("status") == "completed"
            and attempt.get("test_access_attempted") is True and attempt.get("automatic_retry_allowed") is False,
            "Final attempt is not completed; no partial/failed/active result")
    require(attempt.get("test_identity") == test_identity and attempt.get("release_sha256") == release_sha
            and attempt.get("evaluation_source_sha256") == release["source_sha256"]
            and attempt.get("output_directory") == str(output), "Canonical final attempt identity mismatch")
    nonempty(attempt.get("created_at")); nonempty(attempt.get("finished_at"))
    start = {"schema_version": 1, "status": "started", "created_at": attempt["created_at"], "test_identity": test_identity,
             "release_sha256": release_sha, "evaluation_source_sha256": release["source_sha256"],
             "output_directory": str(output), "test_access_attempted": True, "automatic_retry_allowed": False}
    serialized = json.dumps(start, indent=2, allow_nan=False) + "\n"
    initial_hashes = {hashlib.sha256(text.encode()).hexdigest() for text in (serialized, serialized.replace("\n", "\r\n"))}
    require(attempt.get("attempt_start_sha256") in initial_hashes, "Initial attempt byte receipt mismatch")
    expected_lock = {"schema_version": 1, "dataset_identity": dataset_identity, "release_sha256": release_sha,
                     "analysis_plan_sha256": release["analysis_plan_sha256"], "compatibility_review_sha256": release["compatibility_review_sha256"],
                     "evaluation_source_sha256": release["source_sha256"], "candidates_sha256": canonical(release["runs"])}
    require(receipts.json(lock_path) == expected_lock, "Canonical dataset release lock mismatch")
    paths = {"result": _fixed_file(output, "result.json"), "sidecar": _fixed_file(output, "predictions.csv.manifest.json"),
             "prediction": _fixed_file(output, "predictions.csv")}
    result = receipts.json(paths["result"], digest(attempt.get("result_sha256")))
    sidecar = receipts.json(paths["sidecar"], digest(attempt.get("prediction_manifest_sha256")))
    require(sidecar.get("schema_version") == 1 and sidecar.get("metric_scope") == "released_final_test", "Wrong final sidecar scope")
    expected_identity = {"run_id": entry["run_id"], "training_identity": record, "training_fold": entry["fold"], "training_seed": entry["seed"],
        "evaluation_protocol": FINAL, "evaluation_implementation_sha256": release["source_sha256"], "test_identity": test_identity,
        "test_release_sha256": release_sha, "analysis_plan_sha256": release["analysis_plan_sha256"],
        "compatibility_review_sha256": release["compatibility_review_sha256"], "attempt_file": str(attempt_path),
        "attempt_start_sha256": attempt["attempt_start_sha256"], "dataset_release_lock_file": str(lock_path),
        "dataset_release_lock_sha256": receipts.files[lock_path]["sha256"], "original_candidate_receipts_verified": True,
        "compatibility_review_required_not_independently_proven": True}
    audit = sidecar.get("dataset_audit", {})
    n = _audit(audit, FINAL, "test", entry["dataset"], -1)
    require(audit.get("test_access") is True and audit.get("source_sha256") == source_sha
            and audit.get("source_path") == str(base / "test.csv")
            and audit.get("hash_scheme") == "ktlab-shared-event-compositional-sha256-v1"
            and audit.get("hash_scope") == "selected_targets_in_original_csv_and_event_order", "Wrong held-out event audit")
    fixed_semantics = {
        "hash_labels": "target labels excluded; permitted historical responses included in context only",
        "event_id_scheme": "dataset:uid:original_student_event_ordinal",
        "question_missing_policy": "None; any model-specific dummy belongs to its adapter",
        "kc_order": "original occurrence order with within-event duplicates removed",
        "history_policy": "longest complete contiguous event suffix; history plus query KC count <= 200",
    }
    require(all(audit.get(key) == value for key, value in fixed_semantics.items()), "Final event/history semantic audit mismatch")
    require(integer(audit.get("source_bytes"), 1) == manifest["files"]["test.csv"]["bytes"], "Test source byte receipt mismatch")
    source_students, split_students, indexed_students = (integer(audit.get(key), 1) for key in ("source_students", "split_students", "indexed_students"))
    require(source_students == split_students and indexed_students <= split_students, "Final student audit mismatch")
    exclusions, reasons = audit.get("exclusion_counts", {}), audit.get("no_history_reasons", {})
    require(set(exclusions) == {"target_over_budget", "no_history"}
            and set(reasons) == {"first_event", "immediate_predecessor_exceeds_remaining_budget"}, "Missing exclusion audit")
    require(integer(audit.get("canonical_events"), 1) == n + sum(integer(x) for x in exclusions.values())
            and sum(integer(x) for x in reasons.values()) == exclusions["no_history"], "Unbalanced complete event audit")
    integer(audit.get("duplicate_kc_positions_removed"))
    require(audit.get("optimizer_steps") == 0 and type(audit.get("optimizer_steps")) is int, "Dataset audit is not read-only")
    saved = sidecar.get("result", {})
    require(all(key in result and result[key] == value for key, value in saved.items()), "Result differs from sidecar")
    for value in (saved, result):
        require(all(value.get(key) == expected for key, expected in expected_identity.items()), "Training/evaluation/output identity mismatch")
        require(value.get("protocol") == FINAL and value.get("split") == "test" and value.get("test_accessed") is True
                and value.get("selection_limited") is False and value.get("quality_metrics_computed") is True
                and value.get("model_training_and_checkpoint_selection_not_verified") is True, "Wrong final result scope")
        require(value.get("model") == candidate["config"]["model"] and value.get("model_source_sha256") == record["model_source_sha256"]
                and value.get("implementation_sha256") == release["source_sha256"] and value.get("n") == n, "Model/evaluator receipt mismatch")
        require(all(value.get(key) == audit[key] for key in ("source_sha256", "target_sha256", "context_sha256")), "Final selection identity mismatch")
    require(result.get("prediction_file") == str(paths["prediction"])
            and result.get("prediction_manifest_sha256") == attempt["prediction_manifest_sha256"], "Final output file receipt mismatch")
    prediction_sha = digest(attempt.get("prediction_sha256"))
    require(result.get("prediction_sha256") == sidecar.get("prediction_sha256") == prediction_sha, "Prediction receipt chain mismatch")
    content = receipts.read(paths["prediction"], prediction_sha)
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    require(reader.fieldnames == FIELDS, "Wrong prediction columns")
    rows, probabilities, seen = [], [], set()
    target_hash = hashlib.sha256(b"ktlab.shared.target-sequence.v1\0")
    context_hash = hashlib.sha256(b"ktlab.shared.context-sequence.v1\0")
    for index, row in enumerate(reader):
        require(None not in row and all(v is not None for v in row.values()), "Malformed prediction row")
        key = nonempty(row["uid"]), nonempty(row["event_id"])
        event_prefix = f"{entry['dataset']}:{key[0]}:"
        require(key[1].startswith(event_prefix) and re.fullmatch(r"[1-9][0-9]*", key[1][len(event_prefix):]) is not None,
                "Event ID does not preserve dataset/student/original positive ordinal")
        require(key not in seen and row["label"] in {"0", "1"}, "Duplicate event or invalid label")
        seen.add(key)
        require(re.fullmatch(r"[1-9][0-9]*", row["kc_count"]) is not None and int(row["kc_count"]) <= 199, "Invalid target KC count")
        question = row["question_id"]
        require(question == "" or re.fullmatch(r"0|[1-9][0-9]*", question) is not None, "Invalid saved question ID")
        num_q = manifest["data_config"]["num_q"]
        require((num_q == 0 and question == "") or (num_q > 0 and question != "" and int(question) < num_q), "Question/vocabulary mismatch")
        target, context = bytes.fromhex(digest(row["target_sha256"])), bytes.fromhex(digest(row["context_sha256"]))
        identity = (*key, question, int(row["label"]), int(row["kc_count"]), target, context)
        if reference_rows is not None:
            require(index < len(reference_rows) and identity == reference_rows[index], "Per-event labels/questions/KCs/targets/histories/order differ")
        else:
            rows.append(identity)
        p = float(row["probability"])
        require(math.isfinite(p) and 0 <= p <= 1, "Invalid probability")
        probabilities.append(p); target_hash.update(target); context_hash.update(context)
    require(len(probabilities) == n and (reference_rows is None or len(reference_rows) == n), "Incomplete prediction target set")
    require(len({key[0] for key in seen}) == indexed_students, "CSV students disagree with full audit")
    require(target_hash.hexdigest() == audit["target_sha256"] and context_hash.hexdigest() == audit["context_sha256"], "Ordered target/context digest mismatch")
    rows = rows if reference_rows is None else reference_rows
    labels = np.fromiter((row[3] for row in rows), dtype=np.float64, count=n)
    scores = np.asarray(probabilities, dtype=np.float64)
    metrics = metric_helpers.metrics_for(labels, scores)
    for key in METRICS:
        expected, actual = metrics[key], result.get(key)
        require(actual is None if expected is None else type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"Saved CSV {key} mismatch")
    require(result.get("auc_available") is (metrics["auc"] is not None)
            and result.get("auc_unavailable_reason") == (None if metrics["auc"] is not None else "single_response_class"), "AUC availability receipt mismatch")
    return rows, scores, metrics, {**candidate["inputs"], **{key: receipts.files[path] for key, path in paths.items()},
                                 "attempt": receipts.files[attempt_path], "dataset_lock": receipts.files[lock_path]}, audit


def _describe(values):
    require(values, "Empty metric grid")
    if any(value is None for value in values):
        return {"n": len(values), "mean": None, "sample_std": None, "unavailable": True}
    return {"n": len(values), "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)), "unavailable": False}


def analyze(release_file):
    receipts = Receipts()
    release_path = Path(release_file).resolve()
    release = receipts.json(release_path)
    require(release.get("schema_version") == 1 and release.get("protocol") == FINAL and release.get("status") == "approved"
            and release.get("user_confirmation") and release.get("approved_at"), "Approved final release required")
    digest(release.get("source_sha256"))
    require(isinstance(release.get("runs"), dict) and release["runs"], "Empty release candidates")
    release_sha = receipts.files[release_path]["sha256"]
    plan_path, review_path = _path(release.get("analysis_plan_file")), _path(release.get("compatibility_review_file"))
    plan = receipts.json(plan_path, digest(release.get("analysis_plan_sha256")))
    review = receipts.json(review_path, digest(release.get("compatibility_review_sha256")))
    require(review.get("schema_version") == 1 and review.get("status") == "approved" and review.get("reviewed_by") and review.get("statement")
            and review.get("training_protocol") == DEV and review.get("evaluation_protocol") == FINAL
            and review.get("evaluation_source_sha256") == release["source_sha256"] and review.get("candidates") == release["runs"], "Compatibility review mismatch")
    s, by_run = _plan(release, plan)
    receipts.read(Path(__file__).resolve(), digest(s.get("reader_sha256")))
    stats = _load_pure(ROOT / "src/ktlab/statistics.py", s.get("statistics_sha256"), receipts, "_ktlab_frozen_final_statistics")
    helpers = _load_pure(ROOT / "scripts/analyze_shared_predictions.py", s.get("metric_helpers_sha256"), receipts, "_ktlab_final_pure_metrics")
    datasets = []
    for dataset in sorted({entry["dataset"] for entry in by_run.values()}):
        entries = [entry for entry in s["cohort"] if entry["dataset"] == dataset]
        reference_rows, identity, manifest_sha = None, None, None
        runs, saved = {}, []
        for entry in entries:
            run_id = entry["run_id"]; record = release["runs"][run_id]
            candidate = _candidate(entry, record, s["methods"][entry["method_id"]], receipts)
            require(all(not path.is_relative_to(candidate["base"]) for path in (release_path, plan_path, review_path)), "Control records cannot be prepared data")
            rows, probabilities, metrics, inputs, audit = _read_final(entry, record, candidate, release, release_sha, receipts, reference_rows, helpers)
            current = {key: audit[key] for key in ("protocol", "dataset", "fold", "source_sha256", "target_sha256", "context_sha256", "selected_targets")}
            require(identity is None or current == identity, "Dataset test selections differ")
            require(manifest_sha is None or manifest_sha == record["dataset_manifest_sha256"], "Dataset manifest differs across fits")
            reference_rows, identity, manifest_sha = rows, current, record["dataset_manifest_sha256"]
            runs[run_id] = {"entry": entry, "probabilities": probabilities, "metrics": metrics}
            saved.append({key: entry[key] for key in ("run_id", "dataset", "method_id", "fold", "seed", "max_train_steps")} |
                {"metrics": {key: metrics[key] for key in ("n", *METRICS)}, "training_identity": record,
                 "selected_checkpoint_stage": candidate["stage"], "model_config": candidate["config"]["model_config"], "input_receipts": inputs})
        labels = np.fromiter((row[3] for row in reference_rows), dtype=np.float64)
        users, events = [row[0] for row in reference_rows], [row[1] for row in reference_rows]
        comparisons = []
        for comparison in s["comparisons"]:
            if comparison["dataset"] != dataset: continue
            per_pair, candidate_runs, reference_runs = [], [], []
            for pair in comparison["pairs"]:
                c, r = runs[pair["candidate_run_id"]], runs[pair["reference_run_id"]]
                per_pair.append({**pair, "candidate": {key: c["metrics"][key] for key in METRICS},
                                 "reference": {key: r["metrics"][key] for key in METRICS},
                                 "delta": {key: None if c["metrics"][key] is None else c["metrics"][key] - r["metrics"][key] for key in METRICS}})
                for target, run in ((candidate_runs, c), (reference_runs, r)):
                    target.append(stats.RunPredictions(pair["seed"], pair["fold"], users, events, labels, run["probabilities"], run["entry"]["run_id"]))
            bootstrap, unavailable = None, None
            if np.unique(labels).size < 2:
                unavailable = "single_response_class"
            elif len(set(users)) < 2:
                unavailable = "fewer_than_two_student_clusters"
            else:
                bootstrap = stats.paired_auc_bootstrap(candidate_runs, reference_runs, n_resamples=2000, confidence=.95,
                    family_size=s["bootstrap"]["family_size"], random_state=42, max_attempts=20000)
            comparisons.append({key: comparison[key] for key in ("comparison_id", "candidate_method", "reference_method")} |
                {"pairs": per_pair, "descriptive": {side: {metric: _describe([pair[side][metric] for pair in per_pair]) for metric in METRICS}
                                                   for side in ("candidate", "reference", "delta")},
                 "primary_auc": bootstrap, "auc_interval_unavailable_reason": unavailable,
                 "positive_adjusted_auc_interval": bool(bootstrap is not None and bootstrap["auc_delta"] > 0 and bootstrap["confidence_interval"][0] > 0)})
        cost = {"events_per_fit": len(labels), "fits": len(runs), "primary_comparisons": len(comparisons),
                "stored_probability_array_bytes": len(labels) * len(runs) * 8,
                "accepted_draw_event_fit_visits": len(labels) * 30 * 2000 * len(comparisons),
                "max_attempt_event_fit_visits_upper_bound": len(labels) * 30 * 20000 * len(comparisons),
                "processing": "one_dataset_at_a_time; cached_AUC_sort_and_ties; one_student_weight_vector_per_draw",
                "memory_scope": "score arrays only estimate excludes parsed identities, library validation copies, rank caches and Python overhead",
                "automatic_subsampling_or_fit_reduction": False}
        datasets.append({"dataset": dataset, "selection": identity, "dataset_manifest_sha256": manifest_sha,
                         "events": len(labels), "students": len(set(users)), "runs": saved, "comparisons": comparisons, "cost_estimate": cost})
        # Do not retain another dataset's arrays while analyzing the next one.
        del runs, reference_rows, rows, probabilities, c, r, run, target, labels, users, events, candidate_runs, reference_runs
    receipts.verify()
    return {"schema_version": 1, "analysis": METHOD, "protocol": FINAL, "status": "completed",
            "release_sha256": release_sha, "analysis_plan_sha256": release["analysis_plan_sha256"],
            "compatibility_review_sha256": release["compatibility_review_sha256"],
            "evaluation_source_sha256_recorded": release["source_sha256"],
            "analysis_sources": {key: s[key] for key in ("reader_sha256", "statistics_sha256", "metric_helpers_sha256")},
            "frozen_methods": s["methods"], "bootstrap_plan": s["bootstrap"], "datasets": datasets,
            "training_recipe_fingerprint_excludes": sorted(TRAINING_RECIPE_EXCLUDED),
            "complete_release_candidates": len(by_run), "source_control_receipts": {
                "release": receipts.files[release_path], "plan": receipts.files[plan_path], "review": receipts.files[review_path]},
            "receipt_scope": {"unsigned_saved_receipts_consistent": True, "weights_read_or_verified": False,
                              "raw_datasets_or_mapping_read": False, "training_or_access_order_independently_proven": False},
            "model_forward_calls": 0, "gradient_calls": 0, "optimizer_steps": 0,
            "limitations": ["Intervals condition on these frozen fitted models and test-student sampling, not retraining uncertainty.",
                "Same-dataset comparisons reset seed 42 and share student draws; datasets are not pooled or paired. Reset RNG streams are not claimed mutually independent.",
                "Bonferroni percentile coverage is approximate; per-seed intervals are descriptive. No p-values are manufactured.",
                "NLL/Brier/accuracy/ECE have descriptive run and pair summaries only. Fits share test students and overlapping training sets.",
                "Unsigned receipts do not independently prove training, weight contents, mathematical compatibility, or the historical order of test access.",
                "Saved context hashes are checked for exact agreement, not reconstructed from raw histories. No automatic SOTA claim is made."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), "Output already exists")
        report = analyze(args.release)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
        print(json.dumps({"status": report["status"], "datasets": len(report["datasets"]), "runs": report["complete_release_candidates"],
                          "model_forward_calls": 0, "raw_datasets_or_mapping_read": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as error:
        print(f"Final statistics refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
