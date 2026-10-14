"""Offline, receipt-checked exploratory analysis of saved full v2 validation.

Only NumPy and the standard library are imported. Never opens data sources,
weights, approval records, or model code. See docs/SAVED_PREDICTION_ANALYSIS.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from os import PathLike
from pathlib import Path
import re
import sys

import numpy as np


PROTOCOL = "ktlab-v2-shared-event-context-kc200-dev"
SELECTION = "epoch0_and_each_validation_boundary_strict_best_shared_valid_auc"
FIELDS = ["uid", "event_id", "question_id", "label", "probability", "kc_count",
          "target_sha256", "context_sha256"]
IDENTITY = ("protocol", "dataset", "split", "fold", "source_sha256", "target_sha256",
            "context_sha256", "selected_targets")
MAX_BOOTSTRAP_EVENT_VISITS = 500_000_000
# Fixed before inspecting B003 terminal results; only for separate forwards.
SELECTED_EXPORT_ABS_TOL = 1e-6
UKT_AUTHOR_AUGMENTATION = "ukt-author-first-two-from-last-86ad9d8-v1"
UKT_RECIPES = {"ukt-no-cl", "ukt-cl-same-responses", UKT_AUTHOR_AUGMENTATION}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid {name} SHA256")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {name} integer")
    return value


def selected_export_delta(selected, exported, *, metric="auc"):
    """Check a separate export forward against selection; return signed drift.

    Do not use this tolerance for metrics recomputed from the same CSV or for
    copied receipt values. No relative tolerance or silent rounding is applied.
    """
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (selected, exported)),
            f"Nonfinite or nonnumeric selected/exported {metric}")
    delta = float(exported - selected)
    require(abs(delta) <= SELECTED_EXPORT_ABS_TOL,
            f"Exported {metric} differs from selected value beyond absolute tolerance {SELECTED_EXPORT_ABS_TOL}")
    return delta


def comparison_identity(config):
    """Derive a display/duplicate identity from a receipt-checked configuration.

    This is not a receipt verifier: callers must first bind config bytes to the
    run/batch hashes. Never trust a supplied display name or import model code.
    """
    model = config.get("model")
    require(isinstance(model, str) and bool(model.strip()), "Missing model identity")
    recipe = config.get("ukt_training_recipe")
    if model != "ukt":
        require(recipe is None, "UKT training recipe is only valid for model ukt")
        return {"model": model, "ukt_training_recipe": None, "comparison_label": model}
    require(isinstance(recipe, str) and recipe in UKT_RECIPES, "Missing or unsupported UKT training recipe")
    settings = config.get("model_config")
    require(isinstance(settings, dict), "UKT comparison requires resolved model_config")
    flags = {}
    for key in ("use_CL", "use_uncertainty_aug"):
        value = settings.get(key)
        require(type(value) is bool or (type(value) is int and value in (0, 1)),
                f"UKT {key} must be explicit boolean or integer 0/1")
        flags[key] = bool(value)
    expected = ("ukt-no-cl" if not flags["use_CL"] else
                UKT_AUTHOR_AUGMENTATION if flags["use_uncertainty_aug"] else "ukt-cl-same-responses")
    require(recipe == expected, "UKT training recipe conflicts with resolved model switches")
    return {"model": model, "ukt_training_recipe": recipe,
            "comparison_label": f"{model} / {recipe}"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


class Receipts:
    """Track only explicitly selected artifact bytes, with a final race check."""

    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        receipt = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        require(path not in self.files or self.files[path] == receipt, "Input changed during analysis")
        self.files[path] = receipt
        return content

    def json(self, path):
        def invalid_constant(value):
            raise ValueError("Nonfinite JSON constant")
        result = json.loads(self.read(path), object_pairs_hook=_unique_object,
                            parse_constant=invalid_constant)
        require(isinstance(result, dict), "Expected a JSON object")
        return result

    def verify(self):
        for path, receipt in self.files.items():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            require(hasher.hexdigest() == receipt["sha256"], "Input changed during analysis")


def _audit(audit):
    require(audit.get("protocol") == PROTOCOL and audit.get("split") == "valid",
            "Only full v2 valid predictions are accepted; train/test/v1 are forbidden")
    require(audit.get("kc_budget") == 200, "Expected the KC200 protocol")
    require(audit.get("selection_limited") is False and audit.get("selected_for_debug") is False
            and "max_targets" in audit and audit["max_targets"] is None,
            "Sampled/debug validation is forbidden")
    n = integer(audit.get("selected_targets"), "selected_targets", 1)
    require(audit.get("expected_total") == n == audit.get("eligible_total"), "Missing full validation targets")
    require(audit.get("test_access") is False, "Test access is not sealed")
    require(audit.get("hash_scheme") == "ktlab-shared-event-compositional-sha256-v1"
            and audit.get("hash_scope") == "selected_targets_in_original_csv_and_event_order",
            "Unsupported target/context hash scheme")
    require(integer(audit.get("fold"), "fold") <= 4, "Invalid validation fold")
    require(isinstance(audit.get("dataset"), str) and bool(audit["dataset"]), "Missing dataset identity")
    for key in ("source_sha256", "target_sha256", "context_sha256"):
        digest(audit.get(key), key)
    return {key: audit[key] for key in IDENTITY}


def _matches_subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _matches_subset(v, actual[k])
                                                for k, v in expected.items())
    return type(expected) is type(actual) and expected == actual


def _read_run(path, batch, batch_sha, receipts):
    path = Path(path).resolve()
    require(path.name == "valid_predictions.csv", "Require the final run-root valid_predictions.csv")
    directory = path.parent
    sidecar_path = path.with_suffix(".csv.manifest.json")
    paths = {"prediction": path, "sidecar": sidecar_path,
             "metadata": directory / "metadata.json", "config": directory / "resolved_config.json",
             "metrics": directory / "valid_metrics.json"}
    sidecar = receipts.json(sidecar_path)
    metadata = receipts.json(paths["metadata"])
    config = receipts.json(paths["config"])
    metrics = receipts.json(paths["metrics"])
    run_id = metadata.get("run_id")
    require(run_id == directory.name, "Run ID does not match the final artifact directory")
    entry = next((x for x in batch["runs"] if x["run_id"] == run_id), None)
    require(entry is not None, "Run is absent from the specified batch manifest")
    require(metadata.get("status") == "completed", "Run is not completed")
    require(metadata.get("protocol") == config.get("protocol") == PROTOCOL, "Run protocol is not v2")
    require(metadata.get("test_accessed") is False and metadata.get("partial_debug") is False,
            "Test-accessed or debug run is forbidden")
    require(metadata.get("initial_validation") is True and config.get("initial_validation") is True,
            "Missing shared checkpoint-selection identity")
    require(metadata.get("checkpoint_selection_policy") == SELECTION, "Wrong checkpoint selection policy")
    require(metadata.get("batch_id") == config.get("batch_id") == batch["batch_id"]
            and metadata.get("approved_batch_sha256") == config.get("approved_batch_sha256") == batch_sha,
            "Missing or inconsistent recorded approved batch identity")
    require(_matches_subset(entry["config"], config), "Resolved configuration differs from the batch")
    require(config.get("debug_smoke") is False and config.get("max_validation_targets") is None
            and config.get("max_validation_batches") is None, "Sampled/debug configuration is forbidden")
    config_sha = canonical_hash({k: v for k, v in config.items()
                                 if k not in {"_batch_approved", "test_release_file"}})
    require(metadata.get("config_sha256") == config_sha, "Resolved configuration hash mismatch")
    comparison_identity(config)
    require(metadata.get("implementation_sha256") == batch.get("source_sha256"), "Implementation receipt mismatch")
    require(metadata.get("dataset_manifest_sha256") == entry.get("dataset_manifest_sha256"),
            "Dataset manifest receipt mismatch")
    for key in ("checkpoint_sha256", "config_sha256", "implementation_sha256",
                "model_source_sha256", "dataset_manifest_sha256"):
        digest(metadata.get(key), key)
    steps = integer(metadata.get("optimizer_steps"), "optimizer_steps", 1)
    require(steps <= integer(config.get("max_train_steps"), "max_train_steps", 1), "Training step budget exceeded")
    epoch = integer(metadata.get("best_epoch"), "best_epoch")
    require(epoch <= integer(config.get("epochs"), "epochs", 1), "Best epoch exceeds configured training")
    completed = metadata.get("best_epoch_completed")
    require(type(completed) is bool and not (epoch == 0 and completed), "Invalid best epoch completion state")
    stage = "initial_untrained" if epoch == 0 else ("after_training_epoch" if completed else "after_partial_training_epoch")
    require(sidecar.get("schema_version") == 1 and sidecar.get("metric_scope") == "development_only_not_sota_or_final_test",
            "Unsupported prediction sidecar")
    audit = sidecar.get("dataset_audit", {})
    identity = _audit(audit)
    require(_audit(metadata.get("valid_event_audit", {})) == identity, "Training validation selection differs")
    require(config.get("dataset") == identity["dataset"] and config.get("fold") == identity["fold"],
            "Configuration dataset/fold mismatch")
    require(integer(metadata.get("seed"), "metadata seed") == integer(config.get("seed"), "training seed"),
            "Training seed receipt mismatch")
    planned = entry.get("shared_validation", {})
    for key in ("protocol", "selected_targets", "source_sha256", "target_sha256", "context_sha256"):
        require(planned.get(key) == audit[key], "Batch validation identity mismatch")
    for kind in ("source", "target", "context"):
        require(metadata.get(f"validation_{kind}_sha256") == audit[f"{kind}_sha256"],
                "Checkpoint-selection validation identity mismatch")
    require(metadata.get("valid_file_sha256") == audit["source_sha256"], "Validation source receipt mismatch")
    result = sidecar.get("result", {})
    for saved in (result, metrics):
        require(saved.get("protocol") == PROTOCOL and saved.get("split") == "valid"
                and saved.get("selection_limited") is False and saved.get("test_accessed") is False
                and saved.get("quality_metrics_computed") is True, "Invalid saved validation scope")
        require(saved.get("n") == audit["selected_targets"] and saved.get("model") == config.get("model"),
                "Model or target-count receipt mismatch")
        for key in ("source_sha256", "target_sha256", "context_sha256"):
            require(saved.get(key) == audit[key], "Saved metrics selection mismatch")
        for key in ("implementation_sha256", "model_source_sha256"):
            require(saved.get(key) == metadata[key], "Saved metrics implementation mismatch")
    require(result.get("model_training_and_checkpoint_selection_not_verified") is True,
            "Unexpected standalone sidecar provenance schema")
    require(all(k in metrics and metrics[k] == v for k, v in result.items()),
            "Final metrics disagree with the prediction sidecar")
    require(metrics.get("partial_debug") is False and metrics.get("checkpoint_selection_policy") == SELECTION,
            "Missing final training evaluator receipt")
    content = receipts.read(path)
    require(sidecar.get("prediction_sha256") == receipts.files[path]["sha256"], "Prediction byte hash mismatch")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    require(reader.fieldnames == FIELDS, "Unsupported or duplicate prediction CSV columns")
    rows = {}
    for row in reader:
        require(None not in row and all(value is not None for value in row.values()), "Malformed CSV row")
        key = row["uid"], row["event_id"]
        require(all(part.strip() for part in key) and key not in rows, "Empty or duplicate event identity")
        require(row["label"] in {"0", "1"}, "Nonbinary saved label")
        require(re.fullmatch(r"[1-9][0-9]*", row["kc_count"]) is not None, "Invalid KC count")
        kc_count = int(row["kc_count"])
        require(kc_count <= 199, "Scored query leaves no room for required history")
        question = row["question_id"]
        require(question == "" or re.fullmatch(r"0|[1-9][0-9]*", question) is not None, "Invalid question ID")
        probability = float(row["probability"])
        require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid saved probability")
        identity_row = (question, int(row["label"]), kc_count,
                        digest(row["target_sha256"], "target"), digest(row["context_sha256"], "context"))
        rows[key] = (identity_row, probability)
    require(len(rows) == audit["selected_targets"], "Missing full validation targets")
    return {"run_id": run_id, "config": config, "metadata": metadata, "metrics": metrics,
            "identity": identity, "rows": rows, "stage": stage,
            "input_receipts": {name: receipts.files[p.resolve()] for name, p in paths.items()}}


class RankedAUC:
    """Cached tie groups, weighted Mann-Whitney AUC; no repeated sorting."""

    def __init__(self, labels, probabilities):
        self.order = np.argsort(probabilities, kind="stable")
        self.starts = np.r_[0, np.flatnonzero(np.diff(probabilities[self.order])) + 1]
        self.labels = labels[self.order]

    def __call__(self, weights):
        weights = weights[self.order]
        positives = np.add.reduceat(weights * self.labels, self.starts)
        negatives = np.add.reduceat(weights * (1 - self.labels), self.starts)
        denominator = positives.sum() * negatives.sum()
        if not denominator:
            return None
        return float(np.sum(positives * (np.cumsum(negatives) - .5 * negatives)) / denominator)


def losses(labels, probabilities):
    p = np.clip(probabilities, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    return -(labels * np.log(p) + (1 - labels) * np.log1p(-p)), (probabilities - labels) ** 2


def metrics_for(labels, probabilities):
    if not len(labels):
        return {"n": 0, "auc": None, "accuracy": None, "nll": None, "brier": None,
                "ece_15bins": None, "auc_unavailable_reason": "empty_stratum"}
    nll, brier = losses(labels, probabilities)
    bins = np.minimum((probabilities * 15).astype(np.int64), 14)
    # This is the event-weighted equivalent of sum(count/n * |mean(p)-mean(y)|).
    ece = np.abs(np.bincount(bins, weights=probabilities - labels, minlength=15)).sum() / len(labels)
    auc = RankedAUC(labels, probabilities)(np.ones(len(labels)))
    return {"n": len(labels), "auc": auc, "accuracy": float(np.mean((probabilities >= .5) == labels)),
            "nll": float(nll.mean()), "brier": float(brier.mean()), "ece_15bins": float(ece),
            "auc_unavailable_reason": "single_class" if auc is None else None}


def paired_bootstrap(labels, scores, student_index, reference, *, replicates=500, seed=42):
    integer(replicates, "bootstrap_replicates", 2)
    require(replicates <= 2000, "At most 2000 bootstrap draws are supported")
    integer(seed, "seed")
    require(len(labels) * len(scores) * replicates <= MAX_BOOTSTRAP_EVENT_VISITS,
            "Bootstrap work limit exceeded; request fewer draws explicitly (no automatic truncation)")
    n_students = int(student_index.max()) + 1
    require(n_students >= 2, "At least two students are required for paired cluster bootstrap")
    counts = np.bincount(student_index, minlength=n_students)
    auc_scorers = {name: RankedAUC(labels, score) for name, score in scores.items()}
    student_losses = {name: [np.bincount(student_index, weights=x, minlength=n_students)
                            for x in losses(labels, score)] for name, score in scores.items()}
    samples = {name: {key: [] for key in ("auc", "nll", "brier")} for name in scores if name != reference}
    rng = np.random.default_rng(seed)
    undefined_auc = 0
    for _ in range(replicates):
        multiplicity = np.bincount(rng.integers(n_students, size=n_students), minlength=n_students)
        weights = multiplicity[student_index].astype(np.float64)
        denominator = float(np.dot(multiplicity, counts))
        auc = {name: scorer(weights) for name, scorer in auc_scorers.items()}
        if auc[reference] is None:
            undefined_auc += 1
        for name, deltas in samples.items():
            if auc[reference] is not None:
                deltas["auc"].append(auc[name] - auc[reference])
            for index, metric in enumerate(("nll", "brier")):
                deltas[metric].append(float(np.dot(multiplicity, student_losses[name][index]
                                                   - student_losses[reference][index]) / denominator))
    output = {}
    points = {name: metrics_for(labels, score) for name, score in scores.items()}
    for name, deltas in samples.items():
        output[name] = {}
        for metric, values in deltas.items():
            output[name][metric] = {
                "delta": (points[name][metric] - points[reference][metric]
                          if points[reference][metric] is not None else None),
                "percentile_95_ci": np.quantile(values, [.025, .975], method="linear").tolist() if len(values) >= 2 else None,
                "usable_draws": len(values), "undefined_draws": replicates - len(values),
                "ci_unavailable_reason": "fewer_than_two_usable_draws" if len(values) < 2 else None}
    return {"comparisons": output, "draws": replicates, "seed": seed, "students": n_students,
            "resampling_unit": "student", "metric_weighting": "events_with_resampled_student_multiplicity",
            "shared_draws_across_models": True, "single_class_auc_draws": undefined_auc,
            "single_class_policy": "no_redraw; AUC omitted, NLL/Brier retained; AUC CI conditional on two classes",
            "delta_direction": "candidate_minus_reference; positive AUC and negative NLL/Brier favor candidate",
            "confidence": .95, "multiplicity_correction": "none", "p_values_computed": False,
            "low_draw_count_warning": replicates < 200,
            "scope": "exploratory_fixed_prediction_validation_student_sampling; validation also selected checkpoints; not confirmatory significance or retraining/multiseed stability"}


def _read_batches(batch_path, receipts):
    """Index explicitly supplied real manifests without synthesizing a batch."""
    paths = [batch_path] if isinstance(batch_path, (str, PathLike)) else batch_path
    require(isinstance(paths, (list, tuple)) and bool(paths),
            "Provide one batch manifest path or a nonempty list/tuple of paths")
    require(all(isinstance(path, (str, PathLike)) for path in paths), "Invalid batch manifest path")
    paths = [Path(path).resolve() for path in paths]
    require(len(set(paths)) == len(paths), "Duplicate batch manifest paths")
    batches, by_run, batch_ids = [], {}, set()
    for path in paths:
        batch = receipts.json(path)
        require(batch.get("test_access") is False and isinstance(batch.get("runs"), list)
                and bool(batch["runs"]), "Invalid sealed batch manifest")
        batch_id = batch.get("batch_id")
        require(isinstance(batch_id, str) and bool(batch_id.strip()) and batch_id not in batch_ids,
                "Duplicate or missing batch ID across manifests")
        batch_ids.add(batch_id)
        digest(batch.get("source_sha256"), "batch source")
        record = {"batch": batch, "canonical_batch_sha256": canonical_hash(batch),
                  "batch_file": receipts.files[path], "selected_run_ids": []}
        for entry in batch["runs"]:
            require(isinstance(entry, dict), "Invalid batch run declaration")
            run_id = entry.get("run_id")
            require(isinstance(run_id, str) and bool(run_id.strip()) and run_id not in by_run,
                    "Duplicate/missing batch run IDs across manifests")
            require(isinstance(entry.get("config"), dict) and bool(entry["config"]),
                    "Missing or ambiguous batch run configuration")
            by_run[run_id] = record
        batches.append(record)
    return batches, by_run


def analyze(paths, batch_path, reference, *, replicates=500, seed=42):
    """Analyze saved predictions against one manifest path or explicit paths."""
    require(len(paths) >= 2, "At least two model prediction files are required")
    receipts = Receipts()
    batches, by_run = _read_batches(batch_path, receipts)
    runs = []
    for path in paths:
        # Directory is only a lookup hint. _read_run checks its consistency with
        # metadata and config, including this original batch's canonical hash.
        record = by_run.get(Path(path).resolve().parent.name)
        require(record is not None, "Run is absent from the specified batch manifests")
        run = _read_run(path, record["batch"], record["canonical_batch_sha256"], receipts)
        record["selected_run_ids"].append(run["run_id"])
        runs.append(run)
    require(len({r["run_id"] for r in runs}) == len(runs), "Duplicate input runs")
    comparisons = {r["run_id"]: comparison_identity(r["config"]) for r in runs}
    variants = {(identity["model"], identity["ukt_training_recipe"]) for identity in comparisons.values()}
    require(len(variants) >= 2, "At least two distinct models or UKT training recipes are required")
    require(len(variants) == len(runs), "Duplicate model/UKT training recipe runs are not supported")
    require(len({r["config"]["seed"] for r in runs}) == 1, "Models must use the same training seed")
    require(reference in {r["run_id"] for r in runs}, "Reference must identify an input run_id")
    baseline = next(r for r in runs if r["run_id"] == reference)
    keys = list(baseline["rows"])
    expected_rows = {key: baseline["rows"][key][0] for key in keys}
    for run in runs:
        require(run["identity"] == baseline["identity"], "Models have different target/context/source selections")
        require(run["metadata"]["dataset_manifest_sha256"] == baseline["metadata"]["dataset_manifest_sha256"],
                "Dataset manifest receipts differ")
        require({key: value[0] for key, value in run["rows"].items()} == expected_rows,
                "Per-event identities, labels, questions, KCs or contexts differ")
    # At least one input must preserve the frozen upstream sequence so its
    # ordered aggregate can be verified. Every other input is aligned by IDs.
    # Thus reference choice and differing CSV row order never alter pairing.
    sequence_verified = False
    for run in runs:
        matches = []
        for name, column, domain in (("target_sha256", 3, b"ktlab.shared.target-sequence.v1\0"),
                                    ("context_sha256", 4, b"ktlab.shared.context-sequence.v1\0")):
            hasher = hashlib.sha256(domain)
            for key in run["rows"]:
                hasher.update(bytes.fromhex(expected_rows[key][column]))
            matches.append(hasher.hexdigest() == baseline["identity"][name])
        if all(matches):
            sequence_verified = True
            break
    require(sequence_verified, "Aggregate target/context hash mismatch; no input retains the frozen order")
    keys = sorted(keys)
    labels = np.array([expected_rows[k][1] for k in keys], dtype=np.float64)
    kcs = np.array([expected_rows[k][2] for k in keys])
    students = sorted({key[0] for key in keys})
    student_lookup = {uid: i for i, uid in enumerate(students)}
    student_index = np.array([student_lookup[k[0]] for k in keys])
    scores = {run["run_id"]: np.array([run["rows"][key][1] for key in keys]) for run in runs}
    summaries = []
    for run in runs:
        p = scores[run["run_id"]]
        overall = metrics_for(labels, p)
        for key in ("auc", "nll", "brier", "accuracy", "ece_15bins"):
            value = run["metrics"].get(key)
            require(type(value) in (int, float) and math.isfinite(value) and overall[key] is not None
                    and math.isclose(value, overall[key], rel_tol=1e-10, abs_tol=1e-12),
                    f"Saved {key} does not match the bound CSV")
        auc_drift = selected_export_delta(run["metadata"].get("best_valid_auc"), overall["auc"])
        strata = {}
        for name, mask in (("single_kc", kcs == 1), ("multiple_kc", kcs > 1)):
            strata[name] = dict(metrics_for(labels[mask], p[mask]), students=int(np.unique(student_index[mask]).size))
        m = run["metadata"]
        summaries.append({"run_id": run["run_id"], **comparisons[run["run_id"]],
                          "seed": run["config"]["seed"], "fold": run["config"]["fold"],
                           "batch_id": m["batch_id"], "canonical_batch_sha256": m["approved_batch_sha256"],
                           "implementation_sha256": m["implementation_sha256"],
                           "selected_valid_auc_recorded": m["best_valid_auc"],
                           "export_minus_selected_auc": auc_drift,
                          "optimizer_steps_recorded": m["optimizer_steps"], "best_epoch": m["best_epoch"],
                          "best_epoch_completed": m["best_epoch_completed"], "selected_checkpoint_stage": run["stage"],
                          "checkpoint_sha256_recorded": m["checkpoint_sha256"], "config_sha256": m["config_sha256"],
                          "model_source_sha256": m["model_source_sha256"], "input_files": run["input_receipts"],
                          "overall": overall, "by_query_kc_count": strata})
    bootstrap = paired_bootstrap(labels, scores, student_index, reference, replicates=replicates, seed=seed)
    labels_hasher = hashlib.sha256(b"ktlab.saved-validation.labels.v1\0")
    for key in sorted(keys):
        labels_hasher.update(json.dumps([*key, expected_rows[key][1]], ensure_ascii=False,
                                       separators=(",", ":")).encode() + b"\n")
    receipts.verify()
    batch_summaries = [{"batch_id": record["batch"]["batch_id"],
                        "canonical_batch_sha256": record["canonical_batch_sha256"],
                        "implementation_sha256": record["batch"]["source_sha256"],
                        "batch_file": record["batch_file"], "selected_run_ids": record["selected_run_ids"]}
                       for record in batches]
    implementation_hashes = sorted({run["metadata"]["implementation_sha256"] for run in runs})
    single = batch_summaries[0] if len(batch_summaries) == 1 else None
    return {"schema_version": 1, "analysis": "saved_full_shared_validation_exploratory",
            "protocol": PROTOCOL, "split": "valid", "selection": baseline["identity"],
            "dataset_manifest_sha256_recorded": baseline["metadata"]["dataset_manifest_sha256"],
            "labels_sha256": labels_hasher.hexdigest(), "students": len(students), "events": len(keys),
            "reference_run_id": reference, "batch_id": single["batch_id"] if single else None,
            "canonical_batch_sha256": single["canonical_batch_sha256"] if single else None,
            "batches": batch_summaries,
            "cross_batch_comparison": {
                "multiple_batches_supplied": len(batches) > 1,
                "multiple_batches_selected": len({run["metadata"]["batch_id"] for run in runs}) > 1,
                "implementation_sha256_values": implementation_hashes,
                "implementation_sources_differ": len(implementation_hashes) > 1,
                "method_equivalence_verified": False,
                "manual_method_review_required": len(implementation_hashes) > 1,
                "explanation": "Each run is bound to its own real batch; matching targets and protocol do not prove method equivalence. Review source/configuration changes before interpreting cross-implementation differences."},
            "training_seed": baseline["config"]["seed"],
            "selected_export_auc_comparison": {
                "absolute_tolerance": SELECTED_EXPORT_ABS_TOL, "relative_tolerance": 0.0,
                "delta_definition": "final_csv_auc_minus_recorded_selection_auc",
                "scope": "Separate selection/export forwards only; same-CSV metric and copied-receipt checks remain strict.",
                "predeclared_before_B003_terminal_results": True},
            "batch_file": single["batch_file"] if single else None, "runs": summaries, "bootstrap": bootstrap,
            "ece": {"bins": 15, "binning": "equal_width_[0,1]; left_closed_right_open_except_last", "threshold_accuracy": .5},
            "receipt_scope": {"saved_receipts_consistent": True, "weights_read_or_verified": False,
                              "direct_checkpoint_to_prediction_hash_binding_available": False,
                              "training_identity_fully_proven": False, "approval_record_read_or_authenticated": False,
                              "explanation": "Final run directory, recorded batch/config/checkpoint/selection identities and saved metrics agree; unsigned receipts are not independent proof of training or of which weight bytes produced the CSV."},
            "unavailable_strata": {"history_length": "not present per event in prediction CSV",
                                   "training_question_frequency": "not present; validation frequency must not substitute for training frequency"},
            "limitations": ["Same validation data selected checkpoints; bootstrap intervals are exploratory, not confirmatory significance.",
                            "No multiplicity correction; no p-values; no multiseed/retraining uncertainty or SOTA conclusion.",
                            "Per-KC strata are descriptive; intervals cover overall paired event-weighted differences only."],
            "new_optimizer_steps": 0, "model_loads": 0, "model_forward_calls": 0,
            "raw_dataset_accessed": False, "test_accessed": False, "automatic_publication": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--batch", type=Path, action="append", required=True,
                        help="Public batch manifest, not private approval; repeat for multiple real batches")
    parser.add_argument("--reference", required=True, help="Exact run_id of the reference among the inputs")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Output already exists; choose a new aggregate report")
        report = analyze(args.predictions, args.batch, args.reference,
                         replicates=args.bootstrap_replicates, seed=args.seed)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"runs": len(report["runs"]), "events": report["events"],
                          "bootstrap_draws": report["bootstrap"]["draws"], "test_accessed": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as exc:
        print(f"Analysis refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
