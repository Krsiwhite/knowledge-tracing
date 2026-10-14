"""Evaluation on a fixed common event/context selection.

This module has no training or checkpoint-selection entry point. In particular,
it cannot promote an existing v1/reference checkpoint to a v2 benchmark result.
Final-test use additionally requires final_test.py's active release capability.
"""
from __future__ import annotations

import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from .evaluation import binary_metrics
from .provenance import sha256_file, source_fingerprint, utc_now, write_json
from .shared_events import FINAL_PROTOCOL

PROTOCOL = "ktlab-v2-shared-event-context-kc200-dev"
FIELDS = ["uid", "event_id", "question_id", "label", "probability", "kc_count", "target_sha256", "context_sha256"]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def evaluate_shared(model, dataset, device, prediction_path=None, *, batch_size=32,
                    max_rows_per_forward=32, compute_metrics=True, deadline=None, final_access=None):
    """Evaluate exactly the dataset's frozen selection, once.

    Debug limits belong to SharedEventDataset(max_targets=...), so the reported
    hashes describe the actual selected targets. This function does not silently
    take a different prefix. Use compute_metrics=False for random-model checks.
    """
    from .shared_adapters import predict_shared
    from .shared_events import context_digests

    _positive_integer(batch_size, "batch_size")
    _positive_integer(max_rows_per_forward, "max_rows_per_forward")
    audit = dataset.audit
    final = final_access is not None
    expected_protocol = FINAL_PROTOCOL if final else PROTOCOL
    if final:
        from .final_test import FinalTestAccess
        if not isinstance(final_access, FinalTestAccess):
            raise PermissionError("Final evaluation requires a registered test attempt")
        final_access.check_dataset(dataset)
        if not compute_metrics:
            raise ValueError("Released final evaluation must compute its frozen metrics")
    if dataset.protocol != expected_protocol or audit.get("protocol") != expected_protocol:
        raise ValueError("Shared evaluation requires the declared common-event protocol")
    if audit.get("split") not in ({"test"} if final else {"train", "valid"}):
        raise PermissionError("Shared evaluation is development-only; test remains sealed")
    if len(dataset) == 0 or audit.get("selected_targets") != len(dataset):
        raise ValueError("Empty or inconsistent frozen target selection")
    for key in ("source_sha256", "target_sha256", "context_sha256"):
        if not audit.get(key):
            raise ValueError(f"Missing shared dataset identity: {key}")
    source_path = Path(audit["source_path"])
    if source_path.resolve() != dataset.path.resolve() or sha256_file(source_path) != audit["source_sha256"]:
        raise ValueError("Shared development source changed after selection")
    target_hasher = hashlib.sha256(b"ktlab.shared.target-sequence.v1\0")
    context_hasher = hashlib.sha256(b"ktlab.shared.context-sequence.v1\0")
    path = Path(prediction_path) if prediction_path is not None else None
    sidecar = path.with_suffix(path.suffix + ".manifest.json") if path else None
    pending = path.with_suffix(path.suffix + ".tmp") if path else None
    if path and (path.exists() or sidecar.exists() or pending.exists()):
        raise FileExistsError("Prediction artifacts already exist; choose a fresh evaluation path")
    started = time.perf_counter()
    seen, labels, probabilities = set(), [], []
    stream = None
    try:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = pending.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
        for start in range(0, len(dataset), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Shared development evaluation deadline reached")
            contexts = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            for context in contexts:
                digests = context_digests(context)
                if any(context[key] != digests[key] for key in ("target_sha256", "context_sha256")):
                    raise ValueError("A target/history context changed after dataset selection")
                target_hasher.update(bytes.fromhex(digests["target_sha256"]))
                context_hasher.update(bytes.fromhex(digests["context_sha256"]))
            output = np.asarray(predict_shared(model, contexts, device, max_rows_per_forward=max_rows_per_forward), dtype=np.float64)
            if output.shape != (len(contexts),) or not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
                raise ValueError("Shared adapter returned invalid or misaligned probabilities")
            for context, probability in zip(contexts, output):
                uid, event_id = str(context["uid"]), str(context["event_id"])
                key = uid, event_id
                if key in seen:
                    raise ValueError(f"Duplicate shared target event {key}")
                seen.add(key)
                label = context["label"]
                if label not in (0, 1):
                    raise ValueError("Shared target labels must be binary")
                labels.append(label)
                probabilities.append(float(probability))
                row = {"uid": uid, "event_id": event_id, "question_id": context["query"]["question"],
                       "label": label, "probability": float(probability), "kc_count": len(context["query"]["concepts"]),
                       "target_sha256": context["target_sha256"], "context_sha256": context["context_sha256"]}
                if stream:
                    writer.writerow(row)
        if target_hasher.hexdigest() != audit["target_sha256"] or context_hasher.hexdigest() != audit["context_sha256"]:
            raise ValueError("Scored target/history sequence differs from the frozen selection")
        if sha256_file(source_path) != audit["source_sha256"]:
            raise ValueError("Shared development source changed during evaluation")
        result = {"n": len(probabilities), "protocol": expected_protocol, "split": audit["split"],
                  "target_sha256": audit["target_sha256"], "context_sha256": audit["context_sha256"],
                  "source_sha256": audit["source_sha256"], "quality_metrics_computed": bool(compute_metrics),
                  "selection_limited": bool(audit.get("selection_limited")),
                  "evaluation_seconds": time.perf_counter() - started, "test_accessed": final,
                  "model": model.model_name, "implementation_sha256": source_fingerprint(),
                  "model_source_sha256": model.ktlab_source_sha256,
                  "model_training_and_checkpoint_selection_not_verified": True}
        if compute_metrics:
            result.update(binary_metrics(labels, probabilities))
        if final:
            final_access.verify_receipts()
            final_access.verify_data_receipts()
            result.update(final_access.output_identity())
        if path:
            stream.close()
            stream = None
            pending.replace(path)
            write_json(sidecar, {"schema_version": 1, "created_at": utc_now(), "dataset_audit": audit,
                                 "result": result, "prediction_sha256": sha256_file(path),
                                 "metric_scope": "released_final_test" if final else "development_only_not_sota_or_final_test"})
        return result
    finally:
        if stream:
            stream.close()
        if pending and pending.exists():
            pending.unlink()


def assert_shared_prediction_alignment(paths):
    """Require matching identities, allowed histories and labels across files.

    This checks data comparability, not hyperparameter-search fairness, selected
    checkpoints, statistical significance or permission for a SOTA claim.
    """
    if len(paths) < 2:
        raise ValueError("At least two prediction files are required")
    reference = None
    reference_audit = None
    for filename in paths:
        path = Path(filename)
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("prediction_sha256") != sha256_file(path):
            raise ValueError("Prediction file no longer matches its manifest")
        audit = metadata["dataset_audit"]
        if audit.get("protocol") != PROTOCOL or audit.get("split") not in {"train", "valid"}:
            raise ValueError("Only this shared development protocol can be compared")
        keys = ("protocol", "dataset", "split", "fold", "source_sha256", "target_sha256", "context_sha256", "selected_targets")
        identity = {key: audit.get(key) for key in keys}
        if reference_audit is not None and identity != reference_audit:
            raise ValueError("Prediction selections or allowed histories differ")
        reference_audit = identity
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        labels = np.asarray([int(r["label"]) for r in rows])
        scores = np.asarray([float(r["probability"]) for r in rows])
        if not len(rows) or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Saved predictions contain invalid labels or probabilities")
        normalized = {(row["uid"], row["event_id"]): tuple(row[k] for k in
                      ("question_id", "label", "kc_count", "target_sha256", "context_sha256")) for row in rows}
        if len(normalized) != len(rows) or len(rows) != audit["selected_targets"]:
            raise ValueError("Duplicate/missing shared targets in saved predictions")
        if reference is not None and normalized != reference:
            raise ValueError("Per-event targets, histories or labels differ")
        reference = normalized
    return {"protocol": PROTOCOL, "files": len(paths), "matched_events": len(reference),
            "target_sha256": reference_audit["target_sha256"], "context_sha256": reference_audit["context_sha256"],
            "data_comparability_only": True}
