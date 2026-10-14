"""Independent, full-precision question metrics and auditable predictions."""
from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .adapters import forward_model, move_batch


def binary_metrics(labels, probabilities) -> dict:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0 or p.shape != y.shape:
        raise ValueError("Empty or misaligned predictions")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0,1]")
    clipped = np.clip(p, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    # Fixed equal-width bins; do not choose the number of bins from test results.
    bin_ids = np.minimum((p * 15).astype(np.int64), 14)
    ece = sum(float(np.mean(bin_ids == index)) * abs(float(np.mean(p[bin_ids == index])) - float(np.mean(y[bin_ids == index])))
              for index in range(15) if np.any(bin_ids == index))
    return {"n": int(y.size), "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
            "accuracy": float(np.mean((p >= .5) == y)), "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "ece_15bins": float(ece)}


@torch.inference_mode()
def evaluate_loader(model, loader, device, prediction_path: str | Path | None = None,
                    *, max_batches: int | None = None, deadline: float | None = None) -> dict:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    groups = {}
    concept_count = 0
    visited_batches = 0
    for cpu_batch in loader:
        if max_batches is not None and visited_batches >= max_batches:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Run wall-clock budget exhausted during validation")
        visited_batches += 1
        batch = move_batch(cpu_batch, device)
        predicted, _ = forward_model(model, batch)
        predicted = predicted.detach().cpu().double()
        selected = cpu_batch["smasks"]
        for b, t in selected.nonzero(as_tuple=False).tolist():
            uid = str(cpu_batch["uid"][b])
            event = int(cpu_batch["event_ids"][b, t])
            key = (uid, event)
            label = int(cpu_batch["shft_rseqs"][b, t])
            probability = float(predicted[b, t])
            question = int(cpu_batch["shft_qseqs"][b, t])
            group = groups.setdefault(key, {"uid": uid, "event_id": event, "question_id": question,
                                            "label": label, "probabilities": [], "expected": 0})
            if group["label"] != label or group["question_id"] != question:
                raise ValueError(f"Inconsistent label/question for event {key}")
            group["probabilities"].append(probability)
            group["expected"] = max(group["expected"], int(cpu_batch["rests"][b, t]) + 1)
            concept_count += 1
    rows = []
    dropped_partial_groups = 0
    for group in groups.values():
        values = group.pop("probabilities")
        if len(values) != group.pop("expected"):
            if max_batches is not None:
                dropped_partial_groups += 1
                continue
            raise ValueError("Incomplete or duplicated question-level KC group; refusing partial fusion")
        group.update(probability=float(np.mean(values, dtype=np.float64)), kc_count=len(values))
        rows.append(group)
    result = binary_metrics([r["label"] for r in rows], [r["probability"] for r in rows])
    result.update(concept_predictions=concept_count, aggregation="late_mean_full_precision",
                  prediction_unit="question_event", evaluated_batches=visited_batches,
                  partial_debug=max_batches is not None, dropped_partial_groups=dropped_partial_groups)
    if prediction_path is not None:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["uid", "event_id", "question_id", "label", "probability", "kc_count"])
            writer.writeheader()
            writer.writerows(rows)
    return result


def metrics_from_predictions(path: str | Path) -> dict:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return binary_metrics([int(r["label"]) for r in rows], [float(r["probability"]) for r in rows])
