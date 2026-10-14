"""KTST reference development track; deliberately separate from ktlab v1.

Only train/validation is implemented. A corrected official test-window adapter
and a common target/history comparison contract remain required before test.
No optimizer is constructed without the CLI's content-bound batch approval.
"""
from __future__ import annotations

import csv
import json
import math
from numbers import Integral
from pathlib import Path
import platform
import time

import torch
from torch.utils.data import DataLoader

from .adapters import move_batch, resolve_device
from .evaluation import binary_metrics
from .provenance import config_fingerprint, git_revision, source_fingerprint, utc_now

PROTOCOL = "ktlab-ktst-reference-dev-v1"


def _components():
    # Lazy imports keep unrelated pyKT runs independent of this optional track.
    from .ktst_data import KTSTSequenceDataset, collate_ktst
    from .ktst_model import build_ktst_model, forward_ktst, optimizer_parameter_groups
    return KTSTSequenceDataset, collate_ktst, build_ktst_model, forward_ktst, optimizer_parameter_groups


def _loader(dataset, config, *, train=False):
    _, collate, _, _, _ = _components()
    batch_size = config.get("batch_size", 128) if train else config.get("validation_batch_size", 64)
    return DataLoader(dataset, batch_size=int(batch_size),
                      shuffle=train, collate_fn=collate,
                      num_workers=int(config.get("num_workers", 0)),
                      generator=torch.Generator().manual_seed(int(config.get("seed", 42))),
                      pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def evaluate_ktst_loader(model, loader, device, prediction_path=None, *, max_batches=None, deadline=None):
    """One probability per merged event, never KC late-mean fusion."""
    if max_batches is not None and int(max_batches) < 1:
        raise ValueError("max_batches must be positive")
    forward = _components()[3]
    model.eval()
    rows, seen, batches = [], set(), 0
    for cpu_batch in loader:
        if max_batches is not None and batches >= int(max_batches):
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Run wall-clock budget exhausted during KTST validation")
        batches += 1
        if not bool(cpu_batch["score_mask"].any()):
            continue
        batch = move_batch(cpu_batch, device)
        probabilities, _ = forward(model, batch)
        probabilities = probabilities.detach().cpu().double()
        if probabilities.shape != cpu_batch["score_mask"].shape:
            raise ValueError("KTST probability/event mask shape mismatch")
        for b, t in cpu_batch["score_mask"].nonzero(as_tuple=False).tolist():
            uid = str(cpu_batch["uid"][b])
            raw_event = cpu_batch["event_ids"][b][t]
            event_id = str(raw_event.item() if torch.is_tensor(raw_event) else raw_event)
            key = (uid, event_id)
            if key in seen:
                raise ValueError(f"Duplicate KTST target event {key}; refusing implicit aggregation")
            seen.add(key)
            rows.append({"uid": uid, "event_id": event_id,
                         "question_id": int(cpu_batch["question_ids"][b, t]),
                         "label": int(cpu_batch["labels"][b, t]),
                         "probability": float(probabilities[b, t]),
                         "kc_count": int(cpu_batch["kc_counts"][b, t])})
    result = binary_metrics([r["label"] for r in rows], [r["probability"] for r in rows])
    result.update(protocol=PROTOCOL, prediction_unit="question_event", aggregation="native_set_event",
                  evaluated_batches=batches, partial_debug=max_batches is not None)
    if prediction_path is not None:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["uid", "event_id", "question_id", "label", "probability", "kc_count"])
            writer.writeheader()
            writer.writerows(rows)
    return result


def _check_config(config):
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"KTST requires explicit protocol={PROTOCOL}; scores cannot silently join ktlab v1")
    if config.get("model") not in {"ktst_mean", "ktst_mhsa"}:
        raise ValueError("Unknown KTST model variant")
    if not config.get("shuffle", True) or not config.get("initial_validation", True):
        raise ValueError("KTST reference development requires shuffled training and initial validation")
    for key in ("learning_rate", "theta_learning_rate"):
        if key not in config or not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"An explicit positive {key} is required")
    for key, default, minimum in (("epochs", 200, 1), ("patience", 10, 1), ("batch_size", 128, 1),
                                  ("validation_batch_size", 64, 1), ("seed", 42, 0), ("fold", 0, 0),
                                  ("num_workers", 0, 0)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("max_train_steps", "max_validation_batches"):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("grad_clip", "max_wall_seconds"):
        if config.get(key) is not None and (not math.isfinite(float(config[key])) or float(config[key]) <= 0):
            raise ValueError(f"{key} must be positive")
    if config.get("max_validation_batches") is not None and not config.get("debug_smoke", False):
        raise ValueError("Partial validation requires debug_smoke=true")
    if float(config.get("early_stop_min_delta", 0.0)) != 0:
        raise ValueError("KTST reference early stopping uses every strict validation AUC improvement")


def _save_checkpoint(model, path, auc, epoch, manifest_sha256, config=None):
    if not math.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError("A finite validation AUC is required")
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "model_config": model.ktlab_model_config,
                "epoch": epoch, "valid_auc": auc, "protocol": PROTOCOL,
                "dataset_manifest_sha256": manifest_sha256,
                "config_sha256": config_fingerprint(config) if config is not None else None,
                "implementation_sha256": source_fingerprint(),
                "model_source_sha256": model.ktlab_source_sha256}, temporary)
    temporary.replace(path)


def train_ktst(config: dict, run_dir: Path) -> dict:
    if config.get("_batch_approved") is not True:
        raise PermissionError("KTST training requires explicit per-batch approval via the ktlab CLI")
    _check_config(config)
    from .engine import _json, _seed, _sha, read_manifest
    dataset_type, _, build_model, forward, parameter_groups = _components()
    run_dir = Path(run_dir)
    if (run_dir / "metadata.json").exists() or (run_dir / "best.pt").exists():
        raise FileExistsError("KTST run directory already contains artifacts; use a new run ID")
    config = {k: v for k, v in config.items() if k != "_batch_approved"}
    config.update(shuffle=True, initial_validation=True, grad_clip=float(config.get("grad_clip", 1.0)))
    config["dataset_manifest"] = str(Path(config["dataset_manifest"]).resolve())
    manifest = read_manifest(config)
    device = resolve_device(config.get("device", "auto"))
    _seed(int(config.get("seed", 42)))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config["model"], manifest["data_config"], config.get("model_config"), device)
    training = dataset_type(manifest, int(config.get("fold", 0)), split="train")
    validation = dataset_type(manifest, int(config.get("fold", 0)), split="valid")
    config["model_config"] = model.ktlab_model_config
    run_dir.mkdir(parents=True, exist_ok=True)
    _json(run_dir / "resolved_config.json", config)
    metadata = {"protocol": PROTOCOL, "run_id": run_dir.name, "status": "running", "test_accessed": False,
                "started_at": utc_now(), "git_commit": git_revision(), "implementation_sha256": source_fingerprint(),
                "config_sha256": config_fingerprint(config), "batch_id": config.get("batch_id"),
                "approved_batch_sha256": config.get("approved_batch_sha256"),
                "dataset_manifest_sha256": manifest["_manifest_sha256"],
                "model_source_sha256": model.ktlab_source_sha256, "device": str(device),
                "ktst_commit": model.ktlab_upstream_commit, "import_binding_only": model.ktlab_import_binding_only,
                "torch": torch.__version__, "python": platform.python_version(), "hip": torch.version.hip,
                "train_file_sha256": _sha(training.path), "valid_file_sha256": _sha(validation.path),
                "train_rows": len(training), "valid_rows": len(validation),
                "train_event_audit": training.audit, "valid_event_audit": validation.audit,
                "optimizer_steps": 0, "loss": "upstream_BCEWithLogits_excluding_segment_first_event",
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "seed": int(config.get("seed", 42)), "partial_debug": config.get("max_validation_batches") is not None,
                "not_comparable_with_protocol": "ktlab-v1-causal-valid-full-precision-repeat-kc-train"}
    _json(run_dir / "metadata.json", metadata)
    started, steps = time.monotonic(), 0
    deadline = started + float(config["max_wall_seconds"]) if config.get("max_wall_seconds") else None
    maximum = config.get("max_validation_batches")
    try:
        optimizer = torch.optim.Adam(parameter_groups(model, float(config["learning_rate"]),
                                                      float(config["theta_learning_rate"])),
                                     weight_decay=float(config.get("weight_decay", 0.0)))
        train_loader, valid_loader = _loader(training, config, train=True), _loader(validation, config)
        initial = evaluate_ktst_loader(model, valid_loader, device, max_batches=maximum, deadline=deadline)
        if initial["auc"] is None:
            raise ValueError("Both response classes are required for validation AUC")
        best, stale = initial["auc"], 0
        _save_checkpoint(model, run_dir / "best.pt", best, 0, manifest["_manifest_sha256"], config)
        metadata.update(best_epoch=0, best_valid_auc=best)
        _json(run_dir / "initial_valid_metrics.json", initial)
        stop_reason = "epochs_completed"
        for epoch in range(1, int(config.get("epochs", 200)) + 1):
            model.train()
            total_loss, batches = 0., 0
            for cpu_batch in train_loader:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("KTST wall-clock budget exhausted")
                if not bool(cpu_batch["score_mask"].any()):
                    continue
                batch = move_batch(cpu_batch, device)
                _, loss = forward(model, batch)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite or nonscalar upstream KTST loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                steps += 1
                batches += 1
                total_loss += float(loss.detach())
                if config.get("max_train_steps") is not None and steps >= int(config["max_train_steps"]):
                    break
            if batches == 0:
                raise ValueError("No KTST training targets selected")
            valid = evaluate_ktst_loader(model, valid_loader, device, max_batches=maximum, deadline=deadline)
            auc = valid["auc"]
            if auc is None:
                raise ValueError("Both response classes are required for validation AUC")
            if auc > best:
                best, stale = auc, 0
                _save_checkpoint(model, run_dir / "best.pt", best, epoch, manifest["_manifest_sha256"], config)
                metadata.update(best_epoch=epoch, best_valid_auc=best)
            else:
                stale += 1
            event = {"epoch": epoch, "optimizer_steps": steps, "train_loss": total_loss / batches,
                     "valid": valid, "elapsed_seconds": time.monotonic() - started}
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event, allow_nan=False), flush=True)
            metadata["optimizer_steps"] = steps
            _json(run_dir / "metadata.json", metadata)
            if stale >= int(config.get("patience", 10)):
                stop_reason = "early_stopping"
                break
            if config.get("max_train_steps") is not None and steps >= int(config["max_train_steps"]):
                stop_reason = "max_train_steps"
                break
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        valid = evaluate_ktst_loader(model, valid_loader, device, run_dir / "valid_predictions.csv",
                                     max_batches=maximum, deadline=deadline)
        _json(run_dir / "valid_metrics.json", valid)
        metadata.update(status="completed", stop_reason=stop_reason, checkpoint_sha256=_sha(run_dir / "best.pt"))
        return {"run_dir": str(run_dir), "metadata": metadata, "valid": valid}
    except TimeoutError as exc:
        metadata.update(status="budget_exhausted", error=str(exc), stop_reason="max_wall_seconds")
        return {"run_dir": str(run_dir), "metadata": metadata, "valid": None}
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.update(optimizer_steps=steps, elapsed_seconds=time.monotonic() - started,
                        finished_at=utc_now(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        _json(run_dir / "metadata.json", metadata)
        training.close()
        validation.close()


def evaluate_ktst_run(run_dir, *, split="valid"):
    if split != "valid":
        raise PermissionError("KTST test is sealed and its official corrected-window adapter is not implemented")
    from .engine import _json, read_manifest
    dataset_type, _, build_model, _, _ = _components()
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    _check_config(config)
    manifest = read_manifest(config)
    device = resolve_device(config.get("device", "auto"))
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    if checkpoint.get("protocol") != PROTOCOL or checkpoint["dataset_manifest_sha256"] != manifest["_manifest_sha256"]:
        raise ValueError("KTST checkpoint protocol or dataset manifest mismatch")
    if checkpoint.get("config_sha256") != config_fingerprint(config):
        raise ValueError("KTST checkpoint configuration mismatch")
    if checkpoint.get("implementation_sha256") != source_fingerprint():
        raise ValueError("KTST checkpoint implementation mismatch; use a declared migration instead of silent re-evaluation")
    model = build_model(config["model"], manifest["data_config"], checkpoint["model_config"], device)
    if checkpoint.get("model_source_sha256") != model.ktlab_source_sha256:
        raise ValueError("KTST checkpoint model source mismatch")
    model.load_state_dict(checkpoint["state_dict"])
    dataset = dataset_type(manifest, int(config.get("fold", 0)), split="valid")
    try:
        result = evaluate_ktst_loader(model, _loader(dataset, config), device, run_dir / "valid_predictions.csv",
                                      max_batches=config.get("max_validation_batches"))
        _json(run_dir / "valid_metrics.json", result)
        return result
    finally:
        dataset.close()
