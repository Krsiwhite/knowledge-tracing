"""Approval-gated training and sealed-test evaluation for a declared protocol.

The protocol deliberately differs from exact pyKT reproduction: checkpoint
selection uses causal question validation, saves the true best AUC, and uses
full-precision metrics. Upstream repeat-KC training semantics are retained.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapters import SequenceDataset, build_model, forward_model, move_batch, resolve_device
from .evaluation import evaluate_loader
from .provenance import config_fingerprint, git_revision, source_fingerprint, utc_now

PROTOCOL = "ktlab-v1-causal-valid-full-precision-repeat-kc-train"
SHARED_PROTOCOL = "ktlab-v2-shared-event-context-kc200-dev"
SHARED_SELECTION = "epoch0_and_each_validation_boundary_strict_best_shared_valid_auc"


def save_best_checkpoint(model, path: Path, auc: float, best: float, epoch: int, manifest_sha256: str,
                         *, protocol: str = PROTOCOL, identity: dict | None = None,
                         epoch_completed: bool | None = None) -> bool:
    """Persist every strict improvement, independently of early-stop min_delta."""
    if not math.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError("Checkpoint selection requires finite validation AUC in [0,1]")
    if auc <= best:
        return False
    temporary = path.with_suffix(".tmp")
    if identity and set(identity) & {"state_dict", "model_config", "epoch", "valid_auc", "protocol", "dataset_manifest_sha256",
                                     "checkpoint_stage", "epoch_completed"}:
        raise ValueError("Checkpoint identity cannot override payload/protocol fields")
    shared_fields = {}
    if protocol == SHARED_PROTOCOL:
        if epoch_completed is None:
            epoch_completed = epoch > 0
        if type(epoch_completed) is not bool or (epoch == 0 and epoch_completed):
            raise ValueError("Shared checkpoint requires an explicit valid epoch completion state")
        shared_fields = {"epoch_completed": epoch_completed,
                         "checkpoint_stage": "initial_untrained" if epoch == 0 else
                         ("after_training_epoch" if epoch_completed else "after_partial_training_epoch")}
    torch.save({"state_dict": model.state_dict(), "model_config": model.ktlab_model_config,
                "epoch": epoch, "valid_auc": auc, "protocol": protocol,
                "dataset_manifest_sha256": manifest_sha256, **(identity or {}),
                **shared_fields}, temporary)
    temporary.replace(path)
    return True


def _json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(config: dict) -> dict:
    path = Path(config["dataset_manifest"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") not in {None, manifest["dataset"]}:
        raise ValueError("Configured dataset does not match prepared manifest")
    manifest["_manifest_sha256"] = _sha(path)
    manifest["_manifest_path"] = str(path)
    data_dir = Path(manifest["data_dir"])
    manifest["data_dir"] = str(data_dir if data_dir.is_absolute() else (path.parent / data_dir).resolve())
    if not manifest.get("files"):
        raise ValueError("Prepared manifest must include file checksums")
    base = Path(manifest["data_dir"]).resolve()
    for name, expected in manifest["files"].items():
        artifact = (base / name).resolve()
        if not artifact.is_relative_to(base):
            raise ValueError("Manifest file escapes its data directory")
        if not artifact.is_file() or artifact.stat().st_size != expected["bytes"] or _sha(artifact) != expected["sha256"]:
            raise ValueError(f"Prepared data changed: {name}")
    return manifest


def dataset_for_split(manifest: dict, fold: int, split: str, *, model_name: str = "akt") -> SequenceDataset:
    data_config = manifest["data_config"]
    folds = set(map(int, data_config.get("folds", [0, 1, 2, 3, 4])))
    if fold not in folds:
        raise ValueError(f"Unknown validation fold {fold}")
    question_level = False
    if split == "train":
        filename, selected = data_config["train_valid_file"], folds - {fold}
    elif split == "valid":
        if int(data_config.get("num_q", 0)) > 0:
            filename = manifest.get("validation_question_files", {}).get(str(fold))
            if not filename:
                raise ValueError("Causal question-level validation file is required; run data prepare")
            question_level = True
        else:
            filename = data_config["train_valid_file"]
        selected = {fold}
    elif split == "test":
        question_level = "test_question_file" in data_config
        filename = data_config["test_question_file" if question_level else "test_file"]
        selected = {-1}
    else:
        raise ValueError("split must be train, valid, or test")
    path = Path(manifest["data_dir"]) / filename
    return SequenceDataset(path, selected, question_level=question_level, time_features=model_name == "fa_kt")


def _loader(dataset, config, *, train=False):
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    return DataLoader(dataset, batch_size=int(config.get("batch_size", 64)),
                      shuffle=bool(config.get("shuffle", False)) if train else False,
                      num_workers=int(config.get("num_workers", 0)), generator=generator,
                      pin_memory=torch.cuda.is_available())


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _shared_config(config: dict) -> dict:
    """Resolve only explicitly requested shared validation, never relabel v1."""
    if config.get("protocol") != SHARED_PROTOCOL:
        raise ValueError("Shared validation requires its explicit protocol")
    if config.get("model") not in {"akt", "simplekt", "mockt", "ukt", "ktst_mean", "ktst_mhsa"}:
        raise ValueError("This model has no shared-context adapter")
    if config.get("max_validation_batches") is not None:
        raise ValueError("Shared validation forbids max_validation_batches; limit constructed targets explicitly")
    if config.get("initial_validation", True) is not True:
        raise ValueError("Shared checkpoint selection requires epoch0 initial_validation")
    if float(config.get("early_stop_min_delta", 0.0)) != 0:
        raise ValueError("Shared checkpoint selection uses every strict AUC improvement")
    resolved = dict(config)
    resolved.setdefault("validation_batch_size", 32)
    resolved.setdefault("max_rows_per_forward", 32)
    resolved["initial_validation"] = True
    fold = resolved.get("fold", 0)
    if isinstance(fold, bool) or not isinstance(fold, Integral) or not 0 <= fold <= 4:
        raise ValueError("fold must be an integer in [0,4]")
    for key in ("validation_batch_size", "max_rows_per_forward", "max_validation_targets"):
        value = resolved.get(key)
        if (key != "max_validation_targets" or value is not None) and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if resolved.get("max_validation_targets") is not None and resolved.get("debug_smoke") is not True:
        raise ValueError("max_validation_targets requires debug_smoke=true")
    return resolved


def _shared_dataset(manifest: dict, config: dict):
    from .shared_events import SharedEventDataset
    return SharedEventDataset(manifest, config.get("fold", 0), split="valid",
                              max_targets=config.get("max_validation_targets"))


def _shared_identity(config: dict, model, dataset) -> dict:
    if (dataset.audit.get("protocol") != SHARED_PROTOCOL or dataset.audit.get("split") != "valid"
            or dataset.audit.get("fold") != config.get("fold", 0)):
        raise ValueError("Checkpoint selection requires the configured shared validation fold")
    return {"config_sha256": config_fingerprint(config), "implementation_sha256": source_fingerprint(),
            "model_source_sha256": model.ktlab_source_sha256,
            "validation_source_sha256": dataset.audit["source_sha256"],
            "validation_target_sha256": dataset.audit["target_sha256"],
            "validation_context_sha256": dataset.audit["context_sha256"],
            "checkpoint_selection_policy": SHARED_SELECTION}


def _shared_evaluate(model, dataset, device, config, identity, prediction_path=None, *, deadline=None):
    from .shared_evaluation import evaluate_shared
    current = _shared_identity(config, model, dataset)
    if current != identity:
        raise ValueError("Shared validation implementation/configuration/selection identity changed")
    result = evaluate_shared(model, dataset, device, prediction_path,
                             batch_size=config["validation_batch_size"],
                             max_rows_per_forward=config["max_rows_per_forward"], deadline=deadline)
    if result["implementation_sha256"] != identity["implementation_sha256"]:
        raise ValueError("Shared validation implementation changed during evaluation")
    result.update(partial_debug=config.get("max_validation_targets") is not None,
                  checkpoint_selection_policy=SHARED_SELECTION)
    return result


def _selected_shared_checkpoint(path, metadata, identity, manifest_sha256, model_config, *, device="cpu"):
    """Load exactly the weight artifact selected by the recorded validation."""
    if metadata.get("checkpoint_sha256") != _sha(path):
        raise ValueError("Shared checkpoint file changed since selection")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected = {"protocol": SHARED_PROTOCOL, "dataset_manifest_sha256": manifest_sha256, **identity}
    for key, value in expected.items():
        if checkpoint.get(key) != value or metadata.get(key) != value:
            raise ValueError(f"Shared checkpoint/metadata {key} mismatch")
    if checkpoint.get("model_config") != model_config:
        raise ValueError("Shared checkpoint model configuration mismatch")
    if (checkpoint.get("epoch") != metadata.get("best_epoch")
            or checkpoint.get("valid_auc") != metadata.get("best_valid_auc")):
        raise ValueError("Shared checkpoint differs from the recorded best validation candidate")
    completed = checkpoint.get("epoch_completed")
    if (type(completed) is not bool or metadata.get("best_epoch_completed") is not completed
            or (checkpoint["epoch"] == 0 and completed)):
        raise ValueError("Shared checkpoint epoch completion state mismatch")
    stage = "initial_untrained" if checkpoint["epoch"] == 0 else (
        "after_training_epoch" if completed else "after_partial_training_epoch")
    if checkpoint.get("checkpoint_stage") != stage:
        raise ValueError("Shared checkpoint stage mismatch")
    return checkpoint


def _evaluate_shared_run(run_dir: Path, config: dict, split: str) -> dict:
    """Re-evaluate only checkpoints selected under this same shared contract."""
    if split != "valid":
        raise PermissionError("Shared final test is sealed and not implemented")
    config = _shared_config(config)
    manifest = read_manifest(config)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != _sha(run_dir / "best.pt"):
        raise ValueError("Shared checkpoint file changed since selection")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    expected = {"protocol": SHARED_PROTOCOL, "dataset_manifest_sha256": manifest["_manifest_sha256"],
                "config_sha256": config_fingerprint(config), "implementation_sha256": source_fingerprint(),
                "checkpoint_selection_policy": SHARED_SELECTION}
    for key, value in expected.items():
        if checkpoint.get(key) != value or metadata.get(key) != value:
            raise ValueError(f"Shared checkpoint/metadata {key} mismatch; no silent protocol migration")
    if checkpoint.get("model_config") != config.get("model_config"):
        raise ValueError("Shared checkpoint model configuration mismatch")
    device = resolve_device(config.get("device", "auto"))
    if config["model"].startswith("ktst_"):
        from .ktst_model import build_ktst_model
        model = build_ktst_model(config["model"], manifest["data_config"], checkpoint["model_config"], device)
    else:
        model = build_model(config["model"], manifest["data_config"], checkpoint["model_config"], device)
    dataset = _shared_dataset(manifest, config)
    try:
        identity = _shared_identity(config, model, dataset)
        for key, value in identity.items():
            if checkpoint.get(key) != value or metadata.get(key) != value:
                raise ValueError(f"Shared checkpoint/metadata {key} mismatch")
        checkpoint = _selected_shared_checkpoint(run_dir / "best.pt", metadata, identity,
                                                manifest["_manifest_sha256"], config["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        path = run_dir / "valid_predictions.csv"
        if path.exists() or path.with_suffix(".csv.manifest.json").exists():
            path = run_dir / "evaluations" / f"valid-{time.time_ns()}" / "predictions.csv"
        result = _shared_evaluate(model, dataset, device, config, identity, path)
        result["prediction_file"] = str(path)
        _json(path.with_suffix(".metrics.json"), result)
        return result
    finally:
        dataset.close()


def train(config: dict, run_dir: Path) -> dict:
    """Execute only an explicitly batch-approved configuration.

    The CLI additionally verifies signed-off config hashes. This check prevents
    an accidental direct call from starting optimizer updates.
    """
    if config.get("_batch_approved") is not True:
        raise PermissionError("Training requires explicit per-batch approval via the ktlab CLI")
    if config.get("model") in {"ktst_mean", "ktst_mhsa"}:
        from .ktst_engine import train_ktst
        return train_ktst(config, run_dir)
    shared = config.get("protocol") == SHARED_PROTOCOL
    if shared:
        config = _shared_config(config)
    epochs, patience = int(config.get("epochs", 200)), int(config.get("patience", 10))
    max_steps = config.get("max_train_steps")
    max_validation_batches = config.get("max_validation_batches")
    if max_validation_batches is not None and not config.get("debug_smoke", False):
        raise ValueError("Partial validation is allowed only with debug_smoke=true; never a benchmark result")
    wall_seconds = config.get("max_wall_seconds")
    if wall_seconds is not None and float(wall_seconds) <= 0:
        raise ValueError("max_wall_seconds must be positive")
    if epochs < 1 or patience < 1 or (max_steps is not None and int(max_steps) < 1):
        raise ValueError("epochs, patience, and max_train_steps must be positive")
    run_dir = Path(run_dir)
    if (run_dir / "best.pt").exists() or (run_dir / "history.jsonl").exists():
        raise FileExistsError("Run already has training artifacts; use a new run directory")
    if shared and (run_dir / "metadata.json").exists():
        raise FileExistsError("Shared run already has artifacts; use a new run directory")
    config = {k: v for k, v in config.items() if k != "_batch_approved"}
    config["dataset_manifest"] = str(Path(config["dataset_manifest"]).resolve())
    manifest = read_manifest(config)
    device = resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _seed(int(config.get("seed", 42)))
    model = build_model(config["model"], manifest["data_config"], config.get("model_config"), device)
    if config["model"] == "ukt":
        from .ukt_training import training_recipe
        recipe = training_recipe(model)
        if config.get("ukt_training_recipe", recipe) != recipe:
            raise ValueError("UKT training recipe conflicts with explicit model switches")
        config["ukt_training_recipe"] = recipe
    training = dataset_for_split(manifest, int(config.get("fold", 0)), "train", model_name=config["model"])
    validation = (_shared_dataset(manifest, config) if shared else
                  dataset_for_split(manifest, int(config.get("fold", 0)), "valid", model_name=config["model"]))
    train_loader = _loader(training, config, train=True)
    valid_loader = None if shared else _loader(validation, config)
    config["model_config"] = model.ktlab_model_config
    run_dir.mkdir(parents=True, exist_ok=True)
    _json(run_dir / "resolved_config.json", config)
    metadata = {"protocol": SHARED_PROTOCOL if shared else PROTOCOL, "run_id": run_dir.name, "status": "running", "test_accessed": False,
                "started_at": utc_now(), "git_commit": git_revision(), "implementation_sha256": source_fingerprint(),
                "config_sha256": config_fingerprint(config), "batch_id": config.get("batch_id"),
                "approved_batch_sha256": config.get("approved_batch_sha256"),
                "pykt_commit": manifest.get("pykt_commit"),
                "dataset_manifest_sha256": manifest["_manifest_sha256"],
                "model_source_sha256": model.ktlab_source_sha256, "device": str(device),
                "placement_binding_only": True, "torch": torch.__version__, "python": platform.python_version(),
                "hip": torch.version.hip, "cuda": torch.version.cuda,
                "train_file_sha256": _sha(training.path), "valid_file_sha256": _sha(validation.path),
                "train_rows": len(training), "valid_rows": len(validation), "optimizer_steps": 0,
                "model_parameters": sum(p.numel() for p in model.parameters()), "seed": int(config.get("seed", 42)),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    identity = _shared_identity(config, model, validation) if shared else None
    metadata["partial_debug"] = (config.get("max_validation_targets") is not None if shared else max_validation_batches is not None)
    if shared:
        metadata.update(identity, valid_event_audit=validation.audit,
                        training_representation="original_pykt_repeated_kc", initial_validation=True,
                        initial_checkpoint_untrained=True)
    _json(run_dir / "metadata.json", metadata)
    optimizer = (None if shared else torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)),
                                                       weight_decay=float(config.get("weight_decay", 0.0))))
    best, stop_reference, stale, steps = -math.inf, -math.inf, 0, 0
    started = time.monotonic()
    deadline = started + float(wall_seconds) if wall_seconds is not None else None
    stop_reason = "epochs_completed"
    try:
        if shared:
            initial_started = time.monotonic()
            initial = _shared_evaluate(model, validation, device, config, identity, deadline=deadline)
            if initial["auc"] is None:
                raise ValueError("Validation requires both response classes for AUC selection")
            initial_seconds = time.monotonic() - initial_started
            initial.update(candidate_epoch=0, optimizer_steps=0, training_started=False,
                           epoch_completed=False, validation_seconds=initial_seconds)
            best = stop_reference = initial["auc"]
            save_best_checkpoint(model, run_dir / "best.pt", best, -math.inf, 0, manifest["_manifest_sha256"],
                                 protocol=SHARED_PROTOCOL, identity=identity)
            metadata.update(best_epoch=0, best_valid_auc=best, checkpoint_sha256=_sha(run_dir / "best.pt"),
                            best_epoch_completed=False, initial_validation_seconds=initial_seconds)
            _json(run_dir / "initial_valid_metrics.json", initial)
            _json(run_dir / "metadata.json", metadata)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)),
                                         weight_decay=float(config.get("weight_decay", 0.0)))
        for epoch in range(1, epochs + 1):
            if shared:
                epoch_train_started = time.monotonic()
            model.train()
            total_loss, batches = 0., 0
            epoch_completed = True
            for batch_index, cpu_batch in enumerate(train_loader):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Run wall-clock budget exhausted")
                batch = move_batch(cpu_batch, device)
                prediction, regularizer = forward_model(model, batch)
                selected = batch["smasks"]
                if not bool(selected.any()):
                    continue
                loss = torch.nn.functional.binary_cross_entropy(prediction[selected].double(), batch["shft_rseqs"][selected].double()) + regularizer
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.get("grad_clip") is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()
                steps += 1
                batches += 1
                total_loss += float(loss.detach())
                if max_steps is not None and steps >= int(max_steps):
                    epoch_completed = batch_index + 1 == len(train_loader)
                    break
            if batches == 0:
                raise ValueError("No training targets selected")
            if shared:
                train_seconds = time.monotonic() - epoch_train_started
                validation_started = time.monotonic()
            valid = (_shared_evaluate(model, validation, device, config, identity, deadline=deadline) if shared else
                     evaluate_loader(model, valid_loader, device, max_batches=max_validation_batches, deadline=deadline))
            if shared:
                validation_seconds = time.monotonic() - validation_started
            auc = valid["auc"]
            if auc is None:
                raise ValueError("Validation requires both response classes for AUC selection")
            if save_best_checkpoint(model, run_dir / "best.pt", auc, best, epoch, manifest["_manifest_sha256"],
                                    protocol=SHARED_PROTOCOL if shared else PROTOCOL, identity=identity,
                                    epoch_completed=epoch_completed if shared else None):
                best = auc
                metadata.update(best_epoch=epoch, best_valid_auc=auc)
                if shared:
                    metadata["checkpoint_sha256"] = _sha(run_dir / "best.pt")
                    metadata["best_epoch_completed"] = epoch_completed
            if auc > stop_reference + float(config.get("early_stop_min_delta", 0.0)):
                stop_reference, stale = auc, 0
            else:
                stale += 1
            event = {"epoch": epoch, "optimizer_steps": steps, "train_loss": total_loss / batches,
                     "valid": valid, "elapsed_seconds": time.monotonic() - started}
            if shared:
                event.update(epoch_completed=epoch_completed, train_seconds=train_seconds,
                             validation_seconds=validation_seconds)
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event, allow_nan=False), flush=True)
            metadata["optimizer_steps"] = steps
            _json(run_dir / "metadata.json", metadata)
            if stale >= patience or (max_steps is not None and steps >= int(max_steps)):
                stop_reason = "early_stopping" if stale >= patience else "max_train_steps"
                break
        checkpoint = (_selected_shared_checkpoint(run_dir / "best.pt", metadata, identity,
                                                  manifest["_manifest_sha256"], config["model_config"], device=device)
                      if shared else torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
        model.load_state_dict(checkpoint["state_dict"])
        valid = (_shared_evaluate(model, validation, device, config, identity, run_dir / "valid_predictions.csv", deadline=deadline)
                 if shared else evaluate_loader(model, valid_loader, device, run_dir / "valid_predictions.csv",
                                                max_batches=max_validation_batches, deadline=deadline))
        _json(run_dir / "valid_metrics.json", valid)
        metadata.update(status="completed", optimizer_steps=steps, elapsed_seconds=time.monotonic() - started,
                        checkpoint_sha256=_sha(run_dir / "best.pt"), stop_reason=stop_reason,
                        finished_at=utc_now())
        _json(run_dir / "metadata.json", metadata)
        return {"run_dir": str(run_dir), "metadata": metadata, "valid": valid}
    except TimeoutError as exc:
        metadata.update(status="budget_exhausted", error=str(exc), optimizer_steps=steps,
                        elapsed_seconds=time.monotonic() - started, stop_reason="max_wall_seconds",
                        finished_at=utc_now())
        _json(run_dir / "metadata.json", metadata)
        return {"run_dir": str(run_dir), "metadata": metadata, "valid": None}
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}", optimizer_steps=steps,
                        finished_at=utc_now())
        _json(run_dir / "metadata.json", metadata)
        raise
    finally:
        metadata["ended_at"] = utc_now()
        metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        metadata["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        _json(run_dir / "metadata.json", metadata)
        training.close()
        validation.close()


def evaluate_run(run_dir: Path, split: str = "valid", allow_test: bool = False) -> dict:
    run_dir = Path(run_dir)
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test")
    if split == "test" and not allow_test:
        raise PermissionError("Test is sealed; explicit frozen-candidate test release is required")
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if config.get("protocol") == SHARED_PROTOCOL:
        return _evaluate_shared_run(run_dir, config, split)
    if config.get("model") in {"ktst_mean", "ktst_mhsa"}:
        from .ktst_engine import evaluate_ktst_run
        return evaluate_ktst_run(run_dir, split=split)
    if split == "test":
        from .provenance import validate_test_release
        release = Path(config.get("test_release_file", run_dir.parent / "test-release.json"))
        validate_test_release(run_dir, config, release)
        if (run_dir / "test_metrics.json").exists():
            raise FileExistsError("Final test already evaluated; reuse its saved predictions and metrics")
        if config.get("debug_smoke") or config.get("max_validation_batches") is not None:
            raise ValueError("Debug smoke runs are not eligible for final test evaluation")
    manifest = read_manifest(config)
    device = resolve_device(config.get("device", "auto"))
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    if checkpoint["dataset_manifest_sha256"] != manifest["_manifest_sha256"]:
        raise ValueError("Prepared data manifest changed since training")
    model = build_model(config["model"], manifest["data_config"], checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = dataset_for_split(manifest, int(config.get("fold", 0)), split, model_name=config["model"])
    try:
        result = evaluate_loader(model, _loader(dataset, config), device, run_dir / f"{split}_predictions.csv",
                                 max_batches=config.get("max_validation_batches") if split == "valid" else None)
        _json(run_dir / f"{split}_metrics.json", result)
        if split == "test":
            metadata_path = run_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["test_accessed"] = True
            _json(metadata_path, metadata)
        return result
    finally:
        dataset.close()
