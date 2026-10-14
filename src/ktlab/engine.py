"""Approval-gated training and sealed-test evaluation for a declared protocol.

The protocol deliberately differs from exact pyKT reproduction: checkpoint
selection uses causal question validation, saves the true best AUC, and uses
full-precision metrics. Upstream repeat-KC training semantics are retained.
"""
from __future__ import annotations

import hashlib
import json
import math
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


def save_best_checkpoint(model, path: Path, auc: float, best: float, epoch: int, manifest_sha256: str) -> bool:
    """Persist every strict improvement, independently of early-stop min_delta."""
    if not math.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError("Checkpoint selection requires finite validation AUC in [0,1]")
    if auc <= best:
        return False
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "model_config": model.ktlab_model_config,
                "epoch": epoch, "valid_auc": auc, "protocol": PROTOCOL,
                "dataset_manifest_sha256": manifest_sha256}, temporary)
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
    config = {k: v for k, v in config.items() if k != "_batch_approved"}
    config["dataset_manifest"] = str(Path(config["dataset_manifest"]).resolve())
    manifest = read_manifest(config)
    device = resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _seed(int(config.get("seed", 42)))
    model = build_model(config["model"], manifest["data_config"], config.get("model_config"), device)
    training = dataset_for_split(manifest, int(config.get("fold", 0)), "train", model_name=config["model"])
    validation = dataset_for_split(manifest, int(config.get("fold", 0)), "valid", model_name=config["model"])
    train_loader, valid_loader = _loader(training, config, train=True), _loader(validation, config)
    config["model_config"] = model.ktlab_model_config
    run_dir.mkdir(parents=True, exist_ok=True)
    _json(run_dir / "resolved_config.json", config)
    metadata = {"protocol": PROTOCOL, "run_id": run_dir.name, "status": "running", "test_accessed": False,
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
    metadata["partial_debug"] = max_validation_batches is not None
    _json(run_dir / "metadata.json", metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)),
                                 weight_decay=float(config.get("weight_decay", 0.0)))
    best, stop_reference, stale, steps = -math.inf, -math.inf, 0, 0
    started = time.monotonic()
    deadline = started + float(wall_seconds) if wall_seconds is not None else None
    stop_reason = "epochs_completed"
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, batches = 0., 0
            for cpu_batch in train_loader:
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
                    break
            if batches == 0:
                raise ValueError("No training targets selected")
            valid = evaluate_loader(model, valid_loader, device, max_batches=max_validation_batches, deadline=deadline)
            auc = valid["auc"]
            if auc is None:
                raise ValueError("Validation requires both response classes for AUC selection")
            if save_best_checkpoint(model, run_dir / "best.pt", auc, best, epoch, manifest["_manifest_sha256"]):
                best = auc
                metadata.update(best_epoch=epoch, best_valid_auc=auc)
            if auc > stop_reference + float(config.get("early_stop_min_delta", 0.0)):
                stop_reference, stale = auc, 0
            else:
                stale += 1
            event = {"epoch": epoch, "optimizer_steps": steps, "train_loss": total_loss / batches,
                     "valid": valid, "elapsed_seconds": time.monotonic() - started}
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event, allow_nan=False), flush=True)
            metadata["optimizer_steps"] = steps
            _json(run_dir / "metadata.json", metadata)
            if stale >= patience or (max_steps is not None and steps >= int(max_steps)):
                stop_reason = "early_stopping" if stale >= patience else "max_train_steps"
                break
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        valid = evaluate_loader(model, valid_loader, device, run_dir / "valid_predictions.csv",
                                max_batches=max_validation_batches, deadline=deadline)
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
