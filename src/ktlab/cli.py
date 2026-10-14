"""Command-line interface. Training and final test are separate approved actions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import provenance as p


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _smoke_batch(root: Path) -> dict:
    from .data import inspect_dataset
    manifest = inspect_dataset("assist2015", root)
    if manifest.get("status") != "verified":
        raise ValueError("Prepare and inspect ASSIST2015 before proposing a concrete batch.")
    manifest_path = root / "data/prepared/assist2015/manifest.json"
    runs = []
    for model in ("akt", "simplekt"):
        config = {
            "dataset_manifest": str(manifest_path), "model": model,
            "model_config": {"d_model": 64, "n_blocks": 1, "dropout": 0.1, "d_ff": 128, "final_fc_dim": 128, "num_attn_heads": 4},
            "fold": 0, "seed": 42, "device": "cuda", "batch_size": 32,
            "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0,
            "patience": 10, "early_stop_min_delta": 0.0, "max_train_steps": 5,
            "max_validation_batches": 10, "max_wall_seconds": 900,
            "debug_smoke": True,
            "purpose": "execution_smoke_not_model_quality",
        }
        runs.append({"run_id": f"B001-{model}-assist2015-f0-s42", "dataset_manifest_sha256": p.sha256_file(manifest_path), "config": config})
    return {
        "schema_version": 1, "batch_id": "B001-smoke", "status": "proposed",
        "purpose": "Verify optimizer/checkpoint/validation pipeline, not assess predictive quality or SOTA.",
        "source_sha256": p.source_fingerprint(root), "git_commit_at_preparation": p.git_revision(root),
        "limits": {"total_optimizer_steps": 10, "per_run_wall_seconds": 900, "paid_services": False},
        "test_access": False, "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kt", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Read-only runtime and GPU probe")
    commands.add_parser("status", help="Show data and run presence without training")
    data = commands.add_parser("data").add_subparsers(dest="data_action", required=True)
    for action in ("prepare", "inspect"):
        entry = data.add_parser(action)
        entry.add_argument("dataset", choices=["assist2009", "assist2015", "nips34", "nips_task34", "all"])
    batch = commands.add_parser("batch").add_subparsers(dest="batch_action", required=True)
    proposal = batch.add_parser("prepare-smoke", help="Create a proposal only; never approves or trains")
    proposal.add_argument("--output", type=Path, default=Path("configs/batches/B001-smoke.json"))
    inspection = batch.add_parser("inspect")
    inspection.add_argument("path", type=Path)
    train = commands.add_parser("train", help="Execute only an explicitly approved, unchanged batch")
    train.add_argument("--batch", type=Path, required=True)
    train.add_argument("--approval", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--split", choices=["valid", "test"], default="valid")
    evaluate.add_argument("--allow-test", action="store_true", help="Also requires an approved frozen test-release record")
    commands.add_parser("report", help="List actual completed-run summaries; never invent results")
    args = parser.parse_args(argv)
    root = p.project_root()
    try:
        if args.command == "doctor":
            return subprocess.run([sys.executable, str(root / "scripts/environment_probe.py")], check=False).returncode
        if args.command == "status":
            _emit({"root": str(root), "git_commit": p.git_revision(root), "source_sha256": p.source_fingerprint(root),
                   "prepared_manifests": [str(x.relative_to(root)) for x in sorted((root / "data/prepared").glob("*/manifest.json"))],
                   "run_directories": [x.name for x in sorted((root / "runs").glob("*")) if x.is_dir()],
                   "note": "Presence is not proof of integrity or successful training. Use data inspect and run summaries."})
        elif args.command == "data":
            from .data import inspect_dataset, prepare_dataset
            fn = prepare_dataset if args.data_action == "prepare" else inspect_dataset
            names = ["assist2009", "assist2015", "nips_task34"] if args.dataset == "all" else [args.dataset]
            for name in names:
                result = fn(name, root)
                _emit({k: result[k] for k in ("dataset", "status", "manifest_path", "split_audit") if k in result})
        elif args.command == "batch":
            if args.batch_action == "prepare-smoke":
                proposal = _smoke_batch(root)
                p.write_json(args.output, proposal)
                _emit({"proposal": str(args.output), "batch_sha256": p.canonical_hash(proposal), "status": "pending_user_approval"})
            else:
                content = p.read_json(args.path)
                _emit({"batch_sha256": p.canonical_hash(content), "manifest": content})
        elif args.command == "train":
            # Validate the whole batch before importing models or opening any training loader.
            approved = p.validate_batch_approval(args.batch, args.approval)
            from .engine import train
            for entry in approved["runs"]:
                p.validate_batch_approval(args.batch, args.approval)
                config = dict(entry["config"], _batch_approved=True, batch_id=approved["batch_id"], approved_batch_sha256=p.canonical_hash(approved))
                _emit(train(config, root / "runs" / entry["run_id"]))
        elif args.command == "evaluate":
            from .engine import evaluate_run
            _emit(evaluate_run(args.run, split=args.split, allow_test=args.allow_test))
        elif args.command == "report":
            summaries = []
            for run_dir in sorted((root / "runs").glob("*")):
                if run_dir.is_dir():
                    files = [x.name for x in run_dir.glob("*.json")]
                    metadata_path = run_dir / "metadata.json"
                    metadata = p.read_json(metadata_path) if metadata_path.is_file() else {}
                    metrics = {split: p.read_json(run_dir / f"{split}_metrics.json") for split in ("valid", "test") if (run_dir / f"{split}_metrics.json").is_file()}
                    summaries.append({"run_id": run_dir.name, "status": metadata.get("status", "unknown"),
                                      "partial_debug": metadata.get("partial_debug"), "metrics": metrics,
                                      "json_artifacts": files, "checkpoint_present": (run_dir / "best.pt").exists()})
            _emit({"runs": summaries, "note": "No runs means no experimental result. Smoke results are execution evidence only."})
        return 0
    except (p.ApprovalError, ValueError, OSError, RuntimeError) as exc:
        print(f"kt: {exc}", file=sys.stderr)
        return 2
