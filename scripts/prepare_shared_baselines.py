"""Prepare the next reviewable training proposal; never approve or train it."""
import argparse
import copy

from ktlab.provenance import canonical_hash, git_revision, project_root, read_json, source_fingerprint, write_json
from ktlab.shared_events import PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours-per-run", type=float, default=4.)
    parser.add_argument("--nips-hours-per-run", type=float, default=8.)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if not 0 < args.hours_per_run <= 24:
        parser.error("--hours-per-run must be in (0, 24]")
    if not 0 < args.nips_hours_per_run <= 24:
        parser.error("--nips-hours-per-run must be in (0, 24]")
    root = project_root()
    previous = read_json(root / "configs/batches/B002-one-epoch-calibration.json")
    checks = read_json(root / "reports/SHARED_PROTOCOL_CHECKS_2026-10-14.json")
    source = source_fingerprint()
    if checks["implementation_sha256"] != source:
        raise ValueError("Complete shared protocol checks on the current implementation before preparing this batch")
    audits = {entry["dataset"]: entry for entry in checks["checks"]}
    batch = copy.deepcopy(previous)
    batch.update(batch_id="B003-shared-convergence-baselines", status="proposed", source_sha256=source,
                 git_commit_at_preparation=git_revision(), base_batch_sha256=canonical_hash(previous),
                 purpose="Fixed-configuration baseline convergence under shared development validation; not tuned SOTA reproduction or final test",
                 comparison_boundary="Validation targets/histories and checkpoint selection are shared; original repeated-KC versus set-event training is retained; no paper-best or SOTA claim")
    per_dataset = {"assist2009": int(args.hours_per_run * 3600), "assist2015": int(args.hours_per_run * 3600),
                   "nips_task34": int(args.nips_hours_per_run * 3600)}
    batch["limits"].update(epochs_per_run=args.epochs, per_run_wall_seconds=max(per_dataset.values()),
                           per_dataset_run_wall_seconds=per_dataset, sum_run_wall_seconds=0,
                           total_optimizer_steps_upper_bound=0)
    batch["selection"] = {"protocol": PROTOCOL, "fold": 0, "seed": 42, "initial_validation": True,
                           "every_epoch_full_validation": True, "strict_best_auc": True,
                           "patience": args.patience, "ties_keep_earlier_checkpoint": True,
                           "final_best_checkpoint_reload_and_export": True,
                           "hyperparameter_search_trials_per_model_dataset": 1}
    for run in batch["runs"]:
        run["run_id"] = run["run_id"].replace("B002-", "B003-", 1)
        config = run["config"]
        seconds = per_dataset[config["dataset"]]
        record = audits[config["dataset"]]
        if run["dataset_manifest_sha256"] != record["dataset_manifest_sha256"]:
            raise ValueError("The shared checks used a different prepared dataset manifest")
        audit = record["full_validation_audit"]
        if audit["selection_limited"] or audit["selected_targets"] != audit["eligible_total"]:
            raise ValueError("A full shared validation audit is required")
        run.pop("v1_validation_rows", None)
        run["shared_validation"] = {key: audit[key] for key in (
            "protocol", "selected_targets", "source_sha256", "target_sha256", "context_sha256",
            "canonical_events", "exclusion_counts")}
        steps = ((run["train_sequence_rows"] + config["batch_size"] - 1) // config["batch_size"]) * args.epochs
        config.update(protocol=PROTOCOL, epochs=args.epochs, patience=args.patience, early_stop_min_delta=0.,
                      max_train_steps=steps, max_wall_seconds=seconds, debug_smoke=False,
                      initial_validation=True, validation_batch_size=32, max_rows_per_forward=32,
                      purpose="fixed_configuration_convergence_anchor_shared_validation_not_sota")
        batch["limits"]["total_optimizer_steps_upper_bound"] += steps
        batch["limits"]["sum_run_wall_seconds"] += seconds
    target = root / "configs/batches/B003-shared-convergence-baselines.json"
    write_json(target, batch)
    print(f"Proposed only: {target}; canonical SHA256={canonical_hash(batch)}")


if __name__ == "__main__":
    main()
