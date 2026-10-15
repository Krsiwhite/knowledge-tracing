"""Prepare B008 from pinned saved evidence; never import a model or read CSVs.

Root runs this only after the separately registered data preflight completes:
  PYTHONPATH=src python .cache/prepare_b008.py --preflight-sha256 <reviewed SHA256>
The only created file is the fixed batch manifest, opened with mode 'x'.
Preparation is not authorization registration and cannot launch training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "knowledge-tracing"
sys.path.insert(0, str(ROOT / "src"))
from ktlab.provenance import canonical_hash, sha256_file, source_fingerprint

SOURCE = "7066077ebd02a0bd0cecccf3d3a6b3c57df9b8162a1370f2e3cc576cb14fbfef"
HISTORY_SOURCE = "c930087ccf7872ba3db7d21229e13f282ea97ca718f2f171c1814356e6f51f50"
BASELINE_ID = "B003-R1-ktst_mean-assist2009-f0-s42"
BASELINE_SOURCE = "bf5fcd0f23645b31f76c9486f4a98b3a5f9667886f5f80d62efc596b9a610a1d"
BASELINE_BATCH_CANONICAL = "c2c4dbc0ffcfee1bb591948e6dffac26366b723bd13d7c8e484ebff886c15bd3"
DATA_SHA = "e2c939dfe4479327c223644bce6aee88c56bc362d1169164332ae9ca479fc703"
CENSUS_SHA = "8ea7dfeb704121f91a9a6733a7d28d00e7fc344e4a1a429171331aedbe814ff2"
CHECK_CANONICAL = "47913ac2b152e0fa2a1343b325ea713f39bf2d7541d173efbb7cdc4142ee8509"
REPAIR_CANONICAL = "ccf7d7dba4bda0b22dba79cff796e4d0cd79affcd7ca064e5c4d1a379624cf41"
LONG = "sampled_complete_suffix_v1"
SHORT = "sampled_chunk_prefix_v1"
BATCH_ID = "B008-ktst-history-assist2009-pilot"
OUTPUT = ROOT / "configs/batches/B008-ktst-history-assist2009-pilot.json"
PREFLIGHT = ROOT / "reports/B008_HISTORY_DATA_PREFLIGHT_2026-10-15.json"
PREFLIGHT_HELPER = ROOT / ".cache/preflight_b008_history_data.py"
CENSUS = ROOT / "reports/KTST_TRAINING_CONTEXT_COVERAGE_2026-10-15.json"
REMOVED_FIELDS = {"batch_id", "approved_batch_sha256"}
ADDED_FIELDS = {"ktst_history_training_recipe", "ktst_history_samples_per_chunk",
                "ktst_history_training_source_sha256"}
CHANGED_FIELDS = {"max_train_steps", "max_wall_seconds", "purpose"}
PURPOSE = "paired_complete_history_training_adaptation_with_frozen_sampling_and_reused_full_supervision_anchor_not_sota"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_file(path, expected_sha=None):
    path = Path(path)
    require(path.suffix == ".json", f"Only JSON evidence may be parsed: {path}")
    if expected_sha is not None:
        require(sha256_file(path) == expected_sha, f"Evidence SHA mismatch: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def receipt(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def sha_format(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def current_git_revision():
    git_pointer = (ROOT / ".git").read_text(encoding="utf-8").strip()
    require(git_pointer.startswith("gitdir: "), "Expected an existing linked Git worktree")
    raw = git_pointer[len("gitdir: "):]
    if sys.platform != "win32" and re.match(r"^[A-Za-z]:/", raw):
        raw = "/mnt/" + raw[0].lower() + raw[2:]
    git_dir = Path(raw)
    if not git_dir.is_absolute():
        git_dir = ROOT / git_dir
    result = subprocess.run(["git", "--git-dir", str(git_dir), "--work-tree", str(ROOT),
                             "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    revision = result.stdout.strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Missing Git identity")
    return revision


def acceptance_evidence():
    initial_path = ROOT / "configs/checks/S010-ktst-history-training-cpu.json"
    repair_path = ROOT / "configs/checks/S010-R1-ktst-history-reader-message.json"
    initial = json_file(initial_path, "0e365cc2c282d9d2e7147f1a962a852c47b0effdce658eb087fb035a4223ac40")
    repair = json_file(repair_path, "fc17e8157968515c7c8419d386340277ae6c3756a503b87a60213fb444345814")
    require(canonical_hash(initial) == CHECK_CANONICAL and canonical_hash(repair) == REPAIR_CANONICAL,
            "Acceptance manifest canonical identity mismatch")
    require(initial["implementation_sha256"] == repair["implementation_sha256"] == SOURCE,
            "Acceptance source differs")
    require(repair["predecessor_manifest_sha256"] == CHECK_CANONICAL, "Repair predecessor differs")
    first_path = ROOT / "reports/S010_KTST_HISTORY_CPU_INITIAL_2026-10-15.json"
    second_path = ROOT / "reports/S010_R1_KTST_HISTORY_READER_MESSAGE_2026-10-15.json"
    first = json_file(first_path, "90652f57cebb770e4b1d3ee01ae81044f8c80c6566be5a65bd723733602ed44d")
    second = json_file(second_path, "d6ef7d04dc18ab833c15a3441a09650607eb1c7ad2ef1c178cd5cf9399b3c106")
    require(first["status"] == "failed" and first["tests_passed"] == 209 and first["exit_code"] == 1,
            "Preserve the initial 209-pass / four-failure receipt")
    require(first["expected_counts_match"] is True and first["actual_counts"] == {
        "forwards": 36, "gradients": 8, "models": 6, "optimizers": 0}, "Initial compute counts differ")
    require(second["status"] == "passed" and second["tests_passed"] == 4 and second["exit_code"] == 0,
            "The four-case message-only repair must have passed")
    require(second["message_only_change_verified"] is True, "Repair scope unverified")
    for result, manifest, path in ((first, initial, initial_path), (second, repair, repair_path)):
        require(result["implementation_sha256"] == SOURCE and result["source_and_bound_files_unchanged"] is True,
                "Acceptance frozen evidence differs")
        require(result["manifest_sha256"] == canonical_hash(manifest)
                and result["manifest_file_sha256"] == sha256_file(path), "Acceptance receipt manifest differs")
        require(result["timed_out"] is False, "Acceptance timed out")
    require(all(second[name] == 0 for name in ("models", "forwards", "gradients", "optimizer_steps")),
            "The repair was not a pure-file check")
    for name, expected in repair["files"].items():
        require(sha256_file(ROOT / name) == expected, f"R1 bound file changed: {name}")
    for name, expected in initial["files"].items():
        if name == "scripts/analyze_shared_predictions.py":
            data = (ROOT / name).read_bytes()
            old = repair["message_only_change"]["old"].encode()
            new = repair["message_only_change"]["new"].encode()
            require(data.count(new) == 1, "Message-only repair is not unique")
            require(hashlib.sha256(data.replace(new, old, 1)).hexdigest() == expected,
                    "Analyzer changed beyond the accepted message repair")
        else:
            require(sha256_file(ROOT / name) == expected, f"S010 bound file changed: {name}")
    return {"initial_manifest": receipt(initial_path), "initial_report": receipt(first_path),
            "repair_manifest": receipt(repair_path), "repair_report": receipt(second_path),
            "initial_status": "209 passed / 4 failed; all four failures were existing duplicate-recipe message expectations",
            "repair_status": "4 passed; message-only repair; no repeated model checks",
            "distinct_cases_supported": 213, "implementation_sha256": SOURCE,
            "limits": "Synthetic acceptance supports scoped implementation correctness, not real-data gains, speed or novelty."}


def baseline_evidence():
    batch_path = MAIN / "configs/batches/B003-R1-shared-convergence-baselines.json"
    batch = json_file(batch_path, "92e211976e8d33d89d75021e9489210d991f05260e6acfebc0a752509bab2310")
    require(canonical_hash(batch) == BASELINE_BATCH_CANONICAL and batch["source_sha256"] == BASELINE_SOURCE,
            "Real baseline batch identity differs")
    entries = [entry for entry in batch["runs"] if entry["run_id"] == BASELINE_ID]
    require(len(entries) == 1, "Baseline run is not uniquely present in its real manifest")
    entry = entries[0]
    run = MAIN / "runs" / BASELINE_ID
    config_path, metadata_path = run / "resolved_config.json", run / "metadata.json"
    config = json_file(config_path, "36ba9522d59115ab293c1edab2eb7fcb58396fa6fc3bdd8a3cdb2bbdeda3deaa")
    metadata = json_file(metadata_path, "4389d45971e53a5844d07ab59abf3ded026e9267a8c3fafebacbdf11a07bde96")
    require(metadata["status"] == "completed" and metadata["test_accessed"] is False,
            "Only the completed development baseline may be reused")
    require(metadata["run_id"] == BASELINE_ID and metadata["implementation_sha256"] == BASELINE_SOURCE,
            "Baseline metadata source/run differs")
    require(metadata["approved_batch_sha256"] == config["approved_batch_sha256"] == BASELINE_BATCH_CANONICAL,
            "Baseline recorded batch hash differs")
    require(metadata["batch_id"] == config["batch_id"] == batch["batch_id"], "Baseline batch ID differs")
    require(canonical_hash(config) == metadata["config_sha256"], "Baseline resolved config identity differs")
    require(metadata["dataset_manifest_sha256"] == entry["dataset_manifest_sha256"] == DATA_SHA,
            "Baseline dataset differs")
    for key, value in entry["config"].items():
        if key == "model_config":
            require(all(config[key].get(k) == v for k, v in value.items()), "Resolved model changed baseline choices")
        else:
            require(config.get(key) == value, f"Resolved baseline choice differs: {key}")
    require(metadata["train_rows"] == entry["train_sequence_rows"] == 3016,
            "Original training chunk count differs")
    require(metadata["train_event_audit"]["scored_event_fragments"] == 182640,
            "Original full supervision count differs")
    for key, actual_key in (("source_sha256", "validation_source_sha256"),
                            ("target_sha256", "validation_target_sha256"),
                            ("context_sha256", "validation_context_sha256")):
        require(entry["shared_validation"][key] == metadata[actual_key], "Baseline shared validation differs")
    require(entry["shared_validation"]["selected_targets"] == metadata["valid_rows"] == 43295,
            "Baseline validation target count differs")
    require(not (ADDED_FIELDS | {"ktst_interaction_recipe", "shared_validation_execution"}) & config.keys(),
            "Baseline unexpectedly contains a later recipe")
    record = {"run_id": BASELINE_ID, "run_directory": str(run), "status_at_preparation": "completed",
              "batch_id": batch["batch_id"], "batch_manifest": receipt(batch_path),
              "batch_canonical_sha256": BASELINE_BATCH_CANONICAL, "implementation_sha256": BASELINE_SOURCE,
              "run_git_commit": metadata["git_commit"], "resolved_config": receipt(config_path),
              "config_sha256": metadata["config_sha256"], "metadata": receipt(metadata_path),
              "model_source_sha256": metadata["model_source_sha256"], "dataset_manifest_sha256": DATA_SHA,
              "checkpoint_sha256_recorded_in_metadata": metadata["checkpoint_sha256"],
              "checkpoint_bytes_read_during_preparation": False, "best_epoch_recorded": metadata["best_epoch"],
              "baseline_retraining": False, "additional_baseline_inference": False,
              "role": "Completed original full-supervision practical anchor; not asserted to be the strongest future AS09 result"}
    return config, entry, record


def preflight_evidence(expected_report_sha):
    require(sha_format(expected_report_sha), "A reviewed lowercase preflight SHA256 is required")
    report = json_file(PREFLIGHT, expected_report_sha)
    require(report["status"] == "completed" and report["preflight_id"] == "B008-HISTORY-DATA-PREFLIGHT-2026-10-15",
            "Successful concrete B008 data preflight is required")
    require(report["source_sha256"] == SOURCE and report["helper_sha256"] == sha256_file(PREFLIGHT_HELPER),
            "Preflight source/helper differs")
    plan_path = Path(report["plan_path"]).resolve()
    require(plan_path.is_relative_to(ROOT.resolve()), "Preflight plan must stay in this worktree")
    plan = json_file(plan_path, report["plan_file_sha256"])
    require(canonical_hash(plan) == report["plan_canonical_sha256"], "Preflight plan canonical identity differs")
    require(plan["source_sha256"] == SOURCE and plan["helper_sha256"] == report["helper_sha256"]
            and plan["inputs"] == report["inputs"], "Preflight report does not bind its actual plan inputs/source")
    require(plan["authorization_kind"] == "existing_direct_user_standing_authorization"
            and plan["user_confirmation"] == "启动B003，但每次最多50轮，3次未见则早停。之后的执行不需要我再批准",
            "Preflight lacks the real standing authorization basis")
    require(report["dataset"] == "assist2009" and report["held_out_fold"] == 0
            and report["seed"] == 42 and report["samples_per_chunk"] == 8, "Preflight dataset/fold/seed/K differs")
    require(len(report["recipes"]) == 2 and set(report["recipes"]) == {LONG, SHORT}, "Preflight recipes differ")
    require(report["receipts_unchanged"] is True, "Preflight input/source receipts were not retained")
    require(report["test_accessed"] is False and report["weights_accessed"] is False,
            "Preflight must not access test data or existing weights")
    census = json_file(CENSUS, CENSUS_SHA)
    require(census["status"] == "completed", "Completed coverage census is required")
    census_rows = [row for row in census["datasets"] if row["dataset"] == "assist2009"]
    require(len(census_rows) == 1, "AS09 census missing or duplicate")
    census_row = census_rows[0]
    primary = census_row["primary"]
    require(primary["eligible_chunks"] == 2915 and primary["eligible_unique_targets"] == 172169
            and primary["original_supervised_fragments"] == 182640
            and primary["expanded_history_targets"] == 51860, "Pinned census scope differs")
    dataset_path = MAIN / "data/prepared/assist2009/manifest.json"
    dataset = json_file(dataset_path, DATA_SHA)
    expected_paths = {"dataset_manifest": dataset_path, "census": CENSUS,
                      "train_valid.csv": dataset_path.parent / "train_valid.csv",
                      "train_valid_sequences.csv": dataset_path.parent / "train_valid_sequences.csv"}
    require(set(report["inputs"]) == set(expected_paths), "Unexpected preflight input collection")
    for key, path in expected_paths.items():
        item = report["inputs"][key]
        require(Path(item["path"]).resolve() == path.resolve(), f"Preflight input location differs: {key}")
        if key in {"dataset_manifest", "census"}:
            require(item["sha256"] == sha256_file(path) and item["bytes"] == path.stat().st_size,
                    f"Preflight JSON evidence differs: {key}")
        else:
            # Deliberately compare saved receipts without opening sequence-file contents.
            require({"sha256": item["sha256"], "bytes": item["bytes"]} == dataset["files"][key]
                    == census_row["source_receipts"][key], f"Preflight data receipts differ: {key}")
    require(census_row["manifest_sha256"] == DATA_SHA, "Census dataset manifest differs")
    eligibility = report["eligibility"]
    require(eligibility["eligible_chunks"] == 2915 and eligibility["eligible_unique_targets"] == 172169
            and eligibility["matched_census"] is True and sha_format(eligibility["eligible_target_index_sha256"]),
            "Actual training-loader eligibility must match the strict primary census")
    sampling = report["sampling"]
    require(sampling["epochs"] == list(range(1, 51)) and sampling["all_recipes_identical"] is True,
            "Preflight must bind both samplers through all 50 epochs")
    rows = sampling["per_epoch"]
    require(len(rows) == 50 and [row["epoch"] for row in rows] == list(range(1, 51)), "Sampling epochs differ")
    for row in rows:
        require(set(row) == {"epoch", "sampling_sha256", "eligible_chunks", "sampled_rows", "unique_targets_sampled"},
                "Unexpected sampling audit schema")
        require(row["eligible_chunks"] == 2915 and row["sampled_rows"] == 23320
                and type(row["unique_targets_sampled"]) is int and 2915 <= row["unique_targets_sampled"] <= 23320
                and sha_format(row["sampling_sha256"]), "Sampling counts or identity differ")
    require(canonical_hash(rows) == sampling["combined_sampling_sha256"], "Combined 50-epoch sampling identity differs")
    projection = report["projection"]
    projection_expected = {"checked": True, "paired_chunks": 32, "chunk_getitem_calls": 64,
                           "target_contexts": 512, "paired_sampled_targets": 256,
                           "training_folds": [1, 2, 3, 4], "includes_first_and_last": True,
                           "all_pair_targets_labels_weights_equal": True, "all_histories_legal": True,
                           "all_target_response_slots_zero": True}
    require(all(projection.get(key) == value for key, value in projection_expected.items()),
            "The bounded real-data projections did not pass their declared checks")
    require(sha_format(projection["selected_chunk_indices_sha256"])
            and sha_format(projection["data_projection_sha256"]), "Projection receipts missing")
    expected_counts = {"dataset_instances": 2, "sampling_audit_calls": 100, "chunk_getitem_calls": 64,
                       "target_contexts": 512, "model_instances": 0, "forward_calls": 0,
                       "gradient_calls": 0, "optimizer_constructions": 0, "optimizer_steps": 0,
                       "forbidden_data_open_attempts": 0}
    require(report["actual_counts"] == expected_counts, "Preflight activity differs from its bounded contract")
    return {"report": receipt(PREFLIGHT), "helper": receipt(PREFLIGHT_HELPER), "plan": receipt(plan_path),
            "plan_canonical_sha256": report["plan_canonical_sha256"], "source_sha256": SOURCE,
            "inputs": deepcopy(report["inputs"]), "eligibility": deepcopy(eligibility),
            "sampling": deepcopy(sampling), "projection": deepcopy(projection),
            "actual_counts": deepcopy(expected_counts), "coverage_census": receipt(CENSUS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args()
    require(not OUTPUT.exists(), "B008 manifest already exists; refuse overwrite or unrecorded revision")
    require(source_fingerprint(ROOT) == SOURCE, "Frozen accepted training source changed")
    require(sha256_file(ROOT / "src/ktlab/ktst_history_training.py") == HISTORY_SOURCE, "History helper changed")
    acceptance = acceptance_evidence()
    original, baseline_entry, baseline = baseline_evidence()
    preflight = preflight_evidence(args.preflight_sha256)
    base = deepcopy(original)
    for key in REMOVED_FIELDS:
        require(key in base, f"Missing old runtime binding: {key}")
        del base[key]
    require(base["epochs"] == 50 and base["patience"] == 3 and base["early_stop_min_delta"] == 0.0
            and base["initial_validation"] is True and base["batch_size"] == 32
            and base["max_rows_per_forward"] == 32 and base["model"] == "ktst_mean"
            and base["fold"] == 0 and base["seed"] == 42, "Inherited execution contract differs")
    chunks, k, epochs, batch_size, micro = 2915, 8, 50, 32, 32
    full_batches, final_chunks = divmod(chunks, batch_size)
    updates = full_batches + bool(final_chunks)
    forwards = full_batches * ((batch_size * k + micro - 1) // micro)
    forwards += (final_chunks * k + micro - 1) // micro
    require((updates, forwards, chunks * k, updates * epochs, forwards * epochs) ==
            (92, 729, 23320, 4600, 36450), "Budget arithmetic differs")
    runs = []
    for recipe in (LONG, SHORT):
        config = deepcopy(base)
        config.update(ktst_history_training_recipe=recipe, ktst_history_samples_per_chunk=k,
                      ktst_history_training_source_sha256=HISTORY_SOURCE, max_train_steps=4600,
                      max_wall_seconds=28800, purpose=PURPOSE)
        require(set(config) - set(base) == ADDED_FIELDS and set(base) - set(config) == set(),
                "Unlisted config addition/removal")
        changed = {key for key in base if config[key] != base[key]}
        require(changed == CHANGED_FIELDS, "Unlisted inherited configuration change")
        require(config["model_config"] == original["model_config"]
                and "ktst_interaction_recipe" not in config and "shared_validation_execution" not in config,
                "No interaction or validation execution recipe may be added")
        run_id = f"B008-ktst_mean-{recipe}-k8-assist2009-f0-s42"
        require(not (ROOT / "runs" / run_id).exists(), f"Run directory already exists: {run_id}")
        runs.append({"run_id": run_id, "dataset_manifest_sha256": DATA_SHA,
                     "train_sequence_rows": chunks, "original_train_sequence_rows": 3016,
                     "eligible_unique_training_targets": 172169, "training_batches_per_epoch": updates,
                     "sampled_training_rows_per_epoch": chunks * k,
                     "training_complete_forwards_per_full_epoch": forwards,
                     "config": config, "shared_validation": deepcopy(baseline_entry["shared_validation"]),
                     "training_data_preflight_report_sha256": args.preflight_sha256,
                     "eligible_target_index_sha256": preflight["eligibility"]["eligible_target_index_sha256"],
                     "planned_epoch_sampling_combined_sha256": preflight["sampling"]["combined_sampling_sha256"]})
    batch = {
        "schema_version": 1, "batch_id": BATCH_ID, "status": "prepared",
        "purpose": "Prioritize a paired AS09 long-history K8 versus short-history K8 experiment, reusing the completed original full-supervision KTST reference",
        "source_sha256": SOURCE, "git_commit_at_preparation": current_git_revision(),
        "test_access": False, "stop_on_noncompleted_run": True,
        "execution_order": [entry["run_id"] for entry in runs],
        "priority_and_authority_basis": {
            "standing_user_instruction": "启动B003，但每次最多50轮，3次未见则早停。之后的执行不需要我再批准",
            "latest_priority_instruction": "能不能优先做咱自己模型的实验",
            "new_confirmation_solicited": False, "approval_record_created_by_this_preparation": False,
            "launch_registration_required": True,
            "choice_basis": "Completed census, completed timing and the existing AS09 anchor; not B007 intermediate metrics",
        },
        "limits": {"total_optimizer_steps_upper_bound": 9200, "epochs_per_run": 50,
                   "per_run_optimizer_steps_upper_bound": 4600, "per_run_wall_seconds": 28800,
                   "sum_run_wall_seconds": 57600,
                   "per_run_training_forward_calls_full_epochs_upper_bound": 36450,
                   "sum_training_forward_calls_full_epochs_upper_bound": 72900,
                   "per_run_sampled_training_rows_full_epochs_upper_bound": 1166000,
                   "sum_sampled_training_rows_full_epochs_upper_bound": 2332000,
                   "wall_limit_scope": "Soft checks inside training/validation after model/data setup; setup is additional; partial microbatch timeout must not commit an update",
                   "paid_services": False, "automatic_retries": False,
                   "automatic_config_shrink_or_budget_extension": False},
        "budget_derivation": {"eligible_chunks": chunks, "chunks_per_optimizer_batch": batch_size,
                              "samples_per_chunk": k, "max_rows_per_forward": micro,
                              "full_optimizer_batches": full_batches, "final_batch_chunks": final_chunks,
                              "final_batch_sampled_rows": final_chunks * k,
                              "optimizer_batches_per_epoch": updates, "complete_training_forwards_per_epoch": forwards,
                              "sampled_rows_per_epoch": chunks * k,
                              "sampling_boundary": "Counts include replacement draws; they are not unique-target coverage or equal FLOPs to original full supervision"},
        "selection": {"protocol": base["protocol"], "fold": 0, "seed": 42, "initial_validation": True,
                      "every_epoch_full_validation": True, "strict_best_auc": True, "early_stop_min_delta": 0.0,
                      "patience": 3, "ties_keep_earlier_checkpoint": True,
                      "final_best_checkpoint_reload_and_export": True,
                      "minimum_epochs_override": None, "hyperparameter_search_trials_per_recipe_dataset": 1},
        "baseline_reference": baseline,
        "configuration_mapping": {
            "source": "The completed baseline's pinned resolved_config.json, rather than a reconstructed model config",
            "removed_runtime_binding_fields": sorted(REMOVED_FIELDS),
            "added_training_identity_fields": sorted(ADDED_FIELDS),
            "modified_existing_fields": {key: {"baseline": base[key], "B008": runs[0]["config"][key]}
                                         for key in sorted(CHANGED_FIELDS)},
            "all_other_fields_identical_to_baseline": True,
            "model_and_optimizer_configuration_identical_to_baseline": True,
            "interaction_recipe": None,
            "runtime_binding_policy": "The CLI injects B008 batch_id and canonical approved_batch_sha256 at execution; old authorization fields are not copied",
            "validation_execution": "Original per-target shared development validation; no prefix-group execution"},
        "acceptance_evidence": acceptance, "data_preflight": preflight,
        "training_comparison_boundary": {
            "direct_pair": "Long and short use the same eligible targets, epoch draws, n_i/K weights, initialization settings and optimizer/microbatch budget; only training history differs",
            "loss": "sum((n_i/K)*BCEWithLogits(target_logit,label))/sum_i(n_i), with one global optimizer-batch denominator",
            "original_supervised_event_fragments_per_epoch": 182640,
            "strict_eligible_unique_targets": 172169,
            "strict_eligible_chunks": 2915, "original_training_chunks": 3016,
            "baseline_role": "Original full-supervision practical anchor, with different supervision coverage, training steps and compute; not a pure history-only control",
            "cross_source_boundary": "Original B003-R1 and B008 retain separate real manifests and implementation identities. S010 supplies scoped synthetic compatibility evidence, not identical GPU trajectories or equal training identity",
            "scientific_scope": "One development fold and seed; a training adaptation, not established novelty, paper reproduction, final-test performance or SOTA",
            "negative_result_boundary": "Sparse sampled supervision and patience3 may undertrain; a negative result does not disprove full-supervision history alignment"},
        "analysis_freeze_requirements": {
            "fixed_direct_analysis_groups": [runs[0]["run_id"], runs[1]["run_id"], BASELINE_ID],
            "before_launch": "After both B007 runs and their saved-result analysis complete, register the strongest completed comparable AS09 reference and freeze the concrete saved-prediction comparisons. Include any stronger B007 variant alongside the fixed three groups",
            "B007_use": "Terminal B007 results may identify an additional comparison reference; they must not change these B008 models, K, optimization settings or execution order",
            "pairing": "Verify each real batch/source identity and exact validation targets, allowed histories and labels; never create a fictitious merged batch manifest",
            "completed_model_retests": False,
            "sampling_audit_check": "Match every actually completed training epoch's recorded sampler receipt/counts to the corresponding frozen preflight epoch before interpreting saved results"},
        "execution_dependencies": [
            "Both B007 runs and the fixed saved-artifact B007 analysis must finish before B008 launch; preserve any failure and resolve it explicitly",
            "Verify actual worker/child identity and GPU idleness; do not overlap B008 with B007 or any other training",
            "Freeze and register the strongest completed comparable AS09 reference and exact analysis plan before launch, without changing B008 parameters based on B007 outcomes",
            "Recheck this exact source, manifest, preflight evidence and actual prepared-file integrity; bind a new launch receipt to the existing standing human authorization",
            "Run long K8 then short K8; stop the entire batch after any noncompleted run, failure, OOM or exhausted budget; no automatic retry, truncation or resizing",
        ],
        "data_safety": {"generator_reads_student_CSVs": False, "generator_reads_checkpoints": False,
                        "generator_constructs_models_or_optimizers": False,
                        "training_folds": [1, 2, 3, 4], "held_out_development_fold": 0,
                        "mixed_development_file_boundary": "Preflight and training index mixed development file bytes; fold0 events are excluded from training targets/history/labels",
                        "actual_preflight_scope": "Existing helper constructed two data loaders and checked bounded projections plus all 50 paired sampling receipts; no model forward or update"},
        "final_test_boundary": "Final test remains sealed. The separately frozen final-test evaluator does not automatically accept this training identity; compatible evaluation and candidate/checkpoint/analysis freeze are required before final test access",
        "preparation": {"generator": receipt(Path(__file__)), "creates_training_or_authority_record": False,
                        "input_sequence_files_read": False, "weights_read": False},
        "runs": runs,
    }
    require(len(batch["runs"]) == 2 and sum(run["config"]["max_train_steps"] for run in runs) == 9200
            and sum(run["config"]["max_wall_seconds"] for run in runs) == 57600, "Total limits differ")
    require(source_fingerprint(ROOT) == SOURCE and sha256_file(PREFLIGHT) == args.preflight_sha256,
            "Source/preflight changed during preparation")
    require("torch" not in sys.modules, "Preparation unexpectedly imported torch")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(batch, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"batch_file": str(OUTPUT), "batch_file_sha256": sha256_file(OUTPUT),
                      "canonical_batch_sha256": canonical_hash(batch), "source_sha256": SOURCE,
                      "preflight_report_sha256": args.preflight_sha256,
                      "runs": [run["run_id"] for run in runs], "status": "prepared_not_launched"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
