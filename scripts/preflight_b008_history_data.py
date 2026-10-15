"""One-shot, explicitly frozen B008 data-adapter preflight; never a model run.

Prepare only: root must bind this file/source/receipts in a plan before invoking.
Only the two development CSVs below may be opened. A failed attempt is terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "knowledge-tracing/data/prepared/assist2009"
REPORT = ROOT / "reports/B008_HISTORY_DATA_PREFLIGHT_2026-10-15.json"
ATTEMPT = ROOT / ".runtime/B008-history-data-preflight-attempt.json"
ID = "B008-HISTORY-DATA-PREFLIGHT-2026-10-15"
RECIPES = ["sampled_chunk_prefix_v1", "sampled_complete_suffix_v1"]
INPUTS = {"dataset_manifest": DATA / "manifest.json",
          "census": ROOT / "reports/KTST_TRAINING_CONTEXT_COVERAGE_2026-10-15.json",
          "train_valid.csv": DATA / "train_valid.csv",
          "train_valid_sequences.csv": DATA / "train_valid_sequences.csv"}
LIMITS = {"max_wall_seconds": 180, "cpu_threads": 1, "dataset_instances": 2,
          "max_chunk_getitem_calls": 64, "max_target_contexts": 512,
          "model_instances": 0, "forward_calls": 0, "gradient_calls": 0,
          "optimizer_constructions": 0, "optimizer_steps": 0, "max_attempts": 1}
AUTHORIZATION = "启动B003，但每次最多50轮，3次未见则早停。之后的执行不需要我再批准"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_sha():
    paths = []
    for folder in (ROOT / "src", ROOT / "third_party/pykt/pykt", ROOT / "third_party/ktst"):
        paths.extend(folder.rglob("*.py"))
    paths.append(ROOT / "pyproject.toml")
    paths.extend((ROOT / "third_party").glob("*.lock.json"))
    return digest(canonical({p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted(paths) if p.is_file()}))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def selected_indices(dataset):
    """Deterministic source-index selection; no target responses are examined."""
    n = len(dataset)
    selected = {0, n - 1}
    by_fold = {}
    for index, descriptor in enumerate(dataset._chunks):
        fold = dataset.base._chunks[descriptor.source_index].fold
        by_fold.setdefault(fold, []).append(index)
    require(set(by_fold) == {1, 2, 3, 4}, "Eligible training-fold coverage differs")
    for indexes in by_fold.values():
        selected.update((indexes[0], indexes[-1]))
    # Spread remaining choices over the whole canonical index, after fold ends.
    for index in (round(i * (n - 1) / 63) for i in range(64)):
        if len(selected) == 32:
            break
        selected.add(index)
    require(len(selected) == 32, "The fixed projection selection needs 32 distinct chunks")
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True, help="Externally frozen SHA256 of exact plan bytes")
    args = parser.parse_args()
    started = time.monotonic()
    plan_path = args.plan.resolve()
    require(plan_path.suffix == ".json" and plan_path.is_relative_to(ROOT) and not plan_path.is_relative_to(DATA),
            "Preflight plan must be a worktree JSON control file")
    plan_bytes = plan_path.read_bytes()
    require(digest(plan_bytes) == args.plan_sha256, "Plan bytes differ from supplied frozen hash")
    plan = json.loads(plan_bytes)
    require(plan.get("schema_version") == 1 and plan.get("preflight_id") == ID, "Wrong preflight schema/identity")
    require(plan.get("helper_sha256") == file_sha(__file__), "Helper differs from frozen plan")
    require(plan.get("dataset") == "assist2009" and plan.get("held_out_fold") == 0
            and plan.get("seed") == 42 and plan.get("samples_per_chunk") == 8
            and plan.get("recipes") == RECIPES and plan.get("epochs") == list(range(1, 51)), "Wrong B008 data contract")
    require(plan.get("expected") == {"eligible_chunks": 2915, "eligible_unique_targets": 172169}, "Wrong census expectations")
    require(plan.get("limits") == LIMITS and plan.get("automatic_retry") is False, "Wrong preflight execution budget")
    require(plan.get("authorization_kind") == "existing_direct_user_standing_authorization"
            and plan.get("user_confirmation") == AUTHORIZATION, "Missing actual standing authorization basis")
    require(plan.get("report_path") == str(REPORT) and plan.get("attempt_path") == str(ATTEMPT), "Preflight output paths are fixed")
    require(set(plan["inputs"]) == set(INPUTS), "Preflight input allowlist differs")
    for key, path in INPUTS.items():
        require(plan["inputs"][key]["path"] == str(path), "Preflight input placement differs")
    require(not REPORT.exists(), "Preflight report exists; no overwrite/retry")
    counts = {"dataset_instances": 0, "sampling_audit_calls": 0, "chunk_getitem_calls": 0,
              "target_contexts": 0, "model_instances": 0, "forward_calls": 0, "gradient_calls": 0,
              "optimizer_constructions": 0, "optimizer_steps": 0, "forbidden_data_open_attempts": 0}
    identity = {"schema_version": 1, "preflight_id": ID, "plan_path": str(plan_path),
                "plan_file_sha256": digest(plan_bytes), "plan_canonical_sha256": digest(canonical(plan)),
                "helper_sha256": plan["helper_sha256"], "source_sha256": plan["source_sha256"],
                "started_at": datetime.now(timezone.utc).isoformat(), "automatic_retry_allowed": False}
    exclusive_json(ATTEMPT, {**identity, "status": "started"})
    attempt_start_sha = file_sha(ATTEMPT)
    stage, datasets = "input_and_source_receipts", []
    result = {**identity, "status": "failed", "inputs": plan["inputs"], "limits": LIMITS}

    def deadline():
        if time.monotonic() - started >= 180:
            raise TimeoutError("Preflight time budget exhausted")

    def timeout(*unused):
        raise TimeoutError("Preflight time budget exhausted")

    # Catch unexpected data/test/weight access even through indirect Python I/O.
    allowed_data = {p.resolve() for k, p in INPUTS.items() if k != "census"}

    def open_guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if ((path.suffix.lower() == ".csv" or path.is_relative_to(DATA.parent.parent)) and path not in allowed_data
                or path.suffix.lower() in {".pt", ".ckpt", ".safetensors"}):
            counts["forbidden_data_open_attempts"] += 1
            raise PermissionError("Preflight rejects nonallowlisted data and weight access")
        if path in allowed_data and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise PermissionError("Preflight data inputs are read-only")

    sys.addaudithook(open_guard)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 180 - (time.monotonic() - started)))
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[name] = ""
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        require(sys.platform == "linux", "The frozen one-shot preflight uses Linux alarm/CPU limits")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

        def verify():
            deadline()
            require(file_sha(plan_path) == identity["plan_file_sha256"] and file_sha(__file__) == identity["helper_sha256"], "Control receipt changed")
            require(source_sha() == identity["source_sha256"], "Source fingerprint changed")
            for key, path in INPUTS.items():
                expected = plan["inputs"][key]
                require(path.stat().st_size == expected["bytes"] and file_sha(path) == expected["sha256"], "Input receipt changed")

        verify()
        manifest = json.loads(INPUTS["dataset_manifest"].read_bytes())
        census = json.loads(INPUTS["census"].read_bytes())
        require(census["status"] == "completed", "Prior census must be complete")
        prior, = [entry for entry in census["datasets"] if entry["dataset"] == "assist2009"]
        require(prior["manifest_sha256"] == plan["inputs"]["dataset_manifest"]["sha256"], "Census manifest differs")
        require(manifest["dataset"] == "assist2009" and Path(manifest["data_dir"]).resolve() == DATA, "Prepared dataset identity differs")
        for name in ("train_valid.csv", "train_valid_sequences.csv"):
            expected = {k: plan["inputs"][name][k] for k in ("sha256", "bytes")}
            require(manifest["files"][name] == expected and prior["source_receipts"][name] == expected, "Pinned development receipts differ")
        require({k: prior["primary"][k] for k in plan["expected"]} == plan["expected"], "Census target policy/counts differ")
        stage = "import_data_adapters"
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)

        def forbid(kind):
            def blocked(*unused, **ignored):
                counts[kind] += 1
                raise RuntimeError("Data preflight forbids all model/forward/gradient/optimizer activity")
            return blocked

        torch.nn.Module.__init__ = forbid("model_instances")
        torch.nn.Module._call_impl = forbid("forward_calls")
        torch.autograd.grad = torch.autograd.backward = torch.Tensor.backward = forbid("gradient_calls")
        torch.optim.Optimizer.__init__ = forbid("optimizer_constructions")
        torch.load = forbid("forbidden_data_open_attempts")
        from ktlab.ktst_history_training import SampledKTSTHistoryDataset
        from ktlab.shared_adapters import collate_shared_ktst
        from ktlab.shared_events import context_digests
        import ktlab.ktst_history_training as implementation
        require(Path(implementation.__file__).resolve() == ROOT / "src/ktlab/ktst_history_training.py", "Wrong worktree adapter imported")
        stage = "construct_two_training_adapters"
        for recipe in RECIPES:
            deadline()
            require(counts["dataset_instances"] < 2, "Dataset construction budget exhausted")
            counts["dataset_instances"] += 1
            datasets.append(SampledKTSTHistoryDataset(manifest, 0, recipe=recipe, samples_per_chunk=8, seed=42))
        short, long = datasets
        for dataset in datasets:
            require(len(dataset) == 2915 and dataset.audit["eligible_unique_targets"] == 172169, "Adapter eligibility differs from census")
            for key in ("excluded_chunk_first", "excluded_partial_query_after_first", "excluded_partial_history_after_first_and_complete_query"):
                require(dataset.audit[key] == prior["primary"][key], "Adapter exclusion counts differ from census")
        require(short.audit["eligible_target_index_sha256"] == long.audit["eligible_target_index_sha256"], "Recipe target indexes differ")
        stage, per_epoch = "compare_50_sampling_plans", []
        fields = ("epoch", "sampling_sha256", "eligible_chunks", "sampled_rows", "unique_targets_sampled")
        for epoch in range(1, 51):
            snapshots = []
            for dataset in datasets:
                deadline()
                dataset.set_epoch(epoch)
                snapshots.append(dataset.epoch_sampling_audit())
                counts["sampling_audit_calls"] += 1
            require(snapshots[0] == snapshots[1], "Paired epoch sampling differs")
            require(snapshots[0]["eligible_chunks"] == 2915 and snapshots[0]["sampled_rows"] == 23320, "Sampling coverage differs")
            per_epoch.append({key: snapshots[0][key] for key in fields})
        stage = "project_fixed_small_chunk_sample"
        for dataset in datasets:
            dataset.set_epoch(1)
        indexes = selected_indices(short)
        require(selected_indices(long) == indexes, "Paired fixed projection selection differs")
        projection_hash = hashlib.sha256()
        expanded, added_events, added_kcs = 0, 0, 0
        for index in indexes:
            pair = []
            for dataset in datasets:
                deadline()
                require(counts["chunk_getitem_calls"] < 64 and counts["target_contexts"] + 8 <= 512, "Projection budget exhausted before access")
                counts["chunk_getitem_calls"] += 1
                counts["target_contexts"] += 8
                item = dataset[index]
                require(len(item["contexts"]) == 8, "Unexpected sample count")
                descriptor = dataset._chunks[index]
                source_chunk = dataset.base._chunks[descriptor.source_index]
                original_events, original_prefix = dataset._student(source_chunk.uid)
                require(torch.equal(item["weights"], torch.full((8,), descriptor.n_eligible / 8, dtype=torch.float64))
                        and bool(((item["labels"] == 0) | (item["labels"] == 1)).all()), "Sample weights or binary labels differ from contract")
                pair.append(item)
                projected = collate_shared_ktst(item["contexts"], dataset.max_concepts)
                for row, context in enumerate(item["contexts"]):
                    history, query = context["history"], context["query"]
                    start, ordinal = context["history_start_index"], context["target_index"]
                    require(descriptor.first_target <= ordinal < descriptor.first_target + descriptor.n_eligible
                            and context["uid"] == source_chunk.uid and context["fold"] == source_chunk.fold,
                            "Sampled target falls outside its eligible source chunk")
                    require(query == {"question": original_events[ordinal].question,
                                      "concepts": list(original_events[ordinal].concepts)}
                            and int(item["labels"][row]) == original_events[ordinal].response,
                            "Sampled query/label is not its original event")
                    require(0 <= start < ordinal and len(history) == ordinal - start
                            and context["event_id"] == f"assist2009:{source_chunk.uid}:{ordinal}", "History ordinal bounds differ")
                    if dataset.recipe == RECIPES[0]:
                        require(start == descriptor.first_event, "Short history does not start at its original complete chunk event")
                    else:
                        require(start == 0 or original_prefix[ordinal + 1] - original_prefix[start - 1] > 200,
                                "Long history is not the maximal complete suffix")
                    require("label" not in context and "response" not in query, "Current label entered context")
                    require(context["fold"] in {1, 2, 3, 4} and history, "Invalid training history membership")
                    require(all(context[k] == v for k, v in context_digests(context).items()), "Context digest changed")
                    position = int(projected["target_positions"][row])
                    require(position == len(history) and int(projected["padding_mask"][row].sum()) == position + 1, "Projection event positions differ")
                    require(projected["history_event_ids"][row] == [e["event_id"] for e in history], "Projection event order differs")
                    for j, event in enumerate(history, start):
                        canonical_event = original_events[j]
                        require(event == {"event_id": f"assist2009:{source_chunk.uid}:{j}",
                                          "question": canonical_event.question, "concepts": list(canonical_event.concepts),
                                          "response": canonical_event.response}, "History is not the contiguous original event interval")
                    for j, event in enumerate(history + [{**query, "response": 0}]):
                        actual = projected["data"][row, j].tolist()
                        require(([c for c in actual[:-2] if c >= 0], actual[-2], actual[-1])
                                == (event["concepts"], event["question"], event["response"]), "Projection history/query values differ")
                    require(bool((projected["data"][row, position + 1:] == -1).all()), "Padding contains content")
                    projection_hash.update(canonical([context["target_sha256"], context["context_sha256"], projected["data"][row].tolist()]))
            left, right = pair
            require(torch.equal(left["labels"], right["labels"]) and torch.equal(left["weights"], right["weights"])
                    and left["sampling_sha256"] == right["sampling_sha256"], "Paired labels/weights/sample identities differ")
            for a, b in zip(left["contexts"], right["contexts"]):
                require(a["event_id"] == b["event_id"] and a["target_sha256"] == b["target_sha256"] and a["query"] == b["query"], "Paired targets differ")
                extra = len(b["history"]) - len(a["history"])
                require(extra >= 0 and b["history"][extra:] == a["history"], "Complete suffix does not extend the original allowed prefix")
                require(b["context_kc_count"] <= 200, "KC200 budget exceeded")
                expanded += extra > 0
                added_events += extra
                added_kcs += b["history_kc_count"] - a["history_kc_count"]
        stage = "final_receipt_verification"
        for dataset in datasets:
            dataset.verify_source()
        verify()
        require(all(counts[key] == 0 for key in ("model_instances", "forward_calls", "gradient_calls", "optimizer_constructions", "optimizer_steps", "forbidden_data_open_attempts")), "Forbidden activity detected")
        result.update(status="completed", dataset="assist2009", held_out_fold=0, seed=42, samples_per_chunk=8,
                      recipes=RECIPES, eligibility={**plan["expected"], "matched_census": True,
                          "eligible_target_index_sha256": short.audit["eligible_target_index_sha256"]},
                      sampling={"epochs": list(range(1, 51)), "per_epoch": per_epoch,
                          "all_recipes_identical": True, "combined_sampling_sha256": digest(canonical(per_epoch))},
                      projection={"checked": True, "paired_chunks": 32, "chunk_getitem_calls": 64,
                          "target_contexts": 512, "paired_sampled_targets": 256, "training_folds": [1, 2, 3, 4],
                          "includes_first_and_last": True, "selected_chunk_indices_sha256": digest(canonical(indexes)),
                          "data_projection_sha256": projection_hash.hexdigest(), "all_pair_targets_labels_weights_equal": True,
                          "all_histories_legal": True, "all_target_response_slots_zero": True,
                          "expanded_history_sampled_targets": expanded, "added_history_events": added_events,
                          "added_history_kcs": added_kcs, "sample_scope": "32 source-index-selected chunks, epoch1; not a performance estimate"},
                      receipts_unchanged=True, cpu_threads=1, cpu_affinity_count=len(os.sched_getaffinity(0)),
                      test_accessed=False, weights_accessed=False,
                      data_scope="Both constructors index mixed development originals; only training folds1-4 supply sampled targets/history/labels")
    except Exception as error:
        # Upstream validation errors can contain a student ID. Never export it.
        result.update(status="failed", failure_stage=stage, error_type=type(error).__name__,
                      error_detail="Omitted to avoid exposing raw development identifiers; no automatic retry")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for dataset in datasets:
            dataset.close()
        result.update(actual_counts=counts, seconds=time.monotonic() - started,
                      finished_at=datetime.now(timezone.utc).isoformat(), attempt_start_sha256=attempt_start_sha)
        exclusive_json(REPORT, result)
        terminal = {**identity, "status": result["status"], "report_path": str(REPORT),
                    "report_sha256": file_sha(REPORT), "attempt_start_sha256": attempt_start_sha,
                    "seconds": result["seconds"], "actual_counts": counts}
        pending = ATTEMPT.with_suffix(".terminal.json")
        exclusive_json(pending, terminal)
        pending.replace(ATTEMPT)
    print(json.dumps({"status": result["status"], "report_path": str(REPORT),
                      "report_sha256": file_sha(REPORT), "seconds": result["seconds"], "actual_counts": counts}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
