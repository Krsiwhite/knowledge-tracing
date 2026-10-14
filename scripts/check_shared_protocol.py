"""Audit shared development contexts and untrained GPU predictions; no metrics."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ktlab.adapters import build_model
from ktlab.ktst_model import build_ktst_model
from ktlab.provenance import project_root, read_json, sha256_file, source_fingerprint, utc_now, write_json
from ktlab.shared_adapters import audit_shared_projection, predict_shared
from ktlab.shared_events import SharedEventDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=project_root())
    parser.add_argument("--sample-targets", type=int, default=128)
    args = parser.parse_args()
    if args.sample_targets < 2:
        parser.error("--sample-targets must be at least 2")
    records = []
    for name in ("assist2009", "assist2015", "nips_task34"):
        path = args.data_root / "data/prepared" / name / "manifest.json"
        manifest = read_json(path)
        started = time.perf_counter()
        dataset = SharedEventDataset(manifest, 0, "valid")
        index_seconds = time.perf_counter() - started
        sample_indices = sorted({i * (len(dataset) - 1) // (args.sample_targets - 1)
                                 for i in range(args.sample_targets)})
        retrieval_started = time.perf_counter()
        contexts = [dataset[i] for i in sample_indices]
        retrieval_seconds = time.perf_counter() - retrieval_started
        projection = audit_shared_projection(contexts, manifest["data_config"])
        checks = []
        for model_name in ("akt", "simplekt", "mockt", "ktst_mean", "ktst_mhsa"):
            torch.manual_seed(42)
            if model_name.startswith("ktst"):
                model = build_ktst_model(model_name, manifest["data_config"], {
                    "d_model": 128, "nhead_tf": 8, "nhead_agg": 4, "num_layers_tf_enc": 1,
                    "num_layers_tf_dec": 2, "num_layers_agg": 1, "dim_feedforward": 256,
                    "dim_classifier": 256, "dropout": 0.1}, "cuda")
            else:
                model = build_model(model_name, manifest["data_config"], {
                    "d_model": 128, "n_blocks": 2, "d_ff": 256, "num_attn_heads": 8,
                    "final_fc_dim": 256, "dropout": 0.1}, "cuda")
            predict_shared(model, contexts[:2], "cuda", max_rows_per_forward=32)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            probabilities = predict_shared(model, contexts, "cuda", max_rows_per_forward=32)
            torch.cuda.synchronize()
            assert len(probabilities) == len(contexts)
            assert all(0 <= value <= 1 for value in probabilities)
            elapsed = time.perf_counter() - started
            checks.append({"model": model_name, "events": len(probabilities), "finite_in_range": True,
                           "prediction_seconds": elapsed,
                           "rough_full_forward_seconds": elapsed * len(dataset) / len(contexts),
                           "timing_scope": "warmed untrained forward plus collation; excludes dataset retrieval, hashing, metrics and export; extrapolation is not a budget guarantee",
                           "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()})
            del model, probabilities
            gc.collect()
            torch.cuda.empty_cache()
        dataset.verify_source()
        records.append({"dataset": name, "dataset_manifest_sha256": sha256_file(path),
                        "full_validation_audit": dataset.audit, "index_seconds": index_seconds,
                        "projection_sample_indices": sample_indices, "projection_sample_audit": projection,
                        "sample_retrieval_seconds": retrieval_seconds,
                        "sampling": "deterministic equal spacing across the full ordered eligible population, including both endpoints",
                        "sample_history_kc_counts": [context["history_kc_count"] for context in contexts],
                        "sample_query_kc_counts": [context["target_kc_count"] for context in contexts],
                        "untrained_model_checks": checks})
        dataset.close()
        print(json.dumps({"dataset": name, "targets": records[-1]["full_validation_audit"]["selected_targets"],
                          "model_checks": len(checks), "quality_metrics_computed": False}), flush=True)
    write_json(project_root() / "reports/SHARED_PROTOCOL_CHECKS_2026-10-14.json", {
        "checked_at": utc_now(), "implementation_sha256": source_fingerprint(),
        "optimizer_steps": 0, "test_accessed": False, "quality_metrics_computed": False,
        "checks": records})


if __name__ == "__main__":
    main()
