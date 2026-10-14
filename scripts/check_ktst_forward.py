"""Real development data only: GPU forward checks without quality metrics."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from ktlab.adapters import move_batch
from ktlab.engine import read_manifest
from ktlab.ktst_data import KTSTSequenceDataset, collate_ktst
from ktlab.ktst_model import build_ktst_model, forward_ktst, verify_ktst_source
from ktlab.provenance import project_root, source_fingerprint, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=project_root())
    args = parser.parse_args()
    source = verify_ktst_source()
    records = []
    for dataset_name in ("assist2009", "assist2015", "nips_task34"):
        manifest = read_manifest({"dataset_manifest": str(args.data_root / "data/prepared" / dataset_name / "manifest.json")})
        dataset = KTSTSequenceDataset(manifest, 0, "valid", max_rows=32)
        cpu = next(iter(DataLoader(dataset, batch_size=32, collate_fn=collate_ktst)))
        for name in ("ktst_mean", "ktst_mhsa"):
            torch.manual_seed(42)
            model = build_ktst_model(name, manifest["data_config"], {
                "d_model": 64, "nhead_tf": 4, "nhead_agg": 4,
                "num_layers_tf_enc": 1, "num_layers_tf_dec": 1, "num_layers_agg": 1,
                "dim_feedforward": 128, "dim_classifier": 128, "dropout": .1}, "cuda").eval()
            batch = move_batch(cpu, torch.device("cuda"))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                probabilities, loss = forward_ktst(model, batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert probabilities.shape == batch["score_mask"].shape
            assert bool(torch.isfinite(probabilities).all()) and bool(torch.isfinite(loss))
            assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
            records.append({"dataset": dataset_name, "model": name, "input_shape": list(batch["data"].shape),
                            "output_shape": list(probabilities.shape), "selected_targets": int(batch["score_mask"].sum()),
                            "finite_probabilities_and_loss": True, "forward_seconds": elapsed,
                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "dataset_manifest_sha256": manifest["_manifest_sha256"], "data_audit": dataset.audit})
            del model, batch, probabilities, loss
            gc.collect()
            torch.cuda.empty_cache()
        dataset.close()
    result = {"checked_at": utc_now(), "source_sha256": source_fingerprint(), "ktst_source": source,
              "device": torch.cuda.get_device_name(0), "optimizer_steps": 0, "test_accessed": False,
              "quality_metrics_computed": False, "checks": records}
    write_json(project_root() / "reports/KTST_FORWARD_CHECKS_2026-10-14.json", result)
    print(json.dumps({"checks": len(records), "optimizer_steps": 0, "test_accessed": False}))


if __name__ == "__main__":
    main()
