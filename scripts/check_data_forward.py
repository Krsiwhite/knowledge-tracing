"""Forward-only checks on real validation batches. Never computes quality metrics."""
import gc
import time

import torch
from torch.utils.data import DataLoader

from ktlab.adapters import build_model, forward_model, move_batch
from ktlab.engine import dataset_for_split, read_manifest
from ktlab.provenance import project_root, source_fingerprint, utc_now, write_json

root = project_root()
device = torch.device("cuda")
records = []
for dataset_name in ("assist2009", "assist2015", "nips_task34"):
    manifest = read_manifest({"dataset_manifest": str(root / "data/prepared" / dataset_name / "manifest.json")})
    dataset = dataset_for_split(manifest, 0, "valid")
    cpu_batch = next(iter(DataLoader(dataset, batch_size=32)))
    for name in ("akt", "simplekt", "mockt"):
        torch.manual_seed(42)
        model = build_model(name, manifest["data_config"], {"d_model": 64, "d_ff": 128, "n_blocks": 1,
                             "num_attn_heads": 4, "dropout": 0.1, "final_fc_dim": 128}, device).eval()
        batch = move_batch(cpu_batch, device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, regularizer = forward_model(model, batch)
        torch.cuda.synchronize()
        assert prediction.shape == batch["shft_rseqs"].shape
        assert bool(torch.isfinite(prediction).all())
        assert bool(((prediction >= 0) & (prediction <= 1)).all())
        records.append({"dataset": dataset_name, "model": name, "shape": list(prediction.shape),
                        "selected_targets": int(batch["smasks"].sum()), "finite": True,
                        "forward_seconds": time.perf_counter() - started,
                        "dataset_manifest_sha256": manifest["_manifest_sha256"]})
        del model, batch, prediction, regularizer
        gc.collect()
        torch.cuda.empty_cache()
    dataset.close()
result = {"checked_at": utc_now(), "source_sha256": source_fingerprint(), "device": torch.cuda.get_device_name(0),
          "optimizer_steps": 0, "test_accessed": False, "quality_metrics_computed": False, "checks": records}
write_json(root / "reports/FORWARD_CHECKS_2026-10-14.json", result)
print(f"Passed {len(records)} real-data GPU forward checks; no training or quality metrics.")
