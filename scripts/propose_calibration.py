"""Prepare B002 from verified development data; never approve or train it."""
import argparse
import math
from pathlib import Path

from ktlab.engine import dataset_for_split, read_manifest
from ktlab.ktst_engine import PROTOCOL as KTST_PROTOCOL
from ktlab.provenance import canonical_hash, git_revision, project_root, source_fingerprint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=project_root())
    args = parser.parse_args()
    runs = []
    for dataset in ("assist2009", "assist2015", "nips_task34"):
        manifest_path = (args.data_root / "data/prepared" / dataset / "manifest.json").resolve()
        manifest = read_manifest({"dataset_manifest": str(manifest_path)})
        training = dataset_for_split(manifest, 0, "train")
        validation = dataset_for_split(manifest, 0, "valid")
        n_train, n_valid = len(training), len(validation)
        training.close()
        validation.close()
        for model in ("akt", "simplekt", "mockt", "ktst_mean"):
            config = {"dataset_manifest": str(manifest_path), "model": model, "dataset": dataset,
                      "fold": 0, "seed": 42, "device": "cuda", "batch_size": 32, "epochs": 1,
                      "learning_rate": .001, "weight_decay": 0., "shuffle": True, "grad_clip": 1.,
                      "patience": 10, "early_stop_min_delta": 0., "max_train_steps": math.ceil(n_train / 32),
                      "max_wall_seconds": 1800, "debug_smoke": True,
                      "purpose": "one_epoch_runtime_and_full_validation_calibration_not_sota"}
            config["model_config"] = {"d_model": 128, "n_blocks": 2, "dropout": .1, "d_ff": 256,
                                      "num_attn_heads": 8, "final_fc_dim": 256}
            if model == "mockt":
                config["model_config"].update(emb_type="qid_conv_ker_noexp", kernel_size=4)
            if model == "ktst_mean":
                config.update(protocol=KTST_PROTOCOL, validation_batch_size=32, learning_rate=.0001,
                              theta_learning_rate=.0005, weight_decay=.00005, initial_validation=True)
                config["model_config"] = {"d_model": 128, "nhead_tf": 8, "nhead_agg": 4,
                                          "num_layers_tf_enc": 1, "num_layers_tf_dec": 2, "num_layers_agg": 1,
                                          "dim_feedforward": 256, "dim_classifier": 256, "dropout": .1,
                                          "aggregation": "q_mean_c", "use_bias_emb": True, "use_zero_init": True,
                                          "use_decoder_only": False, "attn_variant": "learnable_alibi_monotonic_q_k"}
            runs.append({"run_id": f"B002-{model}-{dataset}-f0-s42",
                         "dataset_manifest_sha256": manifest["_manifest_sha256"],
                         "train_sequence_rows": n_train,
                         "v1_validation_rows": n_valid if model != "ktst_mean" else None,
                         "config": config})
    batch = {"schema_version": 1, "batch_id": "B002-one-epoch-calibration", "status": "proposed",
             "purpose": "Measure one epoch and full validation across three datasets/four baselines; not converged quality or SOTA.",
             "source_sha256": source_fingerprint(), "git_commit_at_preparation": git_revision(),
             "limits": {"total_optimizer_steps_upper_bound": sum(r["config"]["max_train_steps"] for r in runs),
                        "epochs_per_run": 1, "per_run_wall_seconds": 1800, "sum_run_wall_seconds": 21600,
                        "wall_limit_scope": "soft checks between batches after model/data setup; setup is additional",
                        "paid_services": False, "automatic_retries": False},
             "test_access": False, "stop_on_noncompleted_run": True,
             "comparison_boundary": "KTST uses its independent reference development protocol; no cross-track AUC ranking.",
             "runs": runs}
    output = project_root() / "configs/batches/B002-one-epoch-calibration.json"
    write_json(output, batch)
    print({"proposal": str(output), "batch_sha256": canonical_hash(batch), "runs": len(runs),
           "optimizer_steps_upper_bound": batch["limits"]["total_optimizer_steps_upper_bound"], "status": "pending_user_approval"})


if __name__ == "__main__":
    main()
