"""Forward-only synthetic audit of the pinned public KeenKT implementation.

This does not integrate KeenKT, train it, or evaluate any student dataset.
Supply the reviewed source package containing KeenKT.py and utils.py.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch


def git_blob_sha(path):
    content = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def forbid(*args, **kwargs):
    raise RuntimeError("This audit forbids gradients and optimizer construction")


def input_dict(responses):
    concepts = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    result = {"cseqs": concepts[:, :-1], "shft_cseqs": concepts[:, 1:],
              "qseqs": concepts[:, :-1], "shft_qseqs": concepts[:, 1:],
              "rseqs": responses[:, :-1], "shft_rseqs": responses[:, 1:],
              "masks": torch.ones(1, 7, dtype=torch.bool), "smasks": torch.ones(1, 7, dtype=torch.bool)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pinned = {"KeenKT.py": "181e57d6a39d0622d4a56af2be00f2a5c97e9353",
              "utils.py": "925d9527a76bd0de70e75af7170981ee1ec6fabe"}
    for name, expected in pinned.items():
        if git_blob_sha(args.source_dir / name) != expected:
            raise ValueError(f"Source bytes do not match reviewed upstream blob: {name}")
    if args.output.exists():
        raise FileExistsError("Audit report already exists")
    torch.Tensor.backward = forbid
    torch.autograd.backward = forbid
    torch.autograd.grad = forbid
    torch.optim.Optimizer.__init__ = forbid
    # Both upstream modules bind their device at import time; use CPU only.
    torch.cuda.is_available = lambda: False
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.source_dir.resolve().parent))
    module = importlib.import_module(args.source_dir.name + ".KeenKT")
    upstream_uattention = module.uattention

    def zero_first_attention_row(*call_args, **call_kwargs):
        """Diagnostic control only: honor the otherwise unused zero_pad flag."""
        outputs = upstream_uattention(*call_args, **call_kwargs)
        zero_pad = call_kwargs.get("zero_pad", call_args[9] if len(call_args) > 9 else False)
        if zero_pad:
            outputs = tuple(torch.cat((torch.zeros_like(x[:, :, :1, :]), x[:, :, 1:, :]), dim=2)
                            for x in outputs)
        return outputs

    records = []
    with torch.inference_mode():
        for seed in (42, 2026, 3407):
            torch.manual_seed(seed)
            model = module.KEENKT(n_question=12, n_pid=0, emb_type="stoc_qid", use_uncertainty_aug=True,
                                 n_blocks=2, dropout=0., d_model=32, d_ff=64, final_fc_dim=64,
                                 final_fc_dim2=32, seq_len=8, n_heads=4, use_diffusion=True, use_CL=True)
            model.eval()
            responses = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
            gate = model.se_gate
            for variant in ("upstream", "identity_gate", "zero_first_attention_row", "combined_control"):
                model.se_gate = torch.nn.Identity() if variant in ("identity_gate", "combined_control") else gate
                module.uattention = zero_first_attention_row if variant in ("zero_first_attention_row", "combined_control") else upstream_uattention
                original = model(input_dict(responses), train=False)
                repeated = model(input_dict(responses), train=False)
                if not torch.equal(original, repeated):
                    raise ValueError("Repeated identical forward is not deterministic")
                for changed_position in (1, 2, 5):
                    changed = responses.clone()
                    changed[0, changed_position] = 1 - changed[0, changed_position]
                    perturbed = model(input_dict(changed), train=False)
                    for output in (original, perturbed):
                        if output.shape != (1, 8) or not torch.isfinite(output).all():
                            raise ValueError("Invalid synthetic forward output")
                    target_position = 2
                    records.append({"seed": seed, "variant": variant,
                                    "scored_position_zero_based": target_position,
                                    "changed_response_position_zero_based": changed_position,
                                    "perturbation": {1: "history_answer_positive_control", 2: "current_answer", 5: "future_answer"}[changed_position],
                                    "original_probability": original[0, target_position].item(),
                                    "perturbed_probability": perturbed[0, target_position].item(),
                                    "absolute_change": abs((original - perturbed)[0, target_position].item()),
                                    "identical_input_repeat_max_change": (original - repeated).abs().max().item()})
            model.se_gate = gate
            module.uattention = upstream_uattention
    report = {"upstream_commit": "911e4e460b0e2ddb25142fb3c7e46123c1043690", "source_git_blob_sha1": pinned,
              "device": "cpu", "torch": torch.__version__, "sequence_length": 8, "random_untrained_weights": True,
              "source_files_modified": False,
              "upstream_variant_math_modified": False,
              "separate_controls": {
                  "identity_gate": "replace only se_gate with Identity in memory",
                  "zero_first_attention_row": "zero only the first output row of uattention when zero_pad is true",
                  "combined_control": "apply both diagnostic interventions; not an upstream reproduction"},
              "model_config": {"n_question": 12, "n_pid": 0, "emb_type": "stoc_qid", "n_blocks": 2,
                               "dropout": 0., "d_model": 32, "d_ff": 64, "final_fc_dim": 64,
                               "final_fc_dim2": 32, "seq_len": 8, "n_heads": 4, "train": False},
              "synthetic_concepts": list(range(8)), "synthetic_responses": [0, 1, 0, 1, 0, 1, 0, 1],
              "optimizer_steps": 0, "backward_calls": 0, "student_data_accessed": False, "test_accessed": False,
              "quality_metrics_computed": False, "observations": records,
              "scope": "Counterexample for this public code snapshot and full-sequence forward; cannot attribute published paper scores to this mechanism."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(records), "max_current_or_future_change_by_variant": {
        variant: max(x["absolute_change"] for x in records if x["variant"] == variant and x["perturbation"] != "history_answer_positive_control")
        for variant in sorted({x["variant"] for x in records})}}))


if __name__ == "__main__":
    main()
