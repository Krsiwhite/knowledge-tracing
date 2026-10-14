"""Pinned UKT author/pyKT inference audit on synthetic CPU inputs only.

No upstream rewrite, checkpoint load, student data, gradient, or optimizer is used.
The output is a bounded forward-property audit, never a quality evaluation.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import types

import torch


SOURCES = {
    "author": {
        "commit": "86ad9d8f76ee2c761e116fcadc43705f66ffd0b6",
        "url": "https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py",
        "files": {"ukt.py": "8b092537174bbe459d4d2e93333d66b16c77bd94456b2b06dc549af6d1183651",
                  "utils.py": "b7967a4897742af0a581b0cfb0206776c40ce4605c5c69656e1f1b342ace731e"},
    },
    "pykt": {
        "commit": "eec087e0230520e7ae7bbd4f5433d71612df62fb",
        "url": "https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/ukt.py",
        "files": {"ukt.py": "edf78e409e19d7aaf70d8450bab8e10254928cd72f224bac1974b12a0e7c4d55",
                  "utils.py": "7e98c9dbd660ad367e4b8798ff7195f9c48d96429695362771c8abebc3ca3c37"},
    },
}
ATOL = 1e-6


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(name, directory):
    actual = {filename: sha256(directory / filename) for filename in SOURCES[name]["files"]}
    if actual != SOURCES[name]["files"]:
        raise ValueError(f"Pinned {name} source bytes differ: {actual}")
    return actual


def isolated_import(name, directory):
    namespace = f"_ukt_causality_audit_{name}"
    if namespace in sys.modules:
        raise RuntimeError("Audit namespace already loaded; use a fresh process")
    package = types.ModuleType(namespace)
    package.__path__ = [str(directory.resolve())]
    sys.modules[namespace] = package
    spec = importlib.util.spec_from_file_location(namespace + ".ukt", directory / "ukt.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WithoutDocstrings(ast.NodeTransformer):
    def strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        return node
    visit_Module = strip
    visit_ClassDef = strip
    visit_FunctionDef = strip


def effective_nodes(path):
    """Last top-level definition wins, matching Python name binding."""
    tree = WithoutDocstrings().visit(ast.parse(path.read_text(encoding="utf-8")))
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            result[node.name] = node
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        result[node.name + "." + method.name] = method
    return result


def source_comparison(author, pykt):
    a, b = effective_nodes(author / "ukt.py"), effective_nodes(pykt / "ukt.py")
    names = ["UKT.base_emb", "UKT.forward", "Architecture", "TransformerLayer", "MultiHeadAttention",
             "attention", "uattention", "wasserstein_distance_matmul", "WassersteinNCELoss",
             "CosinePositionalEmbedding"]
    result = {name: ast.dump(a[name]) == ast.dump(b[name]) for name in names}
    result["UKT.__init__.body"] = [ast.dump(n) for n in a["UKT.__init__"].body] == [ast.dump(n) for n in b["UKT.__init__"].body]
    result["utils_entire_ast"] = ast.dump(ast.parse((author / "utils.py").read_text())) == ast.dump(ast.parse((pykt / "utils.py").read_text()))
    return {"effective_ast_equal_ignoring_docstrings": result,
            "constructor_signatures_equal": ast.dump(a["UKT.__init__"].args) == ast.dump(b["UKT.__init__"].args),
            "interpretation": "This structural comparison covers named definitions, not complete training/data pipelines."}


def input_dict(responses, *, concepts=None, valid_length=None):
    responses = torch.as_tensor(responses, dtype=torch.long).reshape(1, -1)
    if concepts is None:
        concepts = list(range(responses.shape[1]))
    concepts = torch.as_tensor(concepts, dtype=torch.long).reshape(1, -1)
    if concepts.shape != responses.shape or responses.shape[1] < 2:
        raise ValueError("Synthetic sequence shape mismatch")
    length = responses.shape[1]
    valid_length = length if valid_length is None else valid_length
    valid = torch.arange(length).unsqueeze(0) < valid_length
    masks = valid[:, :-1] & valid[:, 1:]
    result = {"cseqs": concepts[:, :-1], "shft_cseqs": concepts[:, 1:],
              "qseqs": concepts[:, :-1], "shft_qseqs": concepts[:, 1:],
              "rseqs": responses[:, :-1], "shft_rseqs": responses[:, 1:],
              "masks": masks, "smasks": masks.clone()}
    assert torch.equal(torch.cat((result["rseqs"][:, :1], result["shft_rseqs"]), 1), responses)
    assert torch.equal(torch.cat((result["cseqs"][:, :1], result["shft_cseqs"]), 1), concepts)
    return result


def predict(model, data):
    before = {key: value.clone() for key, value in data.items()}
    output = model(data, train=False)
    expected = (1, data["cseqs"].shape[1] + 1)
    if output.shape != expected or not bool(torch.isfinite(output).all()) or not bool(((output >= 0) & (output <= 1)).all()):
        raise ValueError("Invalid probability output")
    if output.device.type != "cpu" or output.requires_grad:
        raise ValueError("Audit must remain inference-only on CPU")
    if any(not torch.equal(before[key], data[key]) for key in before):
        raise ValueError("Forward mutated its input")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Unexpected parameter gradient")
    return output


def record(records, identity, case, reference, changed, indices, *, relation="invariant", changed_position=None):
    left, right = reference[0, indices], changed[0, indices]
    maximum = (left - right).abs().max().item()
    records.append({**identity, "case": case, "expected_relation": relation,
                    "scored_positions_zero_based": indices, "changed_response_position_zero_based": changed_position,
                    "reference_probabilities": left.tolist(), "compared_probabilities": right.tolist(),
                    "max_absolute_change": maximum, "bitwise_equal": torch.equal(left, right),
                    "within_atol": maximum <= ATOL,
                    "outcome": ("observed_sensitive" if maximum > ATOL else "inconclusive_positive_control")
                    if relation == "history_positive_control" else ("within_tolerance" if maximum <= ATOL else "counterexample")})


def exercise(model, identity, records):
    answers = [0, 1, 0, 1, 0, 1, 0, 1]
    baseline = predict(model, input_dict(answers))
    repeat = predict(model, input_dict(answers))
    record(records, identity, "identical_input_repeat", baseline, repeat, list(range(8)))
    for position, case, scored, relation in [
        (1, "history_answer", [2], "history_positive_control"),
        (2, "current_answer", [0, 1, 2], "invariant"),
        (5, "future_answer", [0, 1, 2], "invariant"),
        (0, "empty_history_current_answer", [0], "invariant"),
        (7, "last_position_current_answer", [7], "invariant"),
    ]:
        changed = answers.copy()
        changed[position] = 1 - changed[position]
        output = predict(model, input_dict(changed))
        record(records, identity, case, baseline, output, scored, relation=relation, changed_position=position)
    changed = answers[:3] + [1 - r for r in answers[3:]]
    record(records, identity, "all_future_answers", baseline, predict(model, input_dict(changed)), [0, 1, 2])
    changed_concepts = [0, 1, 2, 11, 10, 9, 8, 7]
    record(records, identity, "future_question_and_concept_ids", baseline,
           predict(model, input_dict(answers, concepts=changed_concepts)), [0, 1, 2])
    prefix = predict(model, input_dict(answers[:3]))
    record(records, identity, "full_vs_prefix_through_target", baseline, prefix, [0, 1, 2])
    dummy_answers = answers[:3]
    dummy_answers[2] = 1  # Original target is 0; exercise the other dummy value explicitly.
    prefix_dummy = predict(model, input_dict(dummy_answers))
    record(records, identity, "prefix_current_dummy_zero_vs_one", prefix, prefix_dummy, [0, 1, 2], changed_position=2)
    padded = predict(model, input_dict(answers + [0] * 192, concepts=list(range(8)) + [0] * 192, valid_length=8))
    record(records, identity, "full8_vs_right_padded200", baseline, padded, list(range(8)))
    changed_padding = predict(model, input_dict(answers + [1] * 192, concepts=list(range(8)) + [11] * 192, valid_length=8))
    record(records, identity, "right_padding_payload", padded, changed_padding, list(range(8)))
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-source-dir", type=Path, required=True)
    parser.add_argument("--pykt-source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit report already exists")
    directories = {"author": args.author_source_dir, "pykt": args.pykt_source_dir}
    before_hashes = {name: verify_source(name, directory) for name, directory in directories.items()}
    attempts = {"backward_or_autograd": 0, "optimizer_construction": 0}
    def deny_gradient(*args, **kwargs):
        attempts["backward_or_autograd"] += 1
        raise RuntimeError("Gradient computation is forbidden in this audit")
    def deny_optimizer(*args, **kwargs):
        attempts["optimizer_construction"] += 1
        raise RuntimeError("Optimizer construction is forbidden in this audit")
    torch.Tensor.backward = deny_gradient
    torch.autograd.backward = deny_gradient
    torch.autograd.grad = deny_gradient
    torch.optim.Optimizer.__init__ = deny_optimizer
    torch.cuda.is_available = lambda: False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    modules = {name: isolated_import(name, directory) for name, directory in directories.items()}
    common = {"n_question": 12, "d_model": 32, "n_blocks": 2, "dropout": 0., "d_ff": 64,
              "final_fc_dim": 64, "final_fc_dim2": 32, "seq_len": 200, "num_attn_heads": 4,
              "emb_type": "stoc_qid", "use_CL": True, "use_uncertainty_aug": True, "separate_qa": False}
    configurations = {"w2_concept_only": {"n_pid": 0, "atten_type": "w2"},
                      "w2_question_ids": {"n_pid": 12, "atten_type": "w2"},
                      "dp_concept_only": {"n_pid": 0, "atten_type": "dp"}}
    records, cross_source = [], []
    with torch.inference_mode():
        for config_name, options in configurations.items():
            for seed in (42, 2026, 3407):
                snapshots, outputs = {}, {}
                for name, module in modules.items():
                    torch.manual_seed(seed)
                    model = module.UKT(**common, **options).cpu().eval()
                    if any(p.device.type != "cpu" for p in model.parameters()):
                        raise ValueError("Non-CPU parameter")
                    identity = {"source": name, "configuration": config_name, "seed": seed}
                    outputs[name] = exercise(model, identity, records)
                    snapshots[name] = {key: value.clone() for key, value in model.state_dict().items()}
                state_equal = snapshots["author"].keys() == snapshots["pykt"].keys() and all(
                    torch.equal(value, snapshots["pykt"][key]) for key, value in snapshots["author"].items())
                a = [r for r in records if r["source"] == "author" and r["configuration"] == config_name and r["seed"] == seed]
                b = [r for r in records if r["source"] == "pykt" and r["configuration"] == config_name and r["seed"] == seed]
                cross_source.append({"configuration": config_name, "seed": seed,
                                     "same_seed_state_dict_bitwise_equal": state_equal,
                                     "all_compared_probabilities_equal": all(x["case"] == y["case"] and
                                         x["reference_probabilities"] == y["reference_probabilities"] and
                                         x["compared_probabilities"] == y["compared_probabilities"] for x, y in zip(a, b)) and len(a) == len(b),
                                     "baseline_max_absolute_change": (outputs["author"] - outputs["pykt"]).abs().max().item()})
    after_hashes = {name: verify_source(name, directory) for name, directory in directories.items()}
    invariant = [r for r in records if r["expected_relation"] == "invariant"]
    positive = [r for r in records if r["expected_relation"] == "history_positive_control"]
    report = {"schema": "ukt-synthetic-causality-audit-v1", "sources": SOURCES,
              "source_sha256_before": before_hashes, "source_sha256_after": after_hashes,
              "source_comparison": source_comparison(args.author_source_dir, args.pykt_source_dir),
              "audit_script_sha256": sha256(Path(__file__)), "device": "cpu", "torch": torch.__version__,
              "python": platform.python_version(), "dtype": "float32", "absolute_tolerance": ATOL,
              "random_untrained_weights": True, "source_math_modified": False,
              "source_files_modified": False, "optimizer_steps": 0, "forbidden_call_attempts": attempts,
              "student_data_accessed": False, "checkpoint_loaded": False, "test_accessed": False,
              "quality_metrics_computed": False, "train_argument": False, "model_mode": "eval",
              "common_model_config": common, "configurations": configurations,
              "synthetic_concepts": list(range(8)), "synthetic_responses": [0, 1, 0, 1, 0, 1, 0, 1],
              "summary": {"config_seed_source_runs": 18, "observations": len(records),
                          "invariance_checks": len(invariant), "counterexamples": sum(not r["within_atol"] for r in invariant),
                          "invariance_max_absolute_change": max(r["max_absolute_change"] for r in invariant),
                          "history_controls": len(positive), "history_sensitive_above_atol": sum(r["max_absolute_change"] > ATOL for r in positive)},
              "cross_source_checks": cross_source, "observations": records,
              "limits": ["Finite synthetic forward checks are not a proof that every configuration or training/data pipeline is leak-free.",
                         "No gradients, training losses, augmentation/CL branch, student datasets, real multi-KC mapping, or trained weights were exercised.",
                         "Question difficulty embeddings use unchanged zero initialization; nonzero learned difficulty was not exercised.",
                         "Only two blocks, the stated dimensions/seeds, separate_qa=False, eval mode, CPU float32 were exercised.",
                         "Length 200 is a right-padding boundary here, not 200 genuine interactions or a sliding-history truncation test.",
                         "Equal outputs for explicit keyword configurations do not prove identical constructor positional/default behavior or author/pyKT experiment results."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
