"""Byte-preserving access to the pinned KTST mean and MHSA models.

Upstream uses absolute ``pkg.*`` imports. A module-local import function maps
only those names to a private, source-path-specific package. It never installs
``pkg`` aliases, changes sys.path, or patches global import/PyTorch functions.
The default path executes original Model.forward and its logits BCE unchanged.
An explicit interaction_recipe selects a separately identified local history
adaptation; it is never described as an import-only upstream model.

This module creates no optimizer and performs no backward or parameter update.
Callers retain responsibility for the project's batch-approval gate.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from types import ModuleType

import torch
from torch import nn


KTST_COMMIT = "e6331a6f45e597f286d1272fbb96e7bb67bcaa52"
KTST_REPOSITORY = "https://github.com/kainbr/kt_set_transformers"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VENDOR = _PROJECT_ROOT / "third_party" / "ktst"
_REAL_IMPORT = builtins.__import__
_IMPORT_LOCK = threading.RLock()
_NAMESPACES: dict[tuple[str, str], "_SourceNamespace"] = {}
_REQUIRED_FILES = {
    "LICENSE", "pkg/__init__.py", "pkg/model/__init__.py", "pkg/model/KTST.py",
    "pkg/model/utils/__init__.py", "pkg/model/utils/PositionalEncoding.py",
    "pkg/model/utils/attentions/__init__.py",
    "pkg/model/utils/attentions/AKTMonotonicAttention.py",
    "pkg/model/utils/attentions/ALiBiMonotonicAttention.py",
    "pkg/model/utils/attentions/LearnableALiBiMonotonicAttention.py",
    "pkg/utils/__init__.py", "pkg/utils/logging.py",
}
_DEFAULT_MODEL = {
    "d_model": 128,
    "nhead_tf": 4,
    "nhead_agg": 1,
    "num_layers_tf_dec": 4,
    "num_layers_tf_enc": 4,
    "num_layers_agg": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "dim_classifier": 256,
    "use_bias_emb": True,
    "use_zero_init": True,
    "use_decoder_only": False,
    "attn_variant": "learnable_alibi_monotonic_q_k",
    "bias_before_agg": False,
}
_AGGREGATIONS = {"ktst_mean": "q_mean_c", "ktst_mhsa": "self_attn_all"}


class KTSTSourceError(RuntimeError):
    """The vendored source does not match its pinned manifest."""


def verify_ktst_source(vendor_root: Path | str | None = None) -> dict:
    """Verify every vendored byte before import, including on cache hits.

    The sibling ``ktst.lock.json`` is itself included in the returned source
    fingerprint. Run approval/provenance must freeze that fingerprint too.
    """
    root = Path(vendor_root or _DEFAULT_VENDOR).resolve()
    lock_path = root.with_name(root.name + ".lock.json")
    try:
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
    except (OSError, ValueError) as exc:
        raise KTSTSourceError(f"Cannot read KTST source lock: {lock_path}") from exc
    if lock.get("commit") != KTST_COMMIT or lock.get("repository") != KTST_REPOSITORY:
        raise KTSTSourceError("KTST source lock has an unsupported repository/commit")
    files = lock.get("files", {})
    if not isinstance(files, dict) or not _REQUIRED_FILES.issubset(files):
        raise KTSTSourceError("KTST source lock is missing required model/license files")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not isinstance(expected, dict):
            raise KTSTSourceError(f"Unsafe KTST source entry: {relative}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise KTSTSourceError(f"Missing KTST source file: {relative}") from exc
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if (len(content) != expected.get("bytes")
                or hashlib.sha256(content).hexdigest() != expected.get("sha256")
                or blob != expected.get("git_blob_sha1")):
            raise KTSTSourceError(f"KTST source bytes changed: {relative}")
    return {**lock, "vendor_root": str(root), "lock_path": str(lock_path),
            "source_sha256": hashlib.sha256(lock_bytes).hexdigest()}


class _SourceNamespace:
    def __init__(self, verified: dict):
        self.root = Path(verified["vendor_root"])
        self.files = verified["files"]
        path_hash = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        self.prefix = f"_ktlab_ktst_{path_hash}_{verified['source_sha256'][:16]}"

    def load(self, logical_name: str) -> ModuleType:
        if logical_name != "pkg" and not logical_name.startswith("pkg."):
            raise ImportError("Only pinned KTST pkg modules can be loaded")
        fullname = self.prefix + logical_name[3:]
        with _IMPORT_LOCK:
            existing = sys.modules.get(fullname)
            if existing is not None:
                return existing
            relative = logical_name.replace(".", "/")
            package_file, module_file = relative + "/__init__.py", relative + ".py"
            is_package = package_file in self.files
            relative_file = package_file if is_package else module_file
            if relative_file not in self.files:
                raise ImportError(f"Module is not in the pinned KTST subset: {logical_name}")
            parent = self.load(logical_name.rpartition(".")[0]) if "." in logical_name else None
            source = self.root / relative_file
            spec = importlib.util.spec_from_file_location(
                fullname, source,
                submodule_search_locations=[str(source.parent)] if is_package else None,
            )
            if spec is None:
                raise ImportError(f"Cannot create KTST module spec: {logical_name}")
            module = importlib.util.module_from_spec(spec)
            # Only the module's builtins dictionary is changed; source bytes and
            # the interpreter-wide import function remain untouched.
            module.__dict__["__builtins__"] = {**vars(builtins), "__import__": self.import_module}
            sys.modules[fullname] = module
            try:
                exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
            except BaseException:
                sys.modules.pop(fullname, None)
                raise
            if parent is not None:
                setattr(parent, logical_name.rsplit(".", 1)[1], module)
            return module

    def import_module(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            absolute = importlib.util.resolve_name("." * level + name, globals["__package__"])
            if absolute == self.prefix or absolute.startswith(self.prefix + "."):
                name, level = "pkg" + absolute[len(self.prefix):], 0
        if level == 0 and (name == "pkg" or name.startswith("pkg.")):
            target = self.load(name)
            if fromlist:
                for item in fromlist:
                    if item != "*" and not hasattr(target, item):
                        relative = (name + "." + item).replace(".", "/")
                        if relative + ".py" in self.files or relative + "/__init__.py" in self.files:
                            self.load(name + "." + item)
                return target
            return self.load("pkg")
        return _REAL_IMPORT(name, globals, locals, fromlist, level)


def load_ktst_reference(module: str = "pkg.data.pykt", *, vendor_root=None) -> ModuleType:
    """Load an original model/data reference module without generic pkg aliases."""
    verified = verify_ktst_source(vendor_root)
    key = (verified["vendor_root"], verified["source_sha256"])
    with _IMPORT_LOCK:
        namespace = _NAMESPACES.setdefault(key, _SourceNamespace(verified))
        return namespace.load(module)


def build_ktst_model(name: str, data_config: dict, model_config: dict | None = None,
                     device: str | torch.device = "cpu", *, vendor_root=None,
                     interaction_recipe: str | None = None) -> nn.Module:
    """Construct mean/MHSA or an explicit local mean interaction variant.

    ``ktlab_model_config`` contains the complete resolved constructor arguments,
    so checkpoint reload can verify the data-derived dimensions as well.
    ASSIST2015 (num_q=0) uses the author's single dummy question vocabulary.
    """
    if name not in _AGGREGATIONS:
        raise ValueError(f"Unknown KTST model {name!r}; expected {tuple(_AGGREGATIONS)}")
    from .ktst_interaction import validate_interaction_recipe
    validate_interaction_recipe(name, interaction_recipe)
    dimensions = {"num_questions": max(1, int(data_config["num_q"])),
                  "num_concepts": int(data_config["num_c"]),
                  "max_concepts": int(data_config["max_concepts"]),
                  "max_len": int(data_config.get("maxlen", 200))}
    if int(data_config["num_q"]) < 0 or any(value < 1 for value in dimensions.values()):
        raise ValueError("KTST requires positive data dimensions (num_q may be zero)")
    config = {**_DEFAULT_MODEL, **dimensions, "aggregation": _AGGREGATIONS[name]}
    overrides = dict(model_config or {})
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"Unknown KTST constructor settings: {sorted(unknown)}")
    for key, expected in {**dimensions, "aggregation": _AGGREGATIONS[name]}.items():
        if key in overrides and overrides[key] != expected:
            raise ValueError(f"KTST {key} must agree with the dataset/model name ({expected})")
    config.update(overrides)
    if "shared" in config["attn_variant"]:
        raise ValueError("SharedTheta attention is not enabled: cross-instance singleton semantics")
    for key in ("d_model", "nhead_tf", "nhead_agg", "num_layers_tf_dec",
                "num_layers_tf_enc", "num_layers_agg", "dim_feedforward", "dim_classifier"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"KTST {key} must be a positive integer")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("KTST dropout must be in [0, 1)")
    if config["d_model"] % config["nhead_tf"] or (
        name == "ktst_mhsa" and config["d_model"] % config["nhead_agg"]
    ):
        raise ValueError("KTST embedding dimension must be divisible by attention heads")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("KTST currently supports verified CPU/CUDA-style ROCm placement")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but is not available")
    verified = verify_ktst_source(vendor_root)
    module = load_ktst_reference("pkg.model.KTST", vendor_root=vendor_root)
    if interaction_recipe is None:
        model = module.Model(**config).to(device)
    else:
        from .ktst_interaction import interaction_model_class
        model = interaction_model_class(module.Model)(interaction_recipe=interaction_recipe, **config).to(device)
    model.model_name = name
    model.ktlab_model_config = config
    model.ktlab_source_sha256 = verified["source_sha256"]
    model.ktlab_upstream_commit = KTST_COMMIT
    model.ktlab_source_root = verified["vendor_root"]
    model.ktlab_import_namespace = module.__name__.rsplit(".model.KTST", 1)[0]
    model.ktlab_import_binding_only = True
    model.ktlab_dummy_question = int(data_config["num_q"]) == 0
    if interaction_recipe is not None:
        from .ktst_interaction import interaction_source_identity
        model.ktlab_model_source_identity, model.ktlab_source_sha256 = interaction_source_identity(
            verified["source_sha256"], interaction_recipe)
        model.ktlab_import_binding_only = False
    return model


def forward_ktst(model: nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected model probabilities and logits BCE; no KC late fusion.

    This intentionally respects the caller's grad mode. All preapproval tests
    must call it inside inference_mode/no_grad; authorized training can later
    retain the original autograd graph and its original loss.
    """
    data, mask = batch["data"], batch["padding_mask"]
    if not isinstance(data, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("KTST data and padding_mask must be tensors")
    if data.dtype != torch.int64 or mask.dtype != torch.bool:
        raise TypeError("KTST needs int64 event data and a bool padding_mask")
    if data.ndim != 3 or data.shape[-1] != model.max_concepts + 2 or mask.shape != data.shape[:2]:
        raise ValueError("KTST expects [B,T,max_concepts+2] data and [B,T] padding_mask")
    if data.shape[0] < 1 or not 2 <= data.shape[1] <= model.max_len:
        raise ValueError("KTST requires a nonempty batch and 2 <= sequence length <= max_len")
    device = next(model.parameters()).device
    if data.device != device or mask.device != device:
        raise ValueError("KTST data, mask and model must be on the same device")
    if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("KTST events must be nonempty and right padded")
    if not bool(mask[:, 1:].any()):
        raise ValueError("KTST batch has no scored target after the initial events")
    if bool((data[~mask] != -1).any()):
        raise ValueError("KTST padding rows must contain only -1")
    if "score_mask" in batch:
        score_mask = batch["score_mask"]
        expected_mask = mask.clone()
        expected_mask[:, 0] = False
        if (not isinstance(score_mask, torch.Tensor) or score_mask.dtype != torch.bool
                or score_mask.device != device or not torch.equal(score_mask, expected_mask)):
            raise ValueError("KTST reference score_mask must select exactly noninitial valid events")
    for key, expected in (("labels", data[..., -1]), ("question_ids", data[..., -2]),
                          ("kc_counts", (data[..., :-2] >= 0).sum(-1))):
        if key in batch:
            value = batch[key]
            if (not isinstance(value, torch.Tensor) or value.shape != mask.shape
                    or value.device != device or not torch.equal(value[mask], expected[mask])):
                raise ValueError(f"KTST {key} must agree with the valid event tensor")
    events = data[mask]
    concepts, questions, responses = events[:, :-2], events[:, -2], events[:, -1]
    if bool(((responses < 0) | (responses > 1)).any()):
        raise ValueError("KTST valid responses must be binary")
    if bool(((questions < 0) | (questions >= model.num_questions)).any()):
        raise ValueError("KTST question ID is outside its vocabulary")
    if bool(((concepts < -1) | (concepts >= model.num_concepts)).any()) or bool((concepts == -1).all(-1).any()):
        raise ValueError("KTST events need at least one in-vocabulary concept")
    if concepts.shape[-1] > 1:
        ordered = concepts.sort(dim=-1).values
        if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
            raise ValueError("KTST concept sets must be deduplicated before forward")
    probabilities, original_loss, _ = model(data=data, padding_mask=mask)
    return probabilities, original_loss


def optimizer_parameter_groups(model: nn.Module, lr: float, theta_lr: float | None) -> list[dict]:
    """Describe author's Adam learning-rate groups, without creating an optimizer.

    Caller supplies weight_decay/other Adam settings once at optimizer creation.
    Thetas and other parameters both inherit those settings, as in train.py.
    A learnable attention must have an explicit theta_lr to avoid silently
    replacing the author's higher-rate parameter group with the base rate.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("KTST base learning rate must be finite and positive")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    theta = [p for n, p in named if n.endswith("thetas")]
    other = [p for n, p in named if not n.endswith("thetas")]
    if theta:
        if theta_lr is None or not math.isfinite(theta_lr) or theta_lr <= 0:
            raise ValueError("KTST learnable ALiBi requires an explicit positive theta_lr")
        return [{"params": theta, "lr": float(theta_lr)}, {"params": other, "lr": float(lr)}]
    return [{"params": other, "lr": float(lr)}]
