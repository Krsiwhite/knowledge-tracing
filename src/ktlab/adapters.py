"""CPU data and isolated imports for the pinned, unmodified pyKT models.

Only module-global device placement is rebound; model implementations and
mathematical operations are never patched. No pykt.models package initializer
is executed, so unrelated optional models cannot break an AKT import.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
from typing import Any

import torch
from torch.utils.data import Dataset

SUPPORTED_MODELS = {"akt": "AKT", "simplekt": "simpleKT", "mockt": "MocKT", "fa_kt": "FA_KT"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = PROJECT_ROOT / "third_party" / "pykt" / "pykt" / "models"


class ModelUnavailableError(RuntimeError):
    """A requested original model cannot run in this environment."""


def resolve_device(value: str = "auto") -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and PyTorch CUDA/ROCm devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelUnavailableError("CUDA/ROCm was requested but torch.cuda.is_available() is false")
    return device


def _load_original(name: str, device: torch.device):
    tag = hashlib.sha256(f"{UPSTREAM}:{device}".encode()).hexdigest()[:12]
    package_name = f"_ktlab_pinned_pykt_{tag}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(UPSTREAM)]
        sys.modules[package_name] = package
    full_name = f"{package_name}.{name}"
    if full_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(full_name, UPSTREAM / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ModelUnavailableError(f"Missing pinned source for {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(full_name, None)
            raise
        # Explicit placement compatibility only, including imported .utils.
        for key, imported in tuple(sys.modules.items()):
            if key.startswith(package_name + ".") and hasattr(imported, "device"):
                imported.device = device
    return sys.modules[full_name]


def build_model(name: str, data_config: dict, model_config: dict | None = None,
                device: str | torch.device = "auto") -> torch.nn.Module:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}; choose {sorted(SUPPORTED_MODELS)}")
    target = resolve_device(str(device))
    if name == "mockt" and target.type != "cuda":
        raise ModelUnavailableError("Pinned MoC-KT hard-codes CUDA tensors; its unchanged original requires CUDA/ROCm")
    if name == "mockt" and target.index not in {None, torch.cuda.current_device()}:
        raise ModelUnavailableError("Pinned MoC-KT uses the default CUDA device; select it with CUDA_VISIBLE_DEVICES")
    settings = {"d_model": 128, "n_blocks": 1, "dropout": 0.1, "d_ff": 256,
                "num_attn_heads": 4, "emb_type": "qid"}
    if name == "mockt":
        settings.update(emb_type="qid_conv_ker_noexp", kernel_size=4)
    if name == "fa_kt":
        settings.update(emb_type="qidband", num_rgap=64, num_sgap=64, num_pcount=64)
    settings.update(model_config or {})
    if name in {"simplekt", "mockt", "fa_kt"}:
        settings.setdefault("seq_len", int(data_config.get("maxlen", 200)))
    if settings["d_model"] % settings["num_attn_heads"]:
        raise ValueError("d_model must be divisible by num_attn_heads")
    if name == "fa_kt":
        emb_type = settings["emb_type"]
        needs_mamba = "onlymamba" in emb_type or (settings.get("use_moe", True) and
            not any(flag in emb_type for flag in ("nomoe", "onlyattn", "onlylstm", "onlycnn")))
        if needs_mamba and importlib.util.find_spec("mamba_ssm") is None:
            raise ModelUnavailableError("Original FA-KT requires mamba-ssm (and compatible kernels). "
                                        "No expert substitution was made; skip or configure an explicitly labelled ablation.")
    try:
        module = _load_original(name, target)
        model = getattr(module, SUPPORTED_MODELS[name])(
            n_question=int(data_config["num_c"]), n_pid=int(data_config.get("num_q", 0)), **settings)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModelUnavailableError(f"Cannot import original {name}: {exc}") from exc
    model = model.to(target)
    model.ktlab_model_config = settings
    model.ktlab_source_sha256 = hashlib.sha256((UPSTREAM / f"{name}.py").read_bytes()).hexdigest()
    model.ktlab_device_binding = str(target)
    return model


def _ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def _gaps(concepts: list[int], timestamps: list[int]) -> dict[str, list[int]]:
    """Same feature formula as DktForgetDataset.calC, evaluated row by row."""
    last, counts = {}, {}
    previous = None
    result = {"rgaps": [], "sgaps": [], "pcounts": []}
    def bucket(x):
        if x < 0:
            raise ValueError("Non-monotonic timestamps; refusing to invent time-gap features")
        return round(math.log2(x + 1))
    for concept, timestamp in zip(concepts, timestamps):
        result["rgaps"].append(0 if concept not in last or concept == -1 else bucket((timestamp - last[concept]) / 60000) + 1)
        result["sgaps"].append(0 if previous is None or timestamp == -1 else bucket((timestamp - previous) / 60000) + 1)
        result["pcounts"].append(bucket(counts.get(concept, 0)))
        last[concept], previous = timestamp, timestamp
        counts[concept] = counts.get(concept, 0) + 1
    return result


class SequenceDataset(Dataset):
    """Index upstream CSV by byte offset; tensors are created on CPU per row.

    Upstream sequence CSV has one physical line per record. No .pkl is loaded
    and no dataset-sized tensor is transferred to the accelerator.
    """
    def __init__(self, path: str | Path, folds: set[int], *, question_level: bool = False,
                 time_features: bool = False):
        self.path = Path(path)
        self.question_level = question_level
        self.time_features = time_features
        self.offsets: list[tuple[int, int]] = []
        self._handle = None
        self._pid = None
        with self.path.open("rb") as stream:
            self.fields = next(csv.reader([stream.readline().decode("utf-8-sig")]))
            required = {"fold", "concepts", "responses", "selectmasks"}
            if question_level:
                required |= {"qidxs", "rest", "orirow"}
            if time_features:
                required.add("timestamps")
            if not required.issubset(self.fields):
                raise ValueError(f"{self.path.name} missing columns {sorted(required - set(self.fields))}")
            row_id = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                values = next(csv.reader([raw.decode("utf-8")]))
                if len(values) != len(self.fields):
                    raise ValueError("Expected one complete upstream sequence record per CSV line")
                if int(values[self.fields.index("fold")]) in folds:
                    self.offsets.append((offset, row_id))
                row_id += 1
        if not self.offsets:
            raise ValueError(f"No rows in {self.path} for folds {sorted(folds)}")

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        return {**self.__dict__, "_handle": None, "_pid": None}

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = self.path.open("rb")
            self._pid = os.getpid()
        offset, row_id = self.offsets[index]
        self._handle.seek(offset)
        row = dict(zip(self.fields, next(csv.reader([self._handle.readline().decode("utf-8")]))))
        c, r, selected = _ints(row["concepts"]), _ints(row["responses"]), _ints(row["selectmasks"])
        n = len(c)
        if len(r) != n or len(selected) != n or n < 2:
            raise ValueError("Misaligned or too-short sequence")
        masks = torch.tensor([c[i] != -1 and c[i + 1] != -1 for i in range(n - 1)], dtype=torch.bool)
        if any(x not in {-1, 0, 1} for x in selected):
            raise ValueError("selectmasks must use -1 padding, 0 exclusion, or 1 selection")
        result: dict[str, Any] = {"masks": masks, "smasks": torch.tensor(selected[1:]) == 1,
                                  "uid": row.get("uid", str(row_id)), "row_id": row_id,
                                  "ground_seq": torch.tensor([sum(x != -1 for x in r)], dtype=torch.float32)}
        result["smasks"] &= masks
        raw_sequences = {"cseqs": c, "rseqs": r,
                         "qseqs": _ints(row["questions"]) if row.get("questions") not in {None, "", "NA"} else [0] * n}
        for source, key in (("timestamps", "tseqs"), ("usetimes", "utseqs")):
            if row.get(source) not in {None, "", "NA"}:
                raw_sequences[key] = _ints(row[source])
        if self.time_features:
            if "tseqs" not in raw_sequences:
                raise ValueError("FA-KT requires real timestamps; no synthetic times are supplied")
            raw_sequences.update(_gaps(c, raw_sequences["tseqs"]))
        for key, sequence in raw_sequences.items():
            if len(sequence) != n:
                raise ValueError(f"Misaligned {key}")
            tensor = torch.tensor(sequence, dtype=torch.float32 if key == "rseqs" else torch.long)
            result[key] = tensor[:-1] * masks
            result["shft_" + key] = tensor[1:] * masks
        if self.question_level:
            result["event_ids"] = torch.tensor(_ints(row["qidxs"])[1:])
            result["rests"] = torch.tensor(_ints(row["rest"])[1:])
        else:
            result["event_ids"] = torch.arange(row_id * n + 1, row_id * n + n)
            result["rests"] = torch.zeros(n - 1, dtype=torch.long)
        return result


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def forward_model(model: torch.nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return probabilities aligned with shft_rseqs and upstream regularizer."""
    def full(key):
        return torch.cat((batch[key][:, :1], batch["shft_" + key]), dim=1).long()
    regularizer = torch.zeros((), device=batch["cseqs"].device)
    if model.model_name == "akt":
        predictions, regularizer = model(full("cseqs"), full("rseqs"), full("qseqs"))
    elif model.model_name == "mockt":
        predictions, regularizer = model(batch["ground_seq"], full("cseqs"), full("rseqs"), full("qseqs"))
    elif model.model_name == "simplekt":
        predictions = model(batch)
    elif model.model_name == "fa_kt":
        gap_names = ("rgaps", "sgaps", "pcounts")
        gaps = {key: batch[key] for name in gap_names for key in (name, "shft_" + name)}
        for name, size in zip(gap_names, (model.time_emb.num_rgap, model.time_emb.num_sgap, model.time_emb.num_pcount)):
            if max(int(gaps[name].max()), int(gaps["shft_" + name].max())) >= size:
                raise ValueError(f"{name} exceeds fixed gap vocabulary; no silent clipping")
        predictions = model(batch, gaps, train=True)[0]
    else:
        raise ValueError(f"Unsupported model {model.model_name}")
    return predictions[:, 1:], torch.as_tensor(regularizer, device=predictions.device)
