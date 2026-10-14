"""Read-only capability probe: no models, gradients, or training data."""
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
disk = shutil.disk_usage(root)
result = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "gpu_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
    "workspace_free_gib": round(disk.free / 2**30, 2),
    "modules": {name: importlib.util.find_spec(name) is not None for name in ["pytest", "toml", "entmax", "yaml", "requests", "einops", "matplotlib"]},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
