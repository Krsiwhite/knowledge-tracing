#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_python="${KT_RUNTIME_PYTHON:-/home/ml/.venvs/pytorch/bin/python}"
kt_env="${KT_ENV:-/home/ml/.venvs/knowledge-tracing}"
test -x "$runtime_python"
if [ ! -x "$kt_env/bin/python" ]; then
  "$runtime_python" -m venv "$kt_env"
fi
runtime_site="$($runtime_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
kt_site="$($kt_env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$runtime_site" "$project_root/src" "$project_root/third_party/pykt" > "$kt_site/ktlab-runtime.pth"
# Share the verified GPU runtime read-only; install only small missing tools locally.
"$kt_env/bin/python" -m pip install --no-deps 'toml==0.10.2' 'entmax==1.3' 'pytest==8.4.2' 'iniconfig==2.1.0' 'pluggy==1.6.0'
"$kt_env/bin/python" "$project_root/scripts/environment_probe.py"
