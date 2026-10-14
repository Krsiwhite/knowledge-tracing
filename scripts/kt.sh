#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kt_env="${KT_ENV:-/home/ml/.venvs/knowledge-tracing}"
cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/third_party/pykt${PYTHONPATH:+:$PYTHONPATH}"
exec "$kt_env/bin/python" -m ktlab "$@"
