#!/bin/bash
# Bounded training replaces the old crash/restart loop. Pass --rom and other CLI flags.
set -euo pipefail
exec python "$(dirname "$0")/baseline_fast_v2.py" "$@"
