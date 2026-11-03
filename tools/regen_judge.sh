#!/usr/bin/env bash
# One-command judge-24 regen (plan 18/K2): rebuilds xfer_5..10, the
# sorted pair list, and prints the pool checksum (expect e71ea1a7...).
# Usage: tools/regen_judge.sh [out_list]  (default /tmp/f1_judge.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
./.venv/bin/python -B tools/gen_transfer.py 4 6 | tail -2
OUT="${1:-/tmp/f1_judge.txt}"
./.venv/bin/python -B -c "
import sys; sys.path.insert(0, '.')
from tools.skynet_train import pool_checksum
pairs = [f'building_xfer_{b}/{w}' for b in range(5, 11)
         for w in ('minneapolis_1', 'minneapolis_5', 'atlanta_4',
                   'singapore_2')]
open('$OUT', 'w').write('\n'.join(sorted(pairs)) + '\n')
print('n =', len(pairs), 'sha256 =', pool_checksum(pairs))
"
