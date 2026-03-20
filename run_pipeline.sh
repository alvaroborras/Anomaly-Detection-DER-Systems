#!/bin/bash
set -euo pipefail
uv run python -m py_compile main.py
rm -rf outputs
start_ns=$(python - <<'PY'
import time
print(time.time_ns())
PY
)
uv run main.py
end_ns=$(python - <<'PY'
import time
print(time.time_ns())
PY
)
out=$(START_NS="$start_ns" python - <<'PY'
from pathlib import Path
import os,sys
start=int(os.environ['START_NS'])
paths=[Path('submission_full_data.csv'),Path('outputs/full_data_hybrid/submission_full_data.csv')]
new=[]
for p in paths:
    if p.exists() and p.stat().st_mtime_ns>=start:
        new.append((p.stat().st_mtime_ns,str(p)))
if not new:
    sys.exit('no fresh submission file found')
print(max(new)[1])
PY
)
read chars wall_ms actual_hash <<EOF
$(OUT_PATH="$out" START_NS="$start_ns" END_NS="$end_ns" python - <<'PY'
from pathlib import Path
import hashlib,os
mp=Path('main.py')
op=Path(os.environ['OUT_PATH'])
chars=len(mp.read_text())
wall_ms=(int(os.environ['END_NS'])-int(os.environ['START_NS']))//1_000_000
h=hashlib.sha256(op.read_bytes()).hexdigest()
print(chars,wall_ms,h)
PY
)
EOF
base=f49de9a5b433a55ea49da611334ec0326b817d80140d2016fc67f7c5d2763196
[ "$actual_hash" = "$base" ]
score=$((chars*1000000+wall_ms))
echo "submission=$out"
echo "sha256=$actual_hash"
echo "METRIC score=$score"
echo "METRIC chars=$chars"
echo "METRIC wall_ms=$wall_ms"
echo "METRIC hash_ok=1"
