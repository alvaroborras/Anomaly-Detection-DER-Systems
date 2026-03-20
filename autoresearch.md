# Autoresearch: minimize `main.py` while preserving the exact submission

## Objective
Shrink `main.py` as much as possible while keeping the generated submission bit-for-bit identical to the baseline submission.

Workload constraints:
- Runtime inputs may only be `train.csv` and `test.csv`.
- The generated submission hash must stay equal to the baseline hash:
  `f49de9a5b433a55ea49da611334ec0326b817d80140d2016fc67f7c5d2763196`.
- For equal character count, lower `uv run main.py` wall time wins.
- No extra Python files.
- No cached temporary results across runs.
- No encoding, compression, or obfuscation tricks.

## Metrics
- **Primary**: `score` (unitless, lower is better) where `score = chars * 1_000_000 + wall_ms`
- **Secondary**: `chars`, `wall_ms`, `hash_ok`

## How to Run
- `./run_pipeline.sh` — syntax-checks `main.py`, removes generated artifacts, runs `uv run main.py`, finds the fresh submission, verifies the baseline SHA-256, and emits `METRIC` lines.
- `./autoresearch.sh` — thin wrapper around `./run_pipeline.sh`.

## Files in Scope
- `main.py` — the artifact being minimized; must still generate the exact baseline submission.
- `run_pipeline.sh` — benchmark/helper script for honest timing + hash verification.
- `autoresearch.sh` — benchmark wrapper for autoresearch.
- `autoresearch.md` — session state and experiment notes.
- `autoresearch.ideas.md` — backlog for larger ideas not immediately attempted.

## Off Limits
- `train.csv`, `test.csv` — inputs only.
- `submission_full_data.csv` — baseline reference only; final runtime logic must not depend on it.
- `uv.lock`, `pyproject.toml` — do not change dependencies.
- Any extra Python files.

## Constraints
- Exact submission hash match on every kept experiment.
- Runtime must honestly measure `uv run main.py` wall time.
- Do not fake artifact generation by reusing stale files.
- Prefer simpler code when score is equal or better.

## Current Understanding
- Current `main.py` is ~130k chars and contains substantial non-essential surface area: large docstrings, reporting/config dataclasses, model/report persistence, zip-path indirection, and audit/report code.
- The safest path is to preserve the current predictor logic while deleting non-essential code and simplifying I/O around `train.csv`/`test.csv`.
- Distillation experiments on engineered features looked intellectually promising, but exact-match models were still structurally large; this is a backup path, not the leading one.

## What's Been Tried
- Analysis only so far.
- Baseline hash identified from the reference submission: `f49de9a5b433a55ea49da611334ec0326b817d80140d2016fc67f7c5d2763196`.
- Explored distillation offline: exact matches were possible with engineered-feature trees/boosters, but model representations stayed too large to obviously beat a direct code-pruning refactor.
- Best current direction: aggressively remove code not needed for final prediction while preserving the original training/inference behavior.
