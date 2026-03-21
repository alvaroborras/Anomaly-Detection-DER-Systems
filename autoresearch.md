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
- The current kept `main.py` is ~69.4k chars, down from ~130k.
- The most productive safe reductions so far have come from deleting non-essential surface area, removing dead duplicate logic, shortening internal names/feature columns, and hoisting pure/stateless helpers out of class scope.
- The current best path is still careful exact-token shortening with an explicit denylist for external API keywords (`cat_features`, `chunksize`, etc.); those mistakes are easy to make and expensive to rerun.
- Large remaining opportunities still appear to be: extra bookkeeping around artifact generation, semantic/scenario helper plumbing, and any remaining verbose setup code that does not affect the final submission.
- Distillation experiments on engineered features looked intellectually promising, but exact-match models were still structurally large; this remains a backup path, not the leading one.

## What's Been Tried
- Baseline hash identified from the reference submission: `f49de9a5b433a55ea49da611334ec0326b817d80140d2016fc67f7c5d2763196`.
- Explored distillation offline: exact matches were possible with engineered-feature trees/boosters, but model representations stayed too large to obviously beat a direct code-pruning refactor.
- Kept: AST-based syntax-preserving minification (drop docstrings/comments via unparse, remove function annotations) reduced `main.py` from 130,112 chars to 112,003 chars and also slightly improved runtime while preserving the exact hash.
- Kept: removed unused report/config/save machinery, switched the runtime to direct `train.csv` / `test.csv` reads, simplified artifact rebuilding to always start fresh, and collapsed `run_pipeline()` to the minimum needed for final prediction. This reduced `main.py` further to 99,978 chars while preserving the exact hash.
- Kept: removed extra annotated assignment syntax, deleted the now-unused semantic-context dataclass wrapper, and trimmed more bookkeeping around context/state setup. This reduced `main.py` again to 98,357 chars while preserving the exact hash.
- Kept: removed a few more dead imports/parameters and collapsed semantic-context capture/activation to plain tuple passing. This brought `main.py` to 97,991 chars while preserving the exact hash.
- Kept: shortened a batch of long internal identifiers (`semantic_*`, scenario-map names, override/config names, etc.) with no behavior change. This cut `main.py` further to 95,974 chars while preserving the exact hash.
- Kept: shortened another batch of long global constants/column-set identifiers (`USECOLS_*`, raw/string-column constants, map names, etc.) with no behavior change. This brought `main.py` to 95,192 chars while preserving the exact hash.
- Kept: shortened another batch of common local variable names (`feature_cols`, masks, score/probability/temp names, output counters, etc.) with no behavior change. This brought `main.py` to 93,852 chars while preserving the exact hash.
- Kept: shortened another batch of helper-method names plus common locals (`_apply_residual_calibration_features`, `_fit_transform_scenario_features`, `_train_*`, curve/scenario temp names, etc.) with no behavior change. This brought `main.py` to 90,596 chars while preserving the exact hash.
- Kept: shortened class/import aliases, more helper/function names, and several remaining long tuning constants with no behavior change. This brought `main.py` to 89,610 chars while preserving the exact hash.
- Kept: shortened several internal engineered feature/column names (`device_family`, hard-rule/scenario feature names, residual-calibration prefixes, etc.) with no behavior change. This brought `main.py` to 87,820 chars while preserving the exact hash.
- Kept: shortened another batch of internal identifiers (feature-ratio names, helper args, cross-validation and integrity variables, etc.) with no behavior change. This brought `main.py` to 84,931 chars while preserving the exact hash.
- Kept: removed the dead earlier `tune_threshold` definition and shortened another broad batch of helper names, constants, and internal feature-column names with no behavior change. This brought `main.py` to 81,646 chars while preserving the exact hash.
- Kept: shortened another broad batch of residual/scenario/curve feature names plus remaining helper, attribute, and local identifiers with no behavior change. This brought `main.py` to 77,940 chars while preserving the exact hash.
- Kept: shortened another batch of remaining hard-rule names, tuning locals, fold-feature columns, and residual/freqdroop feature names with no behavior change. This brought `main.py` to 75,888 chars while preserving the exact hash.
- Crash/reverted: the first attempt to hoist stateless helpers to module scope used an over-broad prefix replacement and broke `self._nsm`/`self._ncm` into undefined globals. The idea was still sound, but the replacement needed exact name boundaries.
- Kept: hoisted the stateless math/selection helpers out of class scope with an exact replacement pass, which safely reduced indentation/decorator overhead and brought `main.py` to 74,686 chars while preserving the exact hash.
- Kept: shortened another safe batch of remaining local identifiers plus the last long internal DC/freqdroop/target feature names, with a narrower non-colliding rename set after the prior crash. This brought `main.py` to 73,484 chars while preserving the exact hash.
- Kept: shortened another safe batch of shared column/field constants plus remaining one-off locals and internal feature names (DC, phase-error, freqdroop, watt-limit terms). This brought `main.py` to 72,246 chars while preserving the exact hash.
- Kept: hoisted another batch of pure helper functions (encoding/binning/hash/stat aggregation/candidate builders) out of class scope and fixed their signatures for direct global calls. This brought `main.py` to 71,807 chars while preserving the exact hash.
- Kept: hoisted another batch of pure helpers (`_cn/_bsw/_bsf/_bsk/_bok/_afi`) out of class scope and shortened internal hyperparameter attribute names while preserving external library keyword args. This brought `main.py` to 71,294 chars while preserving the exact hash.
- Kept: applied another exact-token shortening pass on remaining locals and internal feature terms (residual magnitudes, mode flags, DC/freqdroop temporaries, curve metadata, and capacity-rating locals) without touching external API keywords. This brought `main.py` to 70,591 chars while preserving the exact hash.
- Crash/reverted: one broader rename pass also changed pandas `read_csv(chunksize=...)` into `csz=...`; external API keywords need an explicit denylist just like CatBoost keywords do.
- Kept: re-applied a narrower exact-token shortening pass after the `chunksize` crash, safely renaming remaining internal rule labels, state names, and local plumbing terms while preserving external API keywords. This brought `main.py` to 69,559 chars while preserving the exact hash.
- Kept: shortened the internal family labels and inlined the tiny `rp/main` wrappers into the `__main__` block, plus another safe local/rule-token tightening pass. This brought `main.py` to 69,388 chars while preserving the exact hash.
- Best current direction: continue deleting helper/reporting structures and compacting internal plumbing without changing the trained decision path; naming surface and class-scoped boilerplate are still paying off, but avoid blind replacements of external API keywords.
