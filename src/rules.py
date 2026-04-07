"""Hard-rule metadata shared by feature engineering and modeling."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HardRuleContribution:
    """One boolean source column that contributes to hard-rule aggregates."""

    name: str | None
    column: str
    score_weight: float
    count_weight: int = 1
    default_override: bool = False


@dataclass(frozen=True)
class RuleGroup:
    """Semantic grouping for related hard-rule contributions."""

    name: str
    rules: tuple[HardRuleContribution, ...]


@dataclass(frozen=True)
class HardRuleOutputs:
    """Aggregate hard-rule feature values derived from raw rule columns."""

    hard_override_anomaly: np.ndarray
    hard_rule_count: np.ndarray
    hard_rule_score: np.ndarray
    hard_rule_anomaly: np.ndarray


def _rule(name: str, column: str, score_weight: float, count_weight: int = 1) -> HardRuleContribution:
    return HardRuleContribution(name, column, score_weight, count_weight, True)


def _flatten_contributions(groups: Sequence[RuleGroup]) -> tuple[HardRuleContribution, ...]:
    return tuple(rule for group in groups for rule in group.rules)


# Order matters because feature generation, override handling, and reporting all
# consume a stable rule ordering, so the grouped source of truth must preserve
# the existing flattened order exactly.
HARD_RULE_GROUPS = (
    RuleGroup(
        "identity",
        (
            _rule("noncanonical", "noncanonical", 3.0),
        ),
    ),
    RuleGroup(
        "power_envelope",
        (
            _rule("w_gt_wmax", "w_gt_wmax_tol", 2.0),
            _rule("w_gt_wmaxrtg", "w_gt_wmaxrtg_tol", 2.0),
            _rule("va_gt_vamax", "va_gt_vamax_tol", 2.0),
            _rule("var_gt_injmax", "var_gt_injmax_tol", 2.0),
            _rule("var_lt_absmax", "var_lt_absmax_tol", 2.0),
        ),
    ),
    RuleGroup(
        "control_state",
        (
            # `wsetpct_far` absorbs the legacy `wset_far` aggregate contribution
            # because the two source columns are identical on the fixed
            # train/test data.
            _rule("wsetpct_far", "wsetpct_enabled_far", 3.0, 2),
            _rule("ac_type_rare", "ac_type_is_rare", 1.5),
            _rule("dc_type_rare", "dc_port_type_rare_any", 1.5),
            _rule("enter_state", "enter_service_state_anomaly", 2.0),
        ),
    ),
    RuleGroup(
        "protection",
        (
            _rule("pf_abs", "pf_abs_ext_present", 1.5),
            _rule("pf_abs_rvrt", "pf_abs_rvrt_ext_present", 1.5),
            _rule("trip_power", "trip_any_power_when_outside", 2.0),
        ),
    ),
    RuleGroup(
        "aggregate_only",
        (
            HardRuleContribution(None, "common_missing_any", 2.5),
            HardRuleContribution(None, "varsetpct_enabled_far", 1.0),
            HardRuleContribution(None, "enter_service_blocked_power", 0.35),
            HardRuleContribution(None, "enter_service_blocked_current", 0.35),
        ),
    ),
)
HARD_RULE_CONTRIBUTIONS = _flatten_contributions(HARD_RULE_GROUPS)

HARD_RULE_SPECS = tuple(spec for spec in HARD_RULE_CONTRIBUTIONS if spec.name is not None)
HARD_RULE_COLUMNS = [spec.column for spec in HARD_RULE_CONTRIBUTIONS]
HARD_RULE_NAMES = [spec.name for spec in HARD_RULE_SPECS]
DEFAULT_HARD_OVERRIDE_NAMES = [spec.name for spec in HARD_RULE_SPECS if spec.default_override]
RULE_COLUMN_MAP = {spec.name: spec.column for spec in HARD_RULE_SPECS}


def _infer_row_count(data: Mapping[str, Sequence[object]]) -> int:
    for values in data.values():
        return len(values)
    return 0


def _coerce_rule_flag(values: Sequence[object] | pd.Series) -> np.ndarray:
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()
    else:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            arr = np.nan_to_num(arr, nan=0.0)
        else:
            arr = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(0).to_numpy()
    return np.asarray(arr, dtype=np.int8) == 1


def build_hard_rule_column_flags(
    data: Mapping[str, Sequence[object]] | pd.DataFrame,
    *,
    columns: Sequence[str] = HARD_RULE_COLUMNS,
) -> dict[str, np.ndarray]:
    return {column: _coerce_rule_flag(data[column]) for column in columns}


def build_named_rule_flags(
    data: Mapping[str, Sequence[object]] | pd.DataFrame,
    *,
    column_flags: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    column_flags = column_flags or build_hard_rule_column_flags(data)
    return {spec.name: column_flags[spec.column] for spec in HARD_RULE_SPECS}


def compute_active_override_anomaly(
    data: Mapping[str, Sequence[object]] | pd.DataFrame,
    active_override_names: Sequence[str],
    *,
    rule_flags: Mapping[str, np.ndarray] | None = None,
    row_count: int | None = None,
) -> np.ndarray:
    if rule_flags is None:
        rule_flags = build_named_rule_flags(data)
    if row_count is None:
        row_count = _infer_row_count(rule_flags)
    if not active_override_names:
        return np.zeros(row_count, dtype=np.int8)
    return np.column_stack([rule_flags[name] for name in active_override_names]).any(axis=1).astype(np.int8)


def compute_hard_rule_outputs(
    data: Mapping[str, Sequence[object]] | pd.DataFrame,
    active_override_names: Sequence[str],
    *,
    row_count: int | None = None,
) -> HardRuleOutputs:
    column_flags = build_hard_rule_column_flags(data)
    if row_count is None:
        row_count = _infer_row_count(column_flags)
    rule_flags = build_named_rule_flags(data, column_flags=column_flags)
    hard_override_anomaly = compute_active_override_anomaly(
        data,
        active_override_names,
        rule_flags=rule_flags,
        row_count=row_count,
    )
    hard_rule_count = np.zeros(row_count, dtype=np.int16)
    hard_rule_score = np.zeros(row_count, dtype=np.float32)
    hard_rule_anomaly = np.zeros(row_count, dtype=bool)
    for spec in HARD_RULE_CONTRIBUTIONS:
        flags = column_flags[spec.column]
        hard_rule_count += spec.count_weight * flags.astype(np.int16)
        if spec.score_weight != 0.0:
            hard_rule_score += spec.score_weight * flags.astype(np.float32)
        hard_rule_anomaly |= flags
    return HardRuleOutputs(
        hard_override_anomaly=hard_override_anomaly,
        hard_rule_count=hard_rule_count.astype(np.int8),
        hard_rule_score=hard_rule_score,
        hard_rule_anomaly=hard_rule_anomaly.astype(np.int8),
    )
