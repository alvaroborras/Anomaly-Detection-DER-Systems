"""Shared runtime constants, dataclasses, and run configuration."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KAGGLE_WORKING_DIR = Path("/kaggle/working")
TRAIN_CSV_PATH = Path("/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems/train.csv")
TEST_CSV_PATH = Path("/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems/test.csv")
DEFAULT_SEED = 42
SQRT3 = math.sqrt(3.0)
CANON1 = "DERSec|DER Simulator|10 kW DER|1.2.3|SN-Three-Phase"
CANON2 = "DERSec|DER Simulator 100 kW|1.2.3.1|1.0.0|1100058974"
DEVICE_FAMILY_MAP = {"canon10": 0, "canon100": 1}
RESIDUAL_TAIL_LEVELS = {"tail": 0.95, "extreme": 0.99, "ultra": 0.999}
RESIDUAL_TAIL_FALLBACKS = {"tail": 0.05, "extreme": 0.10, "ultra": 0.20}
FAMILY_THRESHOLD_FLOOR = 0.02
MAX_THRESHOLD = 0.60
CANON100_NEGATIVE_WEIGHT = 1.50
HARD_OVERRIDE_TRAIN_WEIGHT = 0.35
SCENARIO_SMOOTHING = 50.0
AUDIT_TOLERANCE = 0.003
MIN_OVERRIDE_PRECISION = 0.995
CANON100_INTERACTION_FEATURES = [
    "hard_rule_score",
    "scenario_rate",
    "scenario_output_rate",
    "resid_quantile_score",
    "mode_dispatch_w_resid",
]
SURROGATE_TARGETS = {
    "w": ("DERMeasureAC_0_W", "DERCapacity_0_WMaxRtg"),
    "va": ("DERMeasureAC_0_VA", "DERCapacity_0_VAMaxRtg"),
    "var": ("DERMeasureAC_0_Var", "DERCapacity_0_VarMaxInjRtg"),
    "pf": ("DERMeasureAC_0_PF", None),
    "a": ("DERMeasureAC_0_A", "DERCapacity_0_AMaxRtg"),
}
SURROGATE_LEAKY_FEATURES = {
    *(
        f"DERMeasureAC_0_{field}"
        for field in """
    W VA Var PF A WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3
    AL1 AL2 AL3
    """.split()
    ),
    *"""
    w_over_wmaxrtg w_over_wmax va_over_vamax va_over_vamaxrtg var_over_injmax
    var_over_absmax a_over_amax w_minus_wmax w_minus_wmaxrtg va_minus_vamax
    var_minus_injmax var_plus_absmax w_eq_wmaxrtg w_eq_wmax var_eq_varmaxinj
    var_eq_neg_varmaxabs pf_sign_mismatch w_gt_wmax_tol w_gt_wmaxrtg_tol
    va_gt_vamax_tol var_gt_injmax_tol var_lt_absmax_tol va_minus_pqmag
    va_over_pqmag pf_from_w_va pf_error w_phase_sum_error va_phase_sum_error
    var_phase_sum_error phase_w_spread phase_var_spread wset_abs_error
    wsetpct_target wsetpct_abs_error wmaxlim_target wmaxlim_excess
    varset_abs_error varsetpct_target varsetpct_abs_error wset_enabled_far
    wsetpct_enabled_far wmaxlim_enabled_far varsetpct_enabled_far w_pct_of_rtg
    var_pct_of_limit enter_service_blocked_power enter_service_blocked_va
    enter_service_blocked_current pf_inj_target_error pf_inj_reversion_error
    pf_reactive_near_limit trip_lv_power_when_outside trip_hv_power_when_outside
    trip_lf_power_when_outside trip_hf_power_when_outside
    trip_any_power_when_outside voltvar_curve_error voltwatt_curve_error
    wattvar_curve_expected wattvar_curve_error freqdroop_w_over_pmin_pct
    dcw_over_w dcw_over_abs_w ac_zero_dc_positive ac_positive_dc_zero
    ac_dc_same_sign
    """.split(),
}


@dataclass
class MetricSummary:
    f2: float
    precision: float
    recall: float
    positive_rate: float
    rows: int


@dataclass
class ScenarioStats:
    """Keep the four scenario lookup maps together as one learned state block."""

    sum_map: Dict[int, float] = field(default_factory=dict)
    count_map: Dict[int, int] = field(default_factory=dict)
    output_sum_map: Dict[int, float] = field(default_factory=dict)
    output_count_map: Dict[int, int] = field(default_factory=dict)


@dataclass
class FamilySemanticContext:
    """Learned semantic calibration state for one device family."""

    surrogate_feature_cols: List[str] = field(default_factory=list)
    surrogate_models: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    residual_quantiles: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    family_base_rates: Dict[str, float] = field(default_factory=dict)
    scenario_stats: ScenarioStats = field(default_factory=ScenarioStats)


@dataclass
class FamilyModelBundle:
    """Store every learned component needed to score one family."""

    semantic_model: Optional[Any]
    cat_model: Optional[Any]
    semantic_context: FamilySemanticContext
    semantic_feature_cols: List[str]
    cat_feature_cols: List[str]
    threshold: float
    blend_weight: float


@dataclass(frozen=True)
class BaselineConfig:
    """Immutable training knobs for the reusable baseline model."""

    chunksize: int = 5000
    cv_folds: int = 5
    n_estimators: int = 180
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    cat_iterations: int = 400
    cat_depth: int = 8
    cat_learning_rate: float = 0.05
    n_jobs: int = 4
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class RunConfig:
    train_path: Path = TRAIN_CSV_PATH
    test_path: Path = TEST_CSV_PATH
    submission_path: Path = KAGGLE_WORKING_DIR / "submission.csv"
    chunksize: int = 5000
    cv_folds: int = 5
    xgb_n_estimators: int = 180
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    cat_iterations: int = 400
    cat_depth: int = 8
    cat_learning_rate: float = 0.05
    n_jobs: int = 4
    seed: int = DEFAULT_SEED

    def baseline_config(self) -> BaselineConfig:
        """Project a run config down to the model's immutable training knobs."""

        return BaselineConfig(
            chunksize=self.chunksize,
            cv_folds=self.cv_folds,
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            cat_iterations=self.cat_iterations,
            cat_depth=self.cat_depth,
            cat_learning_rate=self.cat_learning_rate,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )
