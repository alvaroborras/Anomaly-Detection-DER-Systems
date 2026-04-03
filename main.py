import gc
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

LOGGER = logging.getLogger(__name__)


def dedupe(columns: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(columns))


def prefixed(prefix: str, fields: Sequence[str]) -> List[str]:
    return [f"{prefix}.{field}" for field in fields]


def build_block_columns(
    prefix: str,
    *,
    base_fields: Sequence[str],
    item_label: str,
    item_count: int,
    item_fields: Sequence[str] = (),
    point_count: int = 0,
    point_fields: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> List[str]:
    columns = prefixed(prefix, base_fields)
    for item_idx in range(item_count):
        item_prefix = f"{prefix}.{item_label}[{item_idx}]"
        columns.extend(prefixed(item_prefix, item_fields))
        if groups:
            for group in groups:
                group_prefix = f"{item_prefix}.{group}"
                columns.append(f"{group_prefix}.ActPt")
                for point_idx in range(point_count):
                    columns.extend(
                        prefixed(
                            f"{group_prefix}.Pt[{point_idx}]",
                            point_fields,
                        )
                    )
            continue
        for point_idx in range(point_count):
            columns.extend(prefixed(f"{item_prefix}.Pt[{point_idx}]", point_fields))
    return columns


COMMON_FIELDS = (
    "Mn",
    "Md",
    "Opt",
    "Vr",
    "SN",
)
COMMON_STR = prefixed("common[0]", COMMON_FIELDS)
COMMON_COLUMNS = prefixed("common[0]", ("ID", "L", *COMMON_FIELDS, "DA"))

MEASURE_AC_FIELDS = (
    "ID",
    "L",
    "ACType",
    "W",
    "VA",
    "Var",
    "PF",
    "A",
    "LLV",
    "LNV",
    "Hz",
    "TmpAmb",
    "TmpCab",
    "TmpSnk",
    "TmpTrns",
    "TmpSw",
    "TmpOt",
    "ThrotPct",
    "ThrotSrc",
    "WL1",
    "WL2",
    "WL3",
    "VAL1",
    "VAL2",
    "VAL3",
    "VarL1",
    "VarL2",
    "VarL3",
    "PFL1",
    "PFL2",
    "PFL3",
    "AL1",
    "AL2",
    "AL3",
    "VL1L2",
    "VL2L3",
    "VL3L1",
    "VL1",
    "VL2",
    "VL3",
)
MEASURE_AC_COLUMNS = prefixed("DERMeasureAC[0]", MEASURE_AC_FIELDS)

CAPACITY_FIELDS = (
    "ID",
    "L",
    "WMaxRtg",
    "VAMaxRtg",
    "VarMaxInjRtg",
    "VarMaxAbsRtg",
    "WChaRteMaxRtg",
    "WDisChaRteMaxRtg",
    "VAChaRteMaxRtg",
    "VADisChaRteMaxRtg",
    "VNomRtg",
    "VMaxRtg",
    "VMinRtg",
    "AMaxRtg",
    "PFOvrExtRtg",
    "PFUndExtRtg",
    "NorOpCatRtg",
    "AbnOpCatRtg",
    "IntIslandCatRtg",
    "WMax",
    "WMaxOvrExt",
    "WOvrExtPF",
    "WMaxUndExt",
    "WUndExtPF",
    "VAMax",
    "VarMaxInj",
    "VarMaxAbs",
    "WChaRteMax",
    "WDisChaRteMax",
    "VAChaRteMax",
    "VADisChaRteMax",
    "VNom",
    "VMax",
    "VMin",
    "AMax",
    "PFOvrExt",
    "PFUndExt",
    "CtrlModes",
    "IntIslandCat",
)
CAPACITY_COLUMNS = prefixed("DERCapacity[0]", CAPACITY_FIELDS)

ENTER_SERVICE_FIELDS = (
    "ID",
    "L",
    "ES",
    "ESVHi",
    "ESVLo",
    "ESHzHi",
    "ESHzLo",
    "ESDlyTms",
    "ESRndTms",
    "ESRmpTms",
    "ESDlyRemTms",
)
ENTER_SERVICE_COLUMNS = prefixed("DEREnterService[0]", ENTER_SERVICE_FIELDS)

CTL_AC_FIELDS = (
    "ID",
    "L",
    "PFWInjEna",
    "PFWInjEnaRvrt",
    "PFWInjRvrtTms",
    "PFWInjRvrtRem",
    "PFWAbsEna",
    "PFWAbsEnaRvrt",
    "PFWAbsRvrtTms",
    "PFWAbsRvrtRem",
    "WMaxLimPctEna",
    "WMaxLimPct",
    "WMaxLimPctRvrt",
    "WMaxLimPctEnaRvrt",
    "WMaxLimPctRvrtTms",
    "WMaxLimPctRvrtRem",
    "WSetEna",
    "WSetMod",
    "WSet",
    "WSetRvrt",
    "WSetPct",
    "WSetPctRvrt",
    "WSetEnaRvrt",
    "WSetRvrtTms",
    "WSetRvrtRem",
    "VarSetEna",
    "VarSetMod",
    "VarSetPri",
    "VarSet",
    "VarSetRvrt",
    "VarSetPct",
    "VarSetPctRvrt",
    "VarSetEnaRvrt",
    "VarSetRvrtTms",
    "VarSetRvrtRem",
    "WRmp",
    "WRmpRef",
    "VarRmp",
    "AntiIslEna",
    "PFWInj.PF",
    "PFWInj.Ext",
    "PFWInjRvrt.PF",
    "PFWInjRvrt.Ext",
    "PFWAbs.Ext",
    "PFWAbsRvrt.Ext",
)
CTL_AC_COLUMNS = prefixed("DERCtlAC[0]", CTL_AC_FIELDS)

CURVE_BLOCK_BASE_FIELDS = (
    "ID",
    "L",
    "Ena",
    "AdptCrvReq",
    "AdptCrvRslt",
    "NPt",
    "NCrv",
    "RvrtTms",
    "RvrtRem",
    "RvrtCrv",
)
VOLT_VAR_ITEM_FIELDS = (
    "ActPt",
    "DeptRef",
    "Pri",
    "VRef",
    "VRefAuto",
    "VRefAutoEna",
    "VRefAutoTms",
    "RspTms",
    "ReadOnly",
)
VOLT_WATT_ITEM_FIELDS = (
    "ActPt",
    "DeptRef",
    "RspTms",
    "ReadOnly",
)
FREQ_DROOP_BASE_FIELDS = (
    "ID",
    "L",
    "Ena",
    "AdptCtlReq",
    "AdptCtlRslt",
    "NCtl",
    "RvrtTms",
    "RvrtRem",
    "RvrtCtl",
)
FREQ_DROOP_ITEM_FIELDS = (
    "DbOf",
    "DbUf",
    "KOf",
    "KUf",
    "RspTms",
    "PMin",
    "ReadOnly",
)
WATT_VAR_ITEM_FIELDS = (
    "ActPt",
    "DeptRef",
    "Pri",
    "ReadOnly",
)
TRIP_BLOCK_BASE_FIELDS = (
    "ID",
    "L",
    "Ena",
    "AdptCrvReq",
    "AdptCrvRslt",
    "NPt",
    "NCrvSet",
)

VOLT_VAR_COLUMNS = build_block_columns(
    "DERVoltVar[0]",
    base_fields=CURVE_BLOCK_BASE_FIELDS,
    item_label="Crv",
    item_count=3,
    item_fields=VOLT_VAR_ITEM_FIELDS,
    point_count=4,
    point_fields=("V", "Var"),
)
VOLT_WATT_COLUMNS = build_block_columns(
    "DERVoltWatt[0]",
    base_fields=CURVE_BLOCK_BASE_FIELDS,
    item_label="Crv",
    item_count=3,
    item_fields=VOLT_WATT_ITEM_FIELDS,
    point_count=2,
    point_fields=("V", "W"),
)
FREQ_DROOP_COLUMNS = build_block_columns(
    "DERFreqDroop[0]",
    base_fields=FREQ_DROOP_BASE_FIELDS,
    item_label="Ctl",
    item_count=3,
    item_fields=FREQ_DROOP_ITEM_FIELDS,
)
WATT_VAR_COLUMNS = build_block_columns(
    "DERWattVar[0]",
    base_fields=CURVE_BLOCK_BASE_FIELDS,
    item_label="Crv",
    item_count=3,
    item_fields=WATT_VAR_ITEM_FIELDS,
    point_count=6,
    point_fields=("W", "Var"),
)

TRIP_SPECS: Dict[str, Tuple[str, str, str]] = {
    "lv": ("DERTripLV[0]", "V", "low"),
    "hv": ("DERTripHV[0]", "V", "high"),
    "lf": ("DERTripLF[0]", "Hz", "low"),
    "hf": ("DERTripHF[0]", "Hz", "high"),
}
TRIP_COLUMNS = {
    short_name: build_block_columns(
        prefix,
        base_fields=TRIP_BLOCK_BASE_FIELDS,
        item_label="Crv",
        item_count=2,
        item_fields=("ReadOnly",),
        point_count=5,
        point_fields=(axis_name, "Tms"),
        groups=("MustTrip", "MayTrip", "MomCess"),
    )
    for short_name, (prefix, axis_name, _) in TRIP_SPECS.items()
}

MEASURE_DC_FIELDS = (
    "ID",
    "L",
    "NPrt",
    "DCA",
    "DCW",
    "Prt[0].PrtTyp",
    "Prt[0].ID",
    "Prt[0].DCA",
    "Prt[0].DCV",
    "Prt[0].DCW",
    "Prt[0].Tmp",
    "Prt[1].PrtTyp",
    "Prt[1].ID",
    "Prt[1].DCA",
    "Prt[1].DCV",
    "Prt[1].DCW",
    "Prt[1].Tmp",
)
MEASURE_DC_COLUMNS = prefixed("DERMeasureDC[0]", MEASURE_DC_FIELDS)

BLOCK_SOURCE_COLUMNS: Dict[str, List[str]] = {
    "common": COMMON_COLUMNS,
    "measure_ac": MEASURE_AC_COLUMNS,
    "capacity": CAPACITY_COLUMNS,
    "enter_service": ENTER_SERVICE_COLUMNS,
    "ctl_ac": CTL_AC_COLUMNS,
    "volt_var": VOLT_VAR_COLUMNS,
    "volt_watt": VOLT_WATT_COLUMNS,
    "freq_droop": FREQ_DROOP_COLUMNS,
    "watt_var": WATT_VAR_COLUMNS,
    "measure_dc": MEASURE_DC_COLUMNS,
}
for short_name, cols in TRIP_COLUMNS.items():
    BLOCK_SOURCE_COLUMNS[f"trip_{short_name}"] = cols

CURVE_BLOCK_META_FIELDS = (
    "Ena",
    "AdptCrvReq",
    "AdptCrvRslt",
    "NPt",
    "NCrv",
    "RvrtTms",
    "RvrtRem",
    "RvrtCrv",
)
FREQ_DROOP_META_FIELDS = (
    "Ena",
    "AdptCtlReq",
    "AdptCtlRslt",
    "NCtl",
    "RvrtTms",
    "RvrtRem",
    "RvrtCtl",
)
TRIP_META_FIELDS = (
    "Ena",
    "AdptCrvReq",
    "AdptCrvRslt",
    "NPt",
    "NCrvSet",
)

RAW_NUMERIC = dedupe(
    [
        "common[0].DA",
        *prefixed("DERMeasureAC[0]", MEASURE_AC_FIELDS[2:]),
        *prefixed("DERCapacity[0]", CAPACITY_FIELDS[2:]),
        *prefixed("DEREnterService[0]", ENTER_SERVICE_FIELDS[2:]),
        *prefixed("DERCtlAC[0]", CTL_AC_FIELDS[2:]),
        *prefixed("DERVoltVar[0]", CURVE_BLOCK_META_FIELDS),
        *prefixed("DERVoltWatt[0]", CURVE_BLOCK_META_FIELDS),
        *prefixed("DERFreqDroop[0]", FREQ_DROOP_META_FIELDS),
        *prefixed("DERWattVar[0]", CURVE_BLOCK_META_FIELDS),
        *prefixed("DERMeasureDC[0]", MEASURE_DC_FIELDS[2:]),
    ]
)

TRIP_META_COLUMNS = [f"{prefix}.{field}" for prefix, _, _ in TRIP_SPECS.values() for field in TRIP_META_FIELDS]
RAW_EXTRA_NUMERIC_COLUMNS = [
    "DERMeasureAC[0].A_SF",
    "DERMeasureAC[0].V_SF",
    "DERMeasureAC[0].Hz_SF",
    "DERMeasureAC[0].W_SF",
    "DERMeasureAC[0].PF_SF",
    "DERMeasureAC[0].VA_SF",
    "DERMeasureAC[0].Var_SF",
    "DERCapacity[0].WOvrExtRtg",
    "DERCapacity[0].WOvrExtRtgPF",
    "DERCapacity[0].WUndExtRtg",
    "DERCapacity[0].WUndExtRtgPF",
    "DERCapacity[0].W_SF",
    "DERCapacity[0].PF_SF",
    "DERCapacity[0].VA_SF",
    "DERCapacity[0].Var_SF",
    "DERCapacity[0].V_SF",
    "DERCapacity[0].A_SF",
    "DERCtlAC[0].WSet_SF",
    "DERMeasureDC[0].DCA_SF",
    "DERMeasureDC[0].DCW_SF",
]
RAW_EXTRA_STRING_COLUMNS = [
    "DERMeasureDC[0].Prt[0].IDStr",
    "DERMeasureDC[0].Prt[1].IDStr",
]
RAW_NUMERIC = dedupe([*RAW_NUMERIC, *TRIP_META_COLUMNS, *RAW_EXTRA_NUMERIC_COLUMNS])
RAW_STRING_COLUMNS = dedupe([*COMMON_STR, *RAW_EXTRA_STRING_COLUMNS])

TRIP_SOURCE_COLUMNS = [col for cols in TRIP_COLUMNS.values() for col in cols]
ALL_SOURCE_COLUMNS = dedupe(
    [
        *COMMON_COLUMNS,
        *MEASURE_AC_COLUMNS,
        *CAPACITY_COLUMNS,
        *ENTER_SERVICE_COLUMNS,
        *CTL_AC_COLUMNS,
        *VOLT_VAR_COLUMNS,
        *VOLT_WATT_COLUMNS,
        *FREQ_DROOP_COLUMNS,
        *WATT_VAR_COLUMNS,
        *MEASURE_DC_COLUMNS,
        *TRIP_SOURCE_COLUMNS,
        *RAW_EXTRA_NUMERIC_COLUMNS,
        *RAW_EXTRA_STRING_COLUMNS,
    ]
)
NUMERIC_SOURCE_COLUMNS = [c for c in ALL_SOURCE_COLUMNS if c not in RAW_STRING_COLUMNS]

USECOLS_TRAIN = dedupe(["Id", "Label", *ALL_SOURCE_COLUMNS])
USECOLS_TEST = dedupe(["Id", *ALL_SOURCE_COLUMNS])
CANON1 = "DERSec|DER Simulator|10 kW DER|1.2.3|SN-Three-Phase"
CANON2 = "DERSec|DER Simulator 100 kW|1.2.3.1|1.0.0|1100058974"
SAFE_RAW = {c: re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in RAW_NUMERIC}
SAFE_STR = {c: re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in RAW_STRING_COLUMNS}
SCRIPT_DIR = Path(".").resolve().parent
KAGGLE_WORKING_DIR = Path("/kaggle/working")
TRAIN_CSV_PATH = Path("/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems/train.csv")
TEST_CSV_PATH = Path("/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems/test.csv")
DEFAULT_SEED = 42
SQRT3 = math.sqrt(3.0)
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
SURROGATE_MEASURE_AC_FIELDS = (
    "W",
    "VA",
    "Var",
    "PF",
    "A",
    "WL1",
    "WL2",
    "WL3",
    "VAL1",
    "VAL2",
    "VAL3",
    "VarL1",
    "VarL2",
    "VarL3",
    "PFL1",
    "PFL2",
    "PFL3",
    "AL1",
    "AL2",
    "AL3",
)
SURROGATE_DERIVED_FEATURES = (
    "w_over_wmaxrtg",
    "w_over_wmax",
    "va_over_vamax",
    "va_over_vamaxrtg",
    "var_over_injmax",
    "var_over_absmax",
    "a_over_amax",
    "w_minus_wmax",
    "w_minus_wmaxrtg",
    "va_minus_vamax",
    "var_minus_injmax",
    "var_plus_absmax",
    "w_eq_wmaxrtg",
    "w_eq_wmax",
    "var_eq_varmaxinj",
    "var_eq_neg_varmaxabs",
    "pf_sign_mismatch",
    "w_gt_wmax_tol",
    "w_gt_wmaxrtg_tol",
    "va_gt_vamax_tol",
    "var_gt_injmax_tol",
    "var_lt_absmax_tol",
    "va_minus_pqmag",
    "va_over_pqmag",
    "pf_from_w_va",
    "pf_error",
    "w_phase_sum_error",
    "va_phase_sum_error",
    "var_phase_sum_error",
    "phase_w_spread",
    "phase_var_spread",
    "wset_abs_error",
    "wsetpct_target",
    "wsetpct_abs_error",
    "wmaxlim_target",
    "wmaxlim_excess",
    "varset_abs_error",
    "varsetpct_target",
    "varsetpct_abs_error",
    "wset_enabled_far",
    "wsetpct_enabled_far",
    "wmaxlim_enabled_far",
    "varsetpct_enabled_far",
    "w_pct_of_rtg",
    "var_pct_of_limit",
    "enter_service_blocked_power",
    "enter_service_blocked_va",
    "enter_service_blocked_current",
    "pf_inj_target_error",
    "pf_inj_reversion_error",
    "pf_reactive_near_limit",
    "trip_lv_power_when_outside",
    "trip_hv_power_when_outside",
    "trip_lf_power_when_outside",
    "trip_hf_power_when_outside",
    "trip_any_power_when_outside",
    "voltvar_curve_error",
    "voltwatt_curve_error",
    "wattvar_curve_expected",
    "wattvar_curve_error",
    "freqdroop_w_over_pmin_pct",
    "dcw_over_w",
    "dcw_over_abs_w",
    "ac_zero_dc_positive",
    "ac_positive_dc_zero",
    "ac_dc_same_sign",
)
SURROGATE_LEAKY_FEATURES = {
    *(f"DERMeasureAC_0_{field}" for field in SURROGATE_MEASURE_AC_FIELDS),
    *SURROGATE_DERIVED_FEATURES,
}

HARD_RULE_NAMES = [
    "noncanonical",
    "common_missing",
    "w_gt_wmax",
    "w_gt_wmaxrtg",
    "va_gt_vamax",
    "var_gt_injmax",
    "var_lt_absmax",
    "wset_far",
    "wsetpct_far",
    "wmaxlim_far",
    "varsetpct_far",
    "model_structure",
    "ac_type_rare",
    "dc_type_rare",
    "enter_state",
    "enter_blocked_power",
    "enter_blocked_current",
    "pf_abs",
    "pf_abs_rvrt",
    "trip_power",
]
DEFAULT_HARD_OVERRIDE_NAMES = [
    "noncanonical",
    "common_missing",
    "w_gt_wmax",
    "w_gt_wmaxrtg",
    "va_gt_vamax",
    "var_gt_injmax",
    "var_lt_absmax",
    "wset_far",
    "wsetpct_far",
    "model_structure",
    "ac_type_rare",
    "dc_type_rare",
    "enter_state",
    "pf_abs",
    "pf_abs_rvrt",
    "trip_power",
]
RULE_COLUMN_MAP = {
    "noncanonical": "noncanonical",
    "common_missing": "common_missing_any",
    "w_gt_wmax": "w_gt_wmax_tol",
    "w_gt_wmaxrtg": "w_gt_wmaxrtg_tol",
    "va_gt_vamax": "va_gt_vamax_tol",
    "var_gt_injmax": "var_gt_injmax_tol",
    "var_lt_absmax": "var_lt_absmax_tol",
    "wset_far": "wset_enabled_far",
    "wsetpct_far": "wsetpct_enabled_far",
    "wmaxlim_far": "wmaxlim_enabled_far",
    "varsetpct_far": "varsetpct_enabled_far",
    "model_structure": "model_structure_anomaly_any",
    "ac_type_rare": "ac_type_is_rare",
    "dc_type_rare": "dc_port_type_rare_any",
    "enter_state": "enter_service_state_anomaly",
    "enter_blocked_power": "enter_service_blocked_power",
    "enter_blocked_current": "enter_service_blocked_current",
    "pf_abs": "pf_abs_ext_present",
    "pf_abs_rvrt": "pf_abs_rvrt_ext_present",
    "trip_power": "trip_any_power_when_outside",
}
CAT_ENGINEERED_COLUMNS = [
    "device_fingerprint",
    "common_missing_pattern",
    "enter_service_missing_pattern",
    "missing_selected_total",
    "missing_selected_blocks",
    "common_missing_any",
    "common_missing_count",
    "common_sn_has_decimal_suffix",
]

EXPECTED_MODEL_META = {
    "common": ("common[0].ID", "common[0].L", 1.0, 66.0),
    "measure_ac": ("DERMeasureAC[0].ID", "DERMeasureAC[0].L", 701.0, 153.0),
    "capacity": ("DERCapacity[0].ID", "DERCapacity[0].L", 702.0, 50.0),
    "enter_service": ("DEREnterService[0].ID", "DEREnterService[0].L", 703.0, 17.0),
    "measure_dc": ("DERMeasureDC[0].ID", "DERMeasureDC[0].L", 714.0, 68.0),
}


@dataclass
class MetricSummary:
    f2: float
    precision: float
    recall: float
    positive_rate: float
    rows: int


@dataclass
class FamilySemanticContext:
    family: str
    surrogate_feature_cols: List[str]
    surrogate_models: Dict[Tuple[str, str], XGBRegressor]
    residual_quantiles: Dict[str, Dict[str, Dict[str, float]]]
    family_base_rates: Dict[str, float]
    scenario_sum_map: Dict[int, float]
    scenario_count_map: Dict[int, int]
    scenario_output_sum_map: Dict[int, float]
    scenario_output_count_map: Dict[int, int]


@dataclass
class FamilyModelBundle:
    family: str
    semantic_model: XGBClassifier
    semantic_context: FamilySemanticContext
    semantic_feature_cols: List[str]
    cat_model: Optional[CatBoostClassifier]
    cat_feature_cols: List[str]
    cat_categorical_cols: List[str]
    threshold: float
    blend_weight: float


@dataclass(frozen=True)
class RunConfig:
    train_path: Path = TRAIN_CSV_PATH
    test_path: Path = TEST_CSV_PATH
    submission_path: Path = KAGGLE_WORKING_DIR / "submission.csv"
    train_row_limit: int = 0
    test_row_limit: int = 0
    write_test_predictions: bool = True
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

    def create_baseline(self) -> "ResearchBaseline":
        return ResearchBaseline(
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


class ResearchBaseline:
    def __init__(
        self,
        *,
        chunksize: int = 5000,
        cv_folds: int = 5,
        n_estimators: int = 150,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        cat_iterations: int = 400,
        cat_depth: int = 8,
        cat_learning_rate: float = 0.05,
        n_jobs: int = 4,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.chunksize = chunksize
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.cat_iterations = cat_iterations
        self.cat_depth = cat_depth
        self.cat_learning_rate = cat_learning_rate
        self.n_jobs = n_jobs
        self.seed = seed
        self.hard_override_names = list(DEFAULT_HARD_OVERRIDE_NAMES)
        self.family_bundles: Dict[str, FamilyModelBundle] = {}
        self.surrogate_feature_cols: Optional[List[str]] = None
        self.surrogate_models: Dict[Tuple[str, str], XGBRegressor] = {}
        self.residual_quantiles: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.family_base_rates: Dict[str, float] = {}
        self.scenario_sum_map: Dict[int, float] = {}
        self.scenario_count_map: Dict[int, int] = {}
        self.scenario_output_sum_map: Dict[int, float] = {}
        self.scenario_output_count_map: Dict[int, int] = {}

    @staticmethod
    def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        out = np.full_like(a, np.nan)
        mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
        out[mask] = a[mask] / b[mask]
        return out

    @staticmethod
    def _nanmin_rows(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        mask = np.isfinite(arr)
        out = np.full(arr.shape[0], np.nan, dtype=np.float32)
        if arr.shape[1] == 0:
            return out
        reduced = np.where(mask, arr, np.inf).min(axis=1)
        valid_rows = mask.any(axis=1)
        out[valid_rows] = reduced[valid_rows]
        return out

    @staticmethod
    def _nanmax_rows(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        mask = np.isfinite(arr)
        out = np.full(arr.shape[0], np.nan, dtype=np.float32)
        if arr.shape[1] == 0:
            return out
        reduced = np.where(mask, arr, -np.inf).max(axis=1)
        valid_rows = mask.any(axis=1)
        out[valid_rows] = reduced[valid_rows]
        return out

    @staticmethod
    def _nanmean_rows(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        mask = np.isfinite(arr)
        out = np.full(arr.shape[0], np.nan, dtype=np.float32)
        counts = mask.sum(axis=1)
        valid_rows = counts > 0
        if valid_rows.any():
            totals = np.where(mask, arr, 0.0).sum(axis=1)
            out[valid_rows] = totals[valid_rows] / counts[valid_rows]
        return out

    @staticmethod
    def _curve_index(raw_idx: np.ndarray, num_options: int) -> np.ndarray:
        idx = np.nan_to_num(np.asarray(raw_idx, dtype=np.float32), nan=1.0)
        idx = idx.astype(np.int16) - 1
        idx[(idx < 0) | (idx >= num_options)] = 0
        return idx.astype(np.int8)

    @staticmethod
    def _select_curve_scalar(curves: Sequence[np.ndarray], idx: np.ndarray) -> np.ndarray:
        stacked = np.stack(curves, axis=1)
        return np.take_along_axis(stacked, idx[:, None], axis=1)[:, 0]

    @staticmethod
    def _select_curve_points(curves: Sequence[np.ndarray], idx: np.ndarray) -> np.ndarray:
        stacked = np.stack(curves, axis=1)
        return np.take_along_axis(stacked, idx[:, None, None], axis=1)[:, 0, :]

    @staticmethod
    def _pair_point_count(x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        return (np.isfinite(np.asarray(x_points, dtype=np.float32)) & np.isfinite(np.asarray(y_points, dtype=np.float32))).sum(axis=1).astype(np.int16)

    @staticmethod
    def _curve_reverse_steps(x_points: np.ndarray) -> np.ndarray:
        x_points = np.asarray(x_points, dtype=np.float32)
        finite_pair = np.isfinite(x_points[:, :-1]) & np.isfinite(x_points[:, 1:])
        return ((np.diff(x_points, axis=1) < -1e-6) & finite_pair).sum(axis=1).astype(np.int8)

    @staticmethod
    def _curve_slope_stats(x_points: np.ndarray, y_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_points = np.asarray(x_points, dtype=np.float32)
        y_points = np.asarray(y_points, dtype=np.float32)
        dx = np.diff(x_points, axis=1)
        dy = np.diff(y_points, axis=1)
        valid = np.isfinite(x_points[:, :-1]) & np.isfinite(x_points[:, 1:]) & np.isfinite(y_points[:, :-1]) & np.isfinite(y_points[:, 1:]) & (np.abs(dx) > 1e-6)
        slopes = np.full(dx.shape, np.nan, dtype=np.float32)
        slopes[valid] = dy[valid] / dx[valid]
        return ResearchBaseline._nanmean_rows(slopes), ResearchBaseline._nanmax_rows(np.abs(slopes))

    @staticmethod
    def _piecewise_interp(x: np.ndarray, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x_points = np.asarray(x_points, dtype=np.float32)
        y_points = np.asarray(y_points, dtype=np.float32)
        n_rows, n_points = x_points.shape
        result = np.full(n_rows, np.nan, dtype=np.float32)
        valid_points = np.isfinite(x_points) & np.isfinite(y_points)
        has_valid = valid_points.any(axis=1)
        if n_points == 0:
            return result

        row_idx = np.arange(n_rows)
        first_valid = np.argmax(valid_points, axis=1)
        last_valid = n_points - 1 - np.argmax(valid_points[:, ::-1], axis=1)

        first_x = np.full(n_rows, np.nan, dtype=np.float32)
        first_y = np.full(n_rows, np.nan, dtype=np.float32)
        last_x = np.full(n_rows, np.nan, dtype=np.float32)
        last_y = np.full(n_rows, np.nan, dtype=np.float32)
        first_x[has_valid] = x_points[row_idx[has_valid], first_valid[has_valid]]
        first_y[has_valid] = y_points[row_idx[has_valid], first_valid[has_valid]]
        last_x[has_valid] = x_points[row_idx[has_valid], last_valid[has_valid]]
        last_y[has_valid] = y_points[row_idx[has_valid], last_valid[has_valid]]

        for seg in range(n_points - 1):
            x0 = x_points[:, seg]
            x1 = x_points[:, seg + 1]
            y0 = y_points[:, seg]
            y1 = y_points[:, seg + 1]
            valid_seg = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1) & (np.abs(x1 - x0) > 1e-6)
            lo = np.minimum(x0, x1)
            hi = np.maximum(x0, x1)
            mask = valid_seg & np.isfinite(x) & np.isnan(result) & (x >= lo) & (x <= hi)
            if mask.any():
                frac = (x[mask] - x0[mask]) / (x1[mask] - x0[mask])
                result[mask] = y0[mask] + frac * (y1[mask] - y0[mask])

        low_mask = has_valid & np.isfinite(x) & np.isnan(result) & (x <= np.minimum(first_x, last_x))
        result[low_mask] = first_y[low_mask]
        high_mask = has_valid & np.isfinite(x) & np.isnan(result) & (x >= np.maximum(first_x, last_x))
        result[high_mask] = last_y[high_mask]
        return result

    @staticmethod
    def _var_pct(var: np.ndarray, varmaxinj: np.ndarray, varmaxabs: np.ndarray) -> np.ndarray:
        var = np.asarray(var, dtype=np.float32)
        denom = np.where(
            var >= 0,
            np.asarray(varmaxinj, dtype=np.float32),
            np.asarray(varmaxabs, dtype=np.float32),
        )
        return 100.0 * ResearchBaseline._safe_div(var, denom)

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> None:
        for col in NUMERIC_SOURCE_COLUMNS:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    @staticmethod
    def _numeric_series(series: pd.Series) -> pd.Series:
        return pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index, copy=False)

    @staticmethod
    def _flag_array(series: pd.Series) -> np.ndarray:
        return ResearchBaseline._numeric_series(series).fillna(0).to_numpy(np.int8) == 1

    @staticmethod
    def _float_arrays(df: pd.DataFrame, prefix: str, fields: Sequence[str]) -> Dict[str, np.ndarray]:
        return {field: df[f"{prefix}.{field}"].to_numpy(float) for field in fields}

    @staticmethod
    def _curve_scalars(
        df: pd.DataFrame,
        prefix: str,
        field: str,
        *,
        count: int,
        item_label: str = "Crv",
        group: Optional[str] = None,
    ) -> List[np.ndarray]:
        group_suffix = f".{group}" if group else ""
        return [df[f"{prefix}.{item_label}[{idx}]{group_suffix}.{field}"].to_numpy(float) for idx in range(count)]

    @staticmethod
    def _curve_points(
        df: pd.DataFrame,
        prefix: str,
        field: str,
        *,
        count: int,
        point_count: int,
        item_label: str = "Crv",
        group: Optional[str] = None,
    ) -> List[np.ndarray]:
        group_suffix = f".{group}" if group else ""
        return [
            np.column_stack([df[f"{prefix}.{item_label}[{idx}]{group_suffix}.Pt[{point_idx}].{field}"].to_numpy(float) for point_idx in range(point_count)]) for idx in range(count)
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return FeatureBuilder(self.hard_override_names).build(df)

    def iter_raw_chunks(
        self,
        csv_path: Path,
        usecols: Sequence[str],
        limit_rows: int = 0,
    ) -> Iterator[pd.DataFrame]:
        yielded = 0
        for chunk in pd.read_csv(
            csv_path,
            usecols=list(usecols),
            chunksize=self.chunksize,
            low_memory=False,
        ):
            if limit_rows > 0:
                remaining = limit_rows - yielded
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            yielded += len(chunk)
            yield chunk
            if limit_rows <= 0:
                continue
            if yielded >= limit_rows:
                break

    @staticmethod
    def _encode_device_family_values(values: np.ndarray) -> np.ndarray:
        family_values = np.asarray(values, dtype=object)
        encoded = np.full(len(family_values), -1.0, dtype=np.float32)
        encoded[family_values == "canon10"] = 0.0
        encoded[family_values == "canon100"] = 1.0
        return encoded

    @staticmethod
    def _get_surrogate_feature_cols(columns: Sequence[str]) -> List[str]:
        excluded = {
            "Id",
            "Label",
            "fold_id",
            "audit_fold_id",
            "device_fingerprint",
            "hard_rule_anomaly",
            "hard_rule_count",
            "hard_rule_score",
            "hard_override_anomaly",
        }
        excluded.update(SAFE_STR.values())
        return [col for col in columns if col not in excluded and col not in SURROGATE_LEAKY_FEATURES]

    @staticmethod
    def _build_sample_weights(x_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        weights = np.ones(len(x_df), dtype=np.float32)
        family = x_df["device_family"].to_numpy()
        hard_override = x_df["hard_override_anomaly"].to_numpy() == 1
        weights[(family == "canon100") & (y == 0)] *= CANON100_NEGATIVE_WEIGHT
        weights[hard_override] *= HARD_OVERRIDE_TRAIN_WEIGHT
        return weights

    @staticmethod
    def _bucketize(
        values: pd.Series,
        *,
        fill_value: int | float,
        dtype: np.dtype,
        scale: float = 1.0,
        round_values: bool = True,
    ) -> pd.Series:
        out = ResearchBaseline._numeric_series(values)
        if scale != 1.0:
            out = out / scale
        if round_values:
            out = out.round()
        return out.fillna(fill_value).astype(dtype)

    def _build_scenario_frame(self, x_df: pd.DataFrame, *, include_output_bins: bool) -> pd.DataFrame:
        frame: Dict[str, pd.Series] = {
            "family": x_df["device_family"].astype(str),
            "throt_src": self._bucketize(x_df["DERMeasureAC_0_ThrotSrc"], fill_value=-1, dtype=np.int16),
            "throt_pct": self._bucketize(
                x_df["DERMeasureAC_0_ThrotPct"],
                scale=5.0,
                fill_value=-1,
                dtype=np.int16,
            ),
            "wmaxlim_pct": self._bucketize(x_df["DERCtlAC_0_WMaxLimPct"], scale=5.0, fill_value=-1, dtype=np.int16),
            "wset_pct": self._bucketize(x_df["DERCtlAC_0_WSetPct"], scale=5.0, fill_value=-1, dtype=np.int16),
            "varset_pct": self._bucketize(x_df["DERCtlAC_0_VarSetPct"], scale=5.0, fill_value=-1, dtype=np.int16),
            "pf_set": self._bucketize(x_df["DERCtlAC_0_PFWInj_PF"], scale=0.02, fill_value=-1, dtype=np.int16),
            "fd_idx": self._bucketize(x_df["DERFreqDroop_0_AdptCtlRslt"], fill_value=-1, dtype=np.int16),
            "vv_idx": self._bucketize(x_df["DERVoltVar_0_AdptCrvRslt"], fill_value=-1, dtype=np.int16),
            "vw_idx": self._bucketize(x_df["DERVoltWatt_0_AdptCrvRslt"], fill_value=-1, dtype=np.int16),
            "wv_idx": self._bucketize(x_df["DERWattVar_0_AdptCrvRslt"], fill_value=-1, dtype=np.int16),
            "volt_bin": self._bucketize(x_df["voltage_pct"], fill_value=-999, dtype=np.int16),
            "hz_bin": self._bucketize(x_df["DERMeasureAC_0_Hz"], scale=0.1, fill_value=-999, dtype=np.int16),
            "enter_idle": self._bucketize(
                x_df["enter_service_should_idle"],
                fill_value=0,
                dtype=np.int8,
                round_values=False,
            ),
            "droop_active": self._bucketize(
                x_df["freqdroop_outside_deadband"],
                fill_value=0,
                dtype=np.int8,
                round_values=False,
            ),
        }
        if include_output_bins:
            frame["w_bin"] = self._bucketize(x_df["w_pct_of_rtg"], scale=5.0, fill_value=-999, dtype=np.int16)
            frame["var_bin"] = self._bucketize(
                x_df["var_pct_of_limit"],
                scale=5.0,
                fill_value=-999,
                dtype=np.int16,
            )
            frame["pf_mode"] = self._bucketize(
                x_df["pf_control_any_enabled"],
                fill_value=0,
                dtype=np.int8,
                round_values=False,
            )
        return pd.DataFrame(frame)

    @staticmethod
    def _hash_frame(frame: pd.DataFrame) -> np.ndarray:
        return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)

    def _build_scenario_keys(self, x_df: pd.DataFrame) -> np.ndarray:
        return self._hash_frame(self._build_scenario_frame(x_df, include_output_bins=False))

    def _build_scenario_output_keys(self, x_df: pd.DataFrame) -> np.ndarray:
        return self._hash_frame(self._build_scenario_frame(x_df, include_output_bins=True))

    @staticmethod
    def _lookup_scenario_stats(
        keys: np.ndarray,
        sum_map: Dict[int, float],
        count_map: Dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        key_series = pd.Series(keys)
        sum_values = key_series.map(sum_map).fillna(0.0).to_numpy(np.float32)
        count_values = key_series.map(count_map).fillna(0).to_numpy(np.int32)
        return sum_values, count_values

    @staticmethod
    def _assign_scenario_features(
        out: pd.DataFrame,
        *,
        family_prior: np.ndarray,
        scenario_rate: np.ndarray,
        scenario_count: np.ndarray,
        scenario_output_rate: np.ndarray,
        scenario_output_count: np.ndarray,
    ) -> pd.DataFrame:
        out["scenario_rate"] = scenario_rate.astype(np.float32)
        out["scenario_rate_delta"] = (scenario_rate - family_prior).astype(np.float32)
        out["scenario_count"] = scenario_count.astype(np.int32)
        out["scenario_log_count"] = np.log1p(scenario_count).astype(np.float32)
        out["scenario_low_support"] = (scenario_count < 20).astype(np.int8)
        out["scenario_output_rate"] = scenario_output_rate.astype(np.float32)
        out["scenario_output_rate_delta"] = (scenario_output_rate - family_prior).astype(np.float32)
        out["scenario_output_count"] = scenario_output_count.astype(np.int32)
        out["scenario_output_log_count"] = np.log1p(scenario_output_count).astype(np.float32)
        out["scenario_output_low_support"] = (scenario_output_count < 20).astype(np.int8)
        return out

    def _fit_transform_scenario_features(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        y_arr = y_train.to_numpy(np.float32)
        family_series = x_train["device_family"].astype(str)
        family_rates = pd.DataFrame({"family": family_series, "y": y_arr}).groupby("family")["y"].mean()
        self.family_base_rates = {
            str(name): float(rate)
            for name, rate in zip(
                family_rates.index.astype(str),
                family_rates.to_numpy(dtype=np.float32, copy=False),
            )
        }
        keys = self._build_scenario_keys(x_train)
        output_keys = self._build_scenario_output_keys(x_train)
        fold_ids = (x_train["Id"].to_numpy(np.int64) % self.cv_folds).astype(np.int8)
        scenario_rate = np.zeros(len(x_train), dtype=np.float32)
        scenario_count = np.zeros(len(x_train), dtype=np.int32)
        scenario_output_rate = np.zeros(len(x_train), dtype=np.float32)
        scenario_output_count = np.zeros(len(x_train), dtype=np.int32)
        global_rate = float(np.mean(y_arr))

        for fold in range(self.cv_folds):
            train_mask = fold_ids != fold
            valid_mask = fold_ids == fold
            if not valid_mask.any():
                continue
            stats = pd.DataFrame({"key": keys[train_mask], "y": y_arr[train_mask]}).groupby("key")["y"].agg(["sum", "count"])
            output_stats = pd.DataFrame({"key": output_keys[train_mask], "y": y_arr[train_mask]}).groupby("key")["y"].agg(["sum", "count"])
            valid_keys = pd.Series(keys[valid_mask])
            valid_sum = valid_keys.map(stats["sum"]).fillna(0.0).to_numpy(np.float32)
            valid_count = valid_keys.map(stats["count"]).fillna(0).to_numpy(np.int32)
            valid_output_keys = pd.Series(output_keys[valid_mask])
            valid_output_sum = valid_output_keys.map(output_stats["sum"]).fillna(0.0).to_numpy(np.float32)
            valid_output_count = valid_output_keys.map(output_stats["count"]).fillna(0).to_numpy(np.int32)
            valid_family = family_series.loc[valid_mask].tolist()
            prior = np.array(
                [self.family_base_rates.get(name, global_rate) for name in valid_family],
                dtype=np.float32,
            )
            scenario_rate[valid_mask] = (valid_sum + SCENARIO_SMOOTHING * prior) / (valid_count + SCENARIO_SMOOTHING)
            scenario_count[valid_mask] = valid_count
            scenario_output_rate[valid_mask] = (valid_output_sum + SCENARIO_SMOOTHING * prior) / (valid_output_count + SCENARIO_SMOOTHING)
            scenario_output_count[valid_mask] = valid_output_count

        full_stats = pd.DataFrame({"key": keys, "y": y_arr}).groupby("key")["y"].agg(["sum", "count"])
        full_output_stats = pd.DataFrame({"key": output_keys, "y": y_arr}).groupby("key")["y"].agg(["sum", "count"])
        full_stat_keys = full_stats.index.to_numpy(dtype=np.uint64, copy=False)
        full_output_keys_arr = full_output_stats.index.to_numpy(dtype=np.uint64, copy=False)
        self.scenario_sum_map = {int(idx): float(val) for idx, val in zip(full_stat_keys, full_stats["sum"].to_numpy(dtype=np.float64, copy=False))}
        self.scenario_count_map = {int(idx): int(val) for idx, val in zip(full_stat_keys, full_stats["count"].to_numpy(dtype=np.int64, copy=False))}
        self.scenario_output_sum_map = {
            int(idx): float(val)
            for idx, val in zip(
                full_output_keys_arr,
                full_output_stats["sum"].to_numpy(dtype=np.float64, copy=False),
            )
        }
        self.scenario_output_count_map = {
            int(idx): int(val)
            for idx, val in zip(
                full_output_keys_arr,
                full_output_stats["count"].to_numpy(dtype=np.int64, copy=False),
            )
        }

        family_prior = family_series.map(self.family_base_rates).fillna(global_rate).to_numpy(np.float32)
        return self._assign_scenario_features(
            x_train,
            family_prior=family_prior,
            scenario_rate=scenario_rate,
            scenario_count=scenario_count,
            scenario_output_rate=scenario_output_rate,
            scenario_output_count=scenario_output_count,
        )

    def _apply_scenario_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.scenario_count_map:
            return x_df

        keys = self._build_scenario_keys(x_df)
        output_keys = self._build_scenario_output_keys(x_df)
        sum_values, count_values = self._lookup_scenario_stats(
            keys,
            self.scenario_sum_map,
            self.scenario_count_map,
        )
        output_sum_values, output_count_values = self._lookup_scenario_stats(
            output_keys,
            self.scenario_output_sum_map,
            self.scenario_output_count_map,
        )
        global_rate = float(np.mean(list(self.family_base_rates.values()))) if self.family_base_rates else 0.5
        family_prior = x_df["device_family"].astype(str).map(self.family_base_rates).fillna(global_rate).to_numpy(np.float32)
        scenario_rate = (sum_values + SCENARIO_SMOOTHING * family_prior) / (count_values + SCENARIO_SMOOTHING)
        scenario_output_rate = (output_sum_values + SCENARIO_SMOOTHING * family_prior) / (output_count_values + SCENARIO_SMOOTHING)
        return self._assign_scenario_features(
            x_df,
            family_prior=family_prior,
            scenario_rate=scenario_rate,
            scenario_count=count_values,
            scenario_output_rate=scenario_output_rate,
            scenario_output_count=output_count_values,
        )

    @staticmethod
    def _add_family_interaction_features(x_df: pd.DataFrame) -> pd.DataFrame:
        canon100_mask = (x_df["device_family"].astype(str) == "canon100").to_numpy(dtype=bool, copy=False)
        for feature_name in CANON100_INTERACTION_FEATURES:
            if feature_name not in x_df.columns:
                continue
            values = ResearchBaseline._numeric_series(x_df[feature_name]).to_numpy(np.float32)
            x_df[f"canon100_{feature_name}"] = np.where(canon100_mask, values, 0.0).astype(np.float32)
        return x_df

    @staticmethod
    def _surrogate_partition_mask(ids: Sequence[int], *, fit_partition: bool) -> np.ndarray:
        ids_arr = np.asarray(ids, dtype=np.int64)
        fit_mask = (ids_arr % 2) == 0
        return fit_mask if fit_partition else ~fit_mask

    def _xgb_shared_params(self, *, eval_metric: str, verbosity: int) -> Dict[str, object]:
        return {
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "eval_metric": eval_metric,
            "tree_method": "hist",
            "n_jobs": self.n_jobs,
            "random_state": self.seed,
            "seed": self.seed,
            "verbosity": verbosity,
        }

    def _new_surrogate_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=max(80, self.n_estimators // 2),
            max_depth=max(4, self.max_depth - 2),
            learning_rate=min(0.08, self.learning_rate * 1.2),
            objective="reg:squarederror",
            **self._xgb_shared_params(eval_metric="rmse", verbosity=0),
        )

    def _new_classifier(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            **self._xgb_shared_params(eval_metric="logloss", verbosity=1),
        )

    @staticmethod
    def _fold_stage_name(fold_col: str) -> str:
        return "primary" if fold_col == "fold_id" else "audit"

    def _cat_verbose_interval(self) -> int:
        return max(1, self.cat_iterations // 5)

    def _log_boost_fit(
        self,
        *,
        model_name: str,
        family: str,
        stage: str,
        split_label: str,
        train_rows: int,
        valid_rows: Optional[int],
        feature_count: int,
        categorical_count: int = 0,
    ) -> None:
        parts = [
            f"[{model_name}]",
            f"[{family}]",
            f"[{stage}]",
            split_label,
            f"train_rows={train_rows:,}",
            f"features={feature_count:,}",
        ]
        if valid_rows is not None:
            parts.append(f"valid_rows={valid_rows:,}")
        if categorical_count:
            parts.append(f"categorical={categorical_count:,}")
        LOGGER.info(" ".join(parts))

    @staticmethod
    def _float32_feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
        return np.ascontiguousarray(df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False))

    def _surrogate_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        row_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mask = None if row_mask is None else np.asarray(row_mask, dtype=bool)
        row_count = len(df) if mask is None else int(mask.sum())
        matrix = np.empty((row_count, len(feature_cols)), dtype=np.float32)
        for col_idx, col in enumerate(feature_cols):
            values = df[col].to_numpy(copy=False)
            if mask is not None:
                values = values[mask]
            if col == "device_family":
                matrix[:, col_idx] = self._encode_device_family_values(values)
            else:
                matrix[:, col_idx] = np.asarray(values, dtype=np.float32)
        return matrix

    def _fit_surrogate_models(self, x_train: pd.DataFrame, y_train: pd.Series, valid_mask: pd.Series) -> None:
        self.surrogate_feature_cols = self._get_surrogate_feature_cols(x_train.columns)
        fit_partition = self._surrogate_partition_mask(x_train["Id"], fit_partition=True)
        normal_mask = (y_train == 0) & (x_train["hard_override_anomaly"] == 0) & (x_train["device_family"] != "other") & (~valid_mask.to_numpy()) & fit_partition
        if not normal_mask.any():
            raise RuntimeError("No rows available to train surrogate models.")

        self.surrogate_models = {}
        surrogate_feature_cols = list(self.surrogate_feature_cols)
        for family in DEVICE_FAMILY_MAP:
            family_mask = normal_mask & (x_train["device_family"] == family)
            if not family_mask.any():
                continue
            x_surrogate_matrix = self._surrogate_feature_matrix(
                x_train,
                surrogate_feature_cols,
                family_mask,
            )
            for target_name, (target_col, _) in SURROGATE_TARGETS.items():
                model = self._new_surrogate_model()
                y_target = x_train.loc[family_mask, target_col].to_numpy(np.float32)
                LOGGER.info(f"[surrogate] training {family}/{target_name} on {int(family_mask.sum()):,} normal rows")
                model.fit(x_surrogate_matrix, y_target)
                self.surrogate_models[(family, target_name)] = model

    def _augment_with_surrogates(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if self.surrogate_feature_cols is None or not self.surrogate_models:
            return x_df

        for target_name in SURROGATE_TARGETS:
            x_df[f"pred_{target_name}"] = np.nan
            x_df[f"resid_{target_name}"] = np.nan
            x_df[f"abs_resid_{target_name}"] = np.nan
            x_df[f"norm_resid_{target_name}"] = np.nan
            x_df[f"abs_norm_resid_{target_name}"] = np.nan
            x_df[f"tail_resid_{target_name}"] = 0
            x_df[f"extreme_resid_{target_name}"] = 0
            x_df[f"ultra_resid_{target_name}"] = 0
            x_df[f"q99_ratio_resid_{target_name}"] = np.nan
        surrogate_feature_cols = list(self.surrogate_feature_cols)
        x_surrogate_matrix = self._surrogate_feature_matrix(x_df, surrogate_feature_cols)
        family_values = x_df["device_family"].to_numpy(dtype=object, copy=False)
        for family in DEVICE_FAMILY_MAP:
            family_mask = family_values == family
            if not np.any(family_mask):
                continue
            x_family = x_surrogate_matrix[family_mask]
            for target_name, (target_col, scale_col) in SURROGATE_TARGETS.items():
                model = self.surrogate_models.get((family, target_name))
                if model is None:
                    continue
                pred = model.predict(x_family).astype(np.float32)
                actual = x_df.loc[family_mask, target_col].to_numpy(np.float32)
                resid = actual - pred
                x_df.loc[family_mask, f"pred_{target_name}"] = pred
                x_df.loc[family_mask, f"resid_{target_name}"] = resid
                x_df.loc[family_mask, f"abs_resid_{target_name}"] = np.abs(resid).astype(np.float32)
                if scale_col is not None:
                    scale = x_df.loc[family_mask, scale_col].to_numpy(np.float32)
                    norm_resid = self._safe_div(resid, scale)
                else:
                    scale = np.maximum(0.05, np.abs(actual))
                    norm_resid = (resid / scale).astype(np.float32)
                x_df.loc[family_mask, f"norm_resid_{target_name}"] = norm_resid.astype(np.float32)
                x_df.loc[family_mask, f"abs_norm_resid_{target_name}"] = np.abs(norm_resid).astype(np.float32)

        x_df["resid_energy_total"] = (
            x_df[
                [
                    "abs_resid_w",
                    "abs_resid_va",
                    "abs_resid_var",
                    "abs_resid_pf",
                    "abs_resid_a",
                ]
            ]
            .sum(axis=1)
            .astype(np.float32)
        )
        x_df["resid_va_minus_pq"] = (x_df["pred_va"] - np.sqrt(np.square(x_df["pred_w"]) + np.square(x_df["pred_var"]))).astype(np.float32)
        x_df["resid_w_var_ratio"] = self._safe_div(
            x_df["abs_resid_w"].to_numpy(float),
            x_df["abs_resid_var"].to_numpy(float) + 1e-3,
        )
        return x_df

    def _compute_residual_quantiles(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        valid_mask: pd.Series,
    ) -> None:
        calibration_partition = self._surrogate_partition_mask(x_train["Id"], fit_partition=False)
        base_mask = (y_train == 0) & (x_train["hard_override_anomaly"] == 0) & (x_train["device_family"] != "other") & (~valid_mask.to_numpy())
        self.residual_quantiles = {}
        for family in DEVICE_FAMILY_MAP:
            family_mask = base_mask & (x_train["device_family"] == family)
            family_calibration = family_mask & calibration_partition
            if not family_calibration.any():
                family_calibration = family_mask
            family_quantiles: Dict[str, Dict[str, float]] = {}
            for target_name in SURROGATE_TARGETS:
                series = x_train.loc[family_calibration, f"abs_norm_resid_{target_name}"]
                values = self._numeric_series(series).to_numpy(np.float32)
                values = values[np.isfinite(values)]
                quantiles = dict(RESIDUAL_TAIL_FALLBACKS)
                if values.size > 0:
                    for level_name, q in RESIDUAL_TAIL_LEVELS.items():
                        quantiles[level_name] = float(np.quantile(values, q))
                family_quantiles[target_name] = {key: max(1e-6, value) for key, value in quantiles.items()}
            self.residual_quantiles[family] = family_quantiles

    def _apply_residual_calibration_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.residual_quantiles:
            return x_df

        for target_name in SURROGATE_TARGETS:
            x_df[f"tail_resid_{target_name}"] = 0
            x_df[f"extreme_resid_{target_name}"] = 0
            x_df[f"ultra_resid_{target_name}"] = 0
            x_df[f"q99_ratio_resid_{target_name}"] = np.nan

        for family in DEVICE_FAMILY_MAP:
            family_mask = x_df["device_family"] == family
            if not family_mask.any():
                continue
            family_quantiles = self.residual_quantiles.get(family, {})
            for target_name in SURROGATE_TARGETS:
                abs_norm = x_df.loc[family_mask, f"abs_norm_resid_{target_name}"].to_numpy(np.float32)
                q = family_quantiles.get(target_name, RESIDUAL_TAIL_FALLBACKS)
                tail = abs_norm >= q["tail"]
                extreme = abs_norm >= q["extreme"]
                ultra = abs_norm >= q["ultra"]
                q99_ratio = self._safe_div(abs_norm, np.full_like(abs_norm, q["extreme"], dtype=np.float32))
                x_df.loc[family_mask, f"tail_resid_{target_name}"] = tail.astype(np.int8)
                x_df.loc[family_mask, f"extreme_resid_{target_name}"] = extreme.astype(np.int8)
                x_df.loc[family_mask, f"ultra_resid_{target_name}"] = ultra.astype(np.int8)
                x_df.loc[family_mask, f"q99_ratio_resid_{target_name}"] = q99_ratio.astype(np.float32)

        abs_norm_w = np.nan_to_num(x_df["abs_norm_resid_w"].to_numpy(np.float32), nan=0.0)
        abs_norm_var = np.nan_to_num(x_df["abs_norm_resid_var"].to_numpy(np.float32), nan=0.0)
        abs_norm_pf = np.nan_to_num(x_df["abs_norm_resid_pf"].to_numpy(np.float32), nan=0.0)
        abs_norm_a = np.nan_to_num(x_df["abs_norm_resid_a"].to_numpy(np.float32), nan=0.0)
        pf_mode = np.nan_to_num(x_df["pf_control_any_enabled"].to_numpy(np.float32), nan=0.0) > 0
        voltvar_mode = (np.nan_to_num(x_df["DERVoltVar_0_Ena"].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(x_df["voltvar_curve_expected"].to_numpy(np.float32))
        voltwatt_mode = (np.nan_to_num(x_df["DERVoltWatt_0_Ena"].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(x_df["voltwatt_curve_expected"].to_numpy(np.float32))
        wattvar_mode = (np.nan_to_num(x_df["DERWattVar_0_Ena"].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(x_df["wattvar_curve_expected"].to_numpy(np.float32))
        droop_mode = np.nan_to_num(x_df["freqdroop_outside_deadband"].to_numpy(np.float32), nan=0.0) > 0
        enter_idle_mode = np.nan_to_num(x_df["enter_service_should_idle"].to_numpy(np.float32), nan=0.0) > 0

        x_df["mode_resid_pf_pf"] = (abs_norm_pf * pf_mode).astype(np.float32)
        x_df["mode_resid_var_pf"] = (abs_norm_var * pf_mode).astype(np.float32)
        x_df["mode_resid_var_voltvar"] = (abs_norm_var * voltvar_mode).astype(np.float32)
        x_df["mode_resid_w_voltwatt"] = (abs_norm_w * voltwatt_mode).astype(np.float32)
        x_df["mode_resid_var_wattvar"] = (abs_norm_var * wattvar_mode).astype(np.float32)
        x_df["mode_resid_w_droop"] = (abs_norm_w * droop_mode).astype(np.float32)
        x_df["mode_resid_w_enter_idle"] = (abs_norm_w * enter_idle_mode).astype(np.float32)
        x_df["mode_resid_a_enter_idle"] = (abs_norm_a * enter_idle_mode).astype(np.float32)
        x_df["mode_curve_var_resid"] = (abs_norm_var * (voltvar_mode | wattvar_mode | pf_mode)).astype(np.float32)
        x_df["mode_dispatch_w_resid"] = (abs_norm_w * (voltwatt_mode | droop_mode | enter_idle_mode)).astype(np.float32)
        x_df["mode_extreme_var_curve"] = (np.nan_to_num(x_df["extreme_resid_var"].to_numpy(np.float32), nan=0.0) * (voltvar_mode | wattvar_mode | pf_mode)).astype(np.int8)
        x_df["mode_extreme_w_dispatch"] = (np.nan_to_num(x_df["extreme_resid_w"].to_numpy(np.float32), nan=0.0) * (voltwatt_mode | droop_mode | enter_idle_mode)).astype(np.int8)
        x_df["mode_tail_count"] = (
            x_df[
                [
                    "mode_extreme_var_curve",
                    "mode_extreme_w_dispatch",
                ]
            ]
            .sum(axis=1)
            .astype(np.int8)
        )
        x_df["resid_tail_count"] = (
            x_df[
                [
                    "tail_resid_w",
                    "tail_resid_va",
                    "tail_resid_var",
                    "tail_resid_pf",
                    "tail_resid_a",
                ]
            ]
            .sum(axis=1)
            .astype(np.int8)
        )
        x_df["resid_extreme_count"] = (
            x_df[
                [
                    "extreme_resid_w",
                    "extreme_resid_va",
                    "extreme_resid_var",
                    "extreme_resid_pf",
                    "extreme_resid_a",
                ]
            ]
            .sum(axis=1)
            .astype(np.int8)
        )
        x_df["resid_ultra_count"] = (
            x_df[
                [
                    "ultra_resid_w",
                    "ultra_resid_va",
                    "ultra_resid_var",
                    "ultra_resid_pf",
                    "ultra_resid_a",
                ]
            ]
            .sum(axis=1)
            .astype(np.int8)
        )
        x_df["resid_quantile_score"] = (
            x_df[
                [
                    "q99_ratio_resid_w",
                    "q99_ratio_resid_va",
                    "q99_ratio_resid_var",
                    "q99_ratio_resid_pf",
                    "q99_ratio_resid_a",
                ]
            ]
            .sum(axis=1)
            .astype(np.float32)
        )
        return x_df

    @staticmethod
    def tune_threshold(
        y_true: np.ndarray,
        prob: np.ndarray,
        *,
        low: float = FAMILY_THRESHOLD_FLOOR,
        high: float = MAX_THRESHOLD,
        step: float = 0.01,
    ) -> Tuple[float, float]:
        if len(y_true) == 0:
            return 0.5, 0.0
        thresholds = np.arange(low, high + 1e-9, step, dtype=np.float32)
        y_true_arr = np.asarray(y_true, dtype=np.int8)
        prob_arr = np.asarray(prob, dtype=np.float32)
        y_positive = y_true_arr == 1
        pred_matrix = prob_arr[:, None] >= thresholds[None, :]
        tp = np.count_nonzero(pred_matrix & y_positive[:, None], axis=0).astype(np.float64)
        pred_positive = pred_matrix.sum(axis=0, dtype=np.int64).astype(np.float64)
        positive_count = float(np.count_nonzero(y_positive))
        fp = pred_positive - tp
        fn = positive_count - tp
        denom = 5.0 * tp + 4.0 * fn + fp
        scores = np.divide(
            5.0 * tp,
            denom,
            out=np.zeros_like(tp, dtype=np.float64),
            where=denom > 0.0,
        )
        best_idx = int(np.argmax(scores))
        return float(thresholds[best_idx]), float(scores[best_idx])

    @staticmethod
    def _binary_classification_counts(y_true: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int]:
        y_true_arr = np.asarray(y_true, dtype=np.int8)
        pred_arr = np.asarray(pred, dtype=np.int8)
        positive_true = y_true_arr == 1
        positive_pred = pred_arr == 1
        tp = int(np.count_nonzero(positive_true & positive_pred))
        fp = int(np.count_nonzero(~positive_true & positive_pred))
        fn = int(np.count_nonzero(positive_true & ~positive_pred))
        return tp, fp, fn

    @staticmethod
    def _f2_from_counts(tp: int, fp: int, fn: int) -> float:
        denom = 5 * tp + 4 * fn + fp
        if denom == 0:
            return 0.0
        return float((5.0 * tp) / denom)

    @staticmethod
    def _f2_from_pred(y_true: np.ndarray, pred: np.ndarray) -> float:
        return ResearchBaseline._f2_from_counts(*ResearchBaseline._binary_classification_counts(y_true, pred))

    @staticmethod
    def _metric_summary_from_pred(y_true: np.ndarray, pred: np.ndarray) -> MetricSummary:
        if len(y_true) == 0:
            return MetricSummary(f2=0.0, precision=0.0, recall=0.0, positive_rate=0.0, rows=0)
        tp, fp, fn = ResearchBaseline._binary_classification_counts(y_true, pred)
        pred_positive = tp + fp
        true_positive = tp + fn
        return MetricSummary(
            f2=ResearchBaseline._f2_from_counts(tp, fp, fn),
            precision=float(tp / pred_positive) if pred_positive else 0.0,
            recall=float(tp / true_positive) if true_positive else 0.0,
            positive_rate=float(np.mean(pred)),
            rows=int(len(y_true)),
        )

    @staticmethod
    def _blend_probs(primary: np.ndarray, secondary: Optional[np.ndarray], weight: float) -> np.ndarray:
        if secondary is None:
            return primary.astype(np.float32)
        return (weight * primary + (1.0 - weight) * secondary).astype(np.float32)

    @staticmethod
    def _select_nonconstant_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
        keep: List[str] = []
        for col in candidates:
            if col not in df.columns:
                continue
            series = df[col]
            if series.notna().sum() == 0:
                continue
            if series.nunique(dropna=True) <= 1:
                continue
            keep.append(col)
        return keep

    @staticmethod
    def _report_rows_to_metric(rows: pd.DataFrame, pred_col: str) -> MetricSummary:
        return ResearchBaseline._metric_summary_from_pred(
            rows["Label"].to_numpy(np.int8),
            rows[pred_col].to_numpy(np.int8),
        )

    def _new_cat_model(self) -> "CatBoostClassifier":
        return CatBoostClassifier(
            iterations=self.cat_iterations,
            depth=self.cat_depth,
            learning_rate=self.cat_learning_rate,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=self.seed,
            thread_count=self.n_jobs,
            allow_writing_files=False,
        )

    def _iter_training_feature_chunks(self, train_path: Path, row_limit: int = 0) -> Iterator[pd.DataFrame]:
        for chunk in self.iter_raw_chunks(train_path, USECOLS_TRAIN, row_limit):
            labels = chunk["Label"].to_numpy(np.int8, copy=False)
            feats = self.build_features(chunk)
            feats["Label"] = labels
            feats["fold_id"] = (feats["Id"].to_numpy(np.int64, copy=False) % self.cv_folds).astype(np.int8)
            scenario_keys = self._build_scenario_keys(feats)
            feats["audit_fold_id"] = (scenario_keys % self.cv_folds).astype(np.int8)
            yield feats

    def _collect_training_family(self, train_path: Path, row_limit: int, family: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        row_count = 0
        LOGGER.info("[fit] collecting %s rows from %s", family, train_path)
        with tqdm(
            self._iter_training_feature_chunks(train_path, row_limit),
            desc=f"{family} chunks",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            for feats in progress:
                family_mask = feats["device_family"] == family
                if family_mask.any():
                    family_df = feats.loc[family_mask].reset_index(drop=True)
                    frames.append(family_df)
                    row_count += len(family_df)
                progress.set_postfix(rows=f"{row_count:,}")
        if not frames:
            return pd.DataFrame()
        family_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
        LOGGER.info("[fit] collected %s %s rows", f"{len(family_df):,}", family)
        return family_df

    def _audit_hard_override_rules(self, train_path: Path, row_limit: int = 0) -> pd.DataFrame:
        per_rule_counts = {name: {"count": 0, "positives": 0} for name in DEFAULT_HARD_OVERRIDE_NAMES}
        other_prediction_rows: List[pd.DataFrame] = []
        other_row_count = 0
        with tqdm(
            self._iter_training_feature_chunks(train_path, row_limit),
            desc="audit chunks",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            for feats in progress:
                labels = feats["Label"].to_numpy(np.int8, copy=False)
                for rule_name in DEFAULT_HARD_OVERRIDE_NAMES:
                    mask = self._flag_array(feats[RULE_COLUMN_MAP[rule_name]])
                    if not mask.any():
                        continue
                    per_rule_counts[rule_name]["count"] += int(mask.sum())
                    per_rule_counts[rule_name]["positives"] += int(labels[mask].sum())

                other_mask = feats["device_family"] == "other"
                if other_mask.any():
                    other_df = feats.loc[other_mask, ["Id", "Label"]]
                    other_prediction_rows.append(
                        pd.DataFrame(
                            {
                                "Id": other_df["Id"].to_numpy(np.int64, copy=False),
                                "Label": other_df["Label"].to_numpy(np.int8, copy=False),
                                "family": "other",
                                "pred_primary": np.ones(len(other_df), dtype=np.int8),
                                "pred_audit": np.ones(len(other_df), dtype=np.int8),
                            }
                        )
                    )
                    other_row_count += len(other_df)
                progress.set_postfix(other_rows=f"{other_row_count:,}")

        demoted: List[str] = []
        for rule_name, counts in per_rule_counts.items():
            count = counts["count"]
            positives = counts["positives"]
            precision = float(positives / count) if count else 1.0
            if count > 0 and precision < MIN_OVERRIDE_PRECISION:
                demoted.append(rule_name)
        self.hard_override_names = [name for name in DEFAULT_HARD_OVERRIDE_NAMES if name not in demoted]
        if demoted:
            LOGGER.info("[fit] demoted hard overrides: %s", ", ".join(demoted))
        if not other_prediction_rows:
            return pd.DataFrame(columns=["Id", "Label", "family", "pred_primary", "pred_audit"])
        return pd.concat(other_prediction_rows, ignore_index=True)

    def _refresh_override_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        hard_rule_flags = np.column_stack([self._flag_array(df[RULE_COLUMN_MAP[name]]) for name in HARD_RULE_NAMES])
        if self.hard_override_names:
            hard_override_flags = np.column_stack([self._flag_array(df[RULE_COLUMN_MAP[name]]) for name in self.hard_override_names])
            df["hard_override_anomaly"] = hard_override_flags.any(axis=1).astype(np.int8)
        else:
            df["hard_override_anomaly"] = np.zeros(len(df), dtype=np.int8)
        df["hard_rule_anomaly"] = hard_rule_flags.any(axis=1).astype(np.int8)
        return df

    def _capture_semantic_context(self, family: str) -> FamilySemanticContext:
        return FamilySemanticContext(
            family=family,
            surrogate_feature_cols=list(self.surrogate_feature_cols or []),
            surrogate_models=dict(self.surrogate_models),
            residual_quantiles=json.loads(json.dumps(self.residual_quantiles)),
            family_base_rates=dict(self.family_base_rates),
            scenario_sum_map=dict(self.scenario_sum_map),
            scenario_count_map=dict(self.scenario_count_map),
            scenario_output_sum_map=dict(self.scenario_output_sum_map),
            scenario_output_count_map=dict(self.scenario_output_count_map),
        )

    def _activate_semantic_context(self, context: FamilySemanticContext) -> None:
        self.surrogate_feature_cols = list(context.surrogate_feature_cols)
        self.surrogate_models = dict(context.surrogate_models)
        self.residual_quantiles = json.loads(json.dumps(context.residual_quantiles))
        self.family_base_rates = dict(context.family_base_rates)
        self.scenario_sum_map = dict(context.scenario_sum_map)
        self.scenario_count_map = dict(context.scenario_count_map)
        self.scenario_output_sum_map = dict(context.scenario_output_sum_map)
        self.scenario_output_count_map = dict(context.scenario_output_count_map)

    def _prepare_family_semantic_frame(
        self,
        base_df: pd.DataFrame,
        y: pd.Series,
        family: str,
    ) -> Tuple[pd.DataFrame, FamilySemanticContext]:
        work = self._refresh_override_columns(base_df)
        no_valid = pd.Series(np.zeros(len(work), dtype=bool), index=work.index)
        self._fit_surrogate_models(work, y, no_valid)
        work = self._augment_with_surrogates(work)
        self._compute_residual_quantiles(work, y, no_valid)
        work = self._apply_residual_calibration_features(work)
        work = self._fit_transform_scenario_features(work, y)
        work = self._add_family_interaction_features(work)
        return work, self._capture_semantic_context(family)

    @staticmethod
    def _semantic_feature_candidates(semantic_df: pd.DataFrame) -> List[str]:
        excluded = {
            "Id",
            "Label",
            "fold_id",
            "audit_fold_id",
            "hard_override_anomaly",
            "device_fingerprint",
        }
        excluded.update(SAFE_STR.values())
        return [col for col in semantic_df.columns if col not in excluded and pd.api.types.is_numeric_dtype(semantic_df[col])]

    def _prepare_cat_frame(self, base_df: pd.DataFrame) -> pd.DataFrame:
        out = self._refresh_override_columns(base_df)
        for col in [*SAFE_STR.values(), "device_fingerprint"]:
            if col in out.columns:
                out[col] = out[col].fillna("<NA>").astype(str)
        return out

    @staticmethod
    def _cat_feature_candidates(cat_df: pd.DataFrame) -> List[str]:
        raw_numeric_cols = [SAFE_RAW[col] for col in RAW_NUMERIC if SAFE_RAW[col] in cat_df.columns]
        missing_cols = [col for col in cat_df.columns if col.startswith("missing_")]
        categorical_cols = [SAFE_STR[col] for col in RAW_STRING_COLUMNS if SAFE_STR[col] in cat_df.columns]
        candidates = dedupe(
            [
                *raw_numeric_cols,
                *categorical_cols,
                "device_fingerprint",
                *CAT_ENGINEERED_COLUMNS,
                *missing_cols,
            ]
        )
        excluded = {
            "Id",
            "Label",
            "fold_id",
            "audit_fold_id",
            "hard_override_anomaly",
            "hard_rule_anomaly",
        }
        return [col for col in candidates if col in cat_df.columns and col not in excluded]

    def _train_semantic_oof(
        self,
        family: str,
        semantic_df: pd.DataFrame,
        y: np.ndarray,
        feature_cols: Sequence[str],
        *,
        fold_col: str,
        fit_final: bool,
    ) -> Tuple[np.ndarray, Optional[XGBClassifier]]:
        probs = np.ones(len(semantic_df), dtype=np.float32)
        model_mask = semantic_df["hard_override_anomaly"].to_numpy(np.int8) == 0
        fold_ids = semantic_df[fold_col].to_numpy(np.int8)
        feature_matrix = self._float32_feature_matrix(semantic_df, feature_cols)
        base_weights = self._build_sample_weights(semantic_df, y)
        final_model: Optional[XGBClassifier] = None
        stage = self._fold_stage_name(fold_col)
        feature_count = int(feature_matrix.shape[1])
        for fold in range(self.cv_folds):
            train_idx = np.flatnonzero(model_mask & (fold_ids != fold))
            valid_idx = np.flatnonzero(model_mask & (fold_ids == fold))
            if len(valid_idx) == 0:
                continue
            self._log_boost_fit(
                model_name="xgb",
                family=family,
                stage=stage,
                split_label=f"fold {fold + 1}/{self.cv_folds}",
                train_rows=len(train_idx),
                valid_rows=len(valid_idx),
                feature_count=feature_count,
            )
            model = self._new_classifier()
            model.fit(
                feature_matrix[train_idx],
                y[train_idx],
                sample_weight=base_weights[train_idx],
            )
            probs[valid_idx] = model.predict_proba(feature_matrix[valid_idx])[:, 1].astype(np.float32)
        if fit_final and model_mask.any():
            final_idx = np.flatnonzero(model_mask)
            self._log_boost_fit(
                model_name="xgb",
                family=family,
                stage=stage,
                split_label="final",
                train_rows=len(final_idx),
                valid_rows=None,
                feature_count=feature_count,
            )
            final_model = self._new_classifier()
            final_model.fit(
                feature_matrix[final_idx],
                y[final_idx],
                sample_weight=base_weights[final_idx],
            )
        return probs, final_model

    def _train_cat_oof(
        self,
        family: str,
        cat_df: pd.DataFrame,
        y: np.ndarray,
        feature_cols: Sequence[str],
        categorical_cols: Sequence[str],
        *,
        fold_col: str,
        fit_final: bool,
    ) -> Tuple[np.ndarray, Optional["CatBoostClassifier"]]:
        probs = np.ones(len(cat_df), dtype=np.float32)
        model_mask = cat_df["hard_override_anomaly"].to_numpy(np.int8) == 0
        fold_ids = cat_df[fold_col].to_numpy(np.int8)
        feature_frame = cat_df.loc[:, feature_cols]
        base_weights = self._build_sample_weights(cat_df, y)
        final_model: Optional["CatBoostClassifier"] = None
        stage = self._fold_stage_name(fold_col)
        feature_count = len(feature_cols)
        categorical_count = len(categorical_cols)
        cat_verbose = self._cat_verbose_interval()
        for fold in range(self.cv_folds):
            train_idx = np.flatnonzero(model_mask & (fold_ids != fold))
            valid_idx = np.flatnonzero(model_mask & (fold_ids == fold))
            if len(valid_idx) == 0:
                continue
            self._log_boost_fit(
                model_name="cat",
                family=family,
                stage=stage,
                split_label=f"fold {fold + 1}/{self.cv_folds}",
                train_rows=len(train_idx),
                valid_rows=len(valid_idx),
                feature_count=feature_count,
                categorical_count=categorical_count,
            )
            model = self._new_cat_model()
            model.fit(
                feature_frame.iloc[train_idx],
                y[train_idx],
                cat_features=list(categorical_cols),
                sample_weight=base_weights[train_idx],
                verbose=cat_verbose,
            )
            probs[valid_idx] = model.predict_proba(feature_frame.iloc[valid_idx])[:, 1].astype(np.float32)
        if fit_final and model_mask.any():
            final_idx = np.flatnonzero(model_mask)
            self._log_boost_fit(
                model_name="cat",
                family=family,
                stage=stage,
                split_label="final",
                train_rows=len(final_idx),
                valid_rows=None,
                feature_count=feature_count,
                categorical_count=categorical_count,
            )
            final_model = self._new_cat_model()
            final_model.fit(
                feature_frame.iloc[final_idx],
                y[final_idx],
                cat_features=list(categorical_cols),
                sample_weight=base_weights[final_idx],
                verbose=cat_verbose,
            )
        return probs, final_model

    def _select_family_blend(
        self,
        y: np.ndarray,
        hard_override: np.ndarray,
        semantic_primary: np.ndarray,
        semantic_audit: np.ndarray,
        cat_primary: Optional[np.ndarray],
        cat_audit: Optional[np.ndarray],
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        baseline_primary_prob = np.where(hard_override, np.float32(1.0), semantic_primary)
        baseline_thr, _ = self.tune_threshold(y, baseline_primary_prob)
        baseline_pred_primary = (baseline_primary_prob >= baseline_thr).astype(np.int8)
        baseline_audit_prob = np.where(hard_override, np.float32(1.0), semantic_audit)
        baseline_pred_audit = (baseline_audit_prob >= baseline_thr).astype(np.int8)
        baseline_primary_score = self._f2_from_pred(y, baseline_pred_primary)
        baseline_audit_score = self._f2_from_pred(y, baseline_pred_audit)

        best_weight = 1.0
        best_thr = baseline_thr
        best_primary_pred = baseline_pred_primary
        best_audit_pred = baseline_pred_audit
        best_primary_score = baseline_primary_score
        best_audit_for_best = baseline_audit_score

        weight_grid = [round(step / 20.0, 2) for step in range(21)] if cat_primary is not None else [1.0]
        for weight in weight_grid:
            blended_primary = self._blend_probs(semantic_primary, cat_primary, weight)
            blended_primary[hard_override] = 1.0
            thr, _ = self.tune_threshold(y, blended_primary)
            pred_primary = (blended_primary >= thr).astype(np.int8)
            blended_audit = self._blend_probs(semantic_audit, cat_audit, weight)
            blended_audit[hard_override] = 1.0
            pred_audit = (blended_audit >= thr).astype(np.int8)
            primary_score = self._f2_from_pred(y, pred_primary)
            audit_score = self._f2_from_pred(y, pred_audit)
            if audit_score < baseline_audit_score - AUDIT_TOLERANCE:
                continue
            if primary_score > best_primary_score + 1e-12 or (abs(primary_score - best_primary_score) <= 1e-12 and audit_score > best_audit_for_best):
                best_weight = weight
                best_thr = thr
                best_primary_pred = pred_primary
                best_audit_pred = pred_audit
                best_primary_score = primary_score
                best_audit_for_best = audit_score
        return (
            best_weight,
            best_thr,
            best_primary_pred.astype(np.int8),
            best_audit_pred.astype(np.int8),
        )

    def fit(self, train_path: Path, row_limit: int = 0) -> None:
        self.family_bundles.clear()
        self.hard_override_names = list(DEFAULT_HARD_OVERRIDE_NAMES)
        prediction_rows: List[pd.DataFrame] = []
        LOGGER.info("[fit] auditing hard overrides from %s", train_path)
        other_prediction_rows = self._audit_hard_override_rules(train_path, row_limit)
        if not other_prediction_rows.empty:
            prediction_rows.append(other_prediction_rows)
        del other_prediction_rows
        gc.collect()

        LOGGER.info("[fit] starting family training")
        with tqdm(
            list(DEVICE_FAMILY_MAP),
            desc="fit families",
            unit="family",
            dynamic_ncols=True,
        ) as family_progress:
            for family in family_progress:
                family_progress.set_postfix(family=family)
                base_df = self._collect_training_family(train_path, row_limit, family)
                if base_df.empty:
                    LOGGER.info("[fit] skipping %s; no training rows", family)
                    continue
                LOGGER.info("[fit] training %s on %s rows", family, f"{len(base_df):,}")
                y_series = base_df["Label"].astype(np.int8, copy=False)
                semantic_df, context = self._prepare_family_semantic_frame(base_df, y_series, family)
                semantic_feature_cols = self._select_nonconstant_columns(
                    semantic_df,
                    self._semantic_feature_candidates(semantic_df),
                )

                y = y_series.to_numpy(np.int8, copy=False)
                family_ids = semantic_df["Id"].to_numpy(np.int64, copy=False)
                hard_override = semantic_df["hard_override_anomaly"].to_numpy(np.int8) == 1

                semantic_primary_prob, semantic_model = self._train_semantic_oof(
                    family,
                    semantic_df,
                    y,
                    semantic_feature_cols,
                    fold_col="fold_id",
                    fit_final=True,
                )
                semantic_audit_prob, _ = self._train_semantic_oof(
                    family,
                    semantic_df,
                    y,
                    semantic_feature_cols,
                    fold_col="audit_fold_id",
                    fit_final=False,
                )
                if semantic_model is None:
                    raise RuntimeError(f"Semantic model training produced no fitted model for family {family}.")

                cat_primary_prob: Optional[np.ndarray] = None
                cat_audit_prob: Optional[np.ndarray] = None
                cat_model: Optional["CatBoostClassifier"] = None
                cat_df = self._prepare_cat_frame(semantic_df)
                cat_feature_cols = self._select_nonconstant_columns(cat_df, self._cat_feature_candidates(cat_df))
                cat_categorical_cols = [col for col in [*SAFE_STR.values(), "device_fingerprint"] if col in cat_feature_cols]

                if cat_feature_cols:
                    cat_primary_prob, cat_model = self._train_cat_oof(
                        family,
                        cat_df,
                        y,
                        cat_feature_cols,
                        cat_categorical_cols,
                        fold_col="fold_id",
                        fit_final=True,
                    )
                    cat_audit_prob, _ = self._train_cat_oof(
                        family,
                        cat_df,
                        y,
                        cat_feature_cols,
                        cat_categorical_cols,
                        fold_col="audit_fold_id",
                        fit_final=False,
                    )

                weight, threshold, primary_pred, audit_pred = self._select_family_blend(
                    y,
                    hard_override,
                    semantic_primary_prob,
                    semantic_audit_prob,
                    cat_primary_prob,
                    cat_audit_prob,
                )
                self.family_bundles[family] = FamilyModelBundle(
                    family=family,
                    semantic_model=semantic_model,
                    semantic_context=context,
                    semantic_feature_cols=list(semantic_feature_cols),
                    cat_model=cat_model,
                    cat_feature_cols=list(cat_feature_cols),
                    cat_categorical_cols=list(cat_categorical_cols),
                    threshold=threshold,
                    blend_weight=weight,
                )

                family_rows = pd.DataFrame(
                    {
                        "Id": family_ids,
                        "Label": y,
                        "family": family,
                        "pred_primary": primary_pred,
                        "pred_audit": audit_pred,
                    }
                )
                prediction_rows.append(family_rows)
                family_primary_metric = self._report_rows_to_metric(family_rows, "pred_primary")
                family_audit_metric = self._report_rows_to_metric(family_rows, "pred_audit")
                LOGGER.info(
                    f"[fit] {family} primary F2={family_primary_metric.f2:.6f}, audit F2={family_audit_metric.f2:.6f}, threshold={threshold:.3f}, blend_weight={weight:.2f}"
                )
                del (
                    base_df,
                    semantic_df,
                    cat_df,
                    y_series,
                    y,
                    family_ids,
                    hard_override,
                    semantic_primary_prob,
                    semantic_audit_prob,
                    cat_primary_prob,
                    cat_audit_prob,
                    family_rows,
                )
                gc.collect()

        if not prediction_rows:
            raise RuntimeError("No training rows were available for model fitting.")
        all_predictions = pd.concat(prediction_rows, ignore_index=True)
        overall_primary_metric = self._report_rows_to_metric(all_predictions, "pred_primary")
        overall_audit_metric = self._report_rows_to_metric(all_predictions, "pred_audit")
        LOGGER.info(f"[fit] overall primary F2={overall_primary_metric.f2:.6f}, precision={overall_primary_metric.precision:.4f}, recall={overall_primary_metric.recall:.4f}")
        LOGGER.info(f"[fit] overall audit F2={overall_audit_metric.f2:.6f}, precision={overall_audit_metric.precision:.4f}, recall={overall_audit_metric.recall:.4f}")

    def _predict_family_chunk(self, family: str, base_df: pd.DataFrame) -> np.ndarray:
        bundle = self.family_bundles.get(family)
        if bundle is None:
            raise RuntimeError(f"Missing fitted semantic model bundle for family {family}.")
        self._activate_semantic_context(bundle.semantic_context)
        semantic_df = self._refresh_override_columns(base_df)
        hard_override = semantic_df["hard_override_anomaly"].to_numpy(np.int8) == 1
        semantic_df = self._augment_with_surrogates(semantic_df)
        semantic_df = self._apply_residual_calibration_features(semantic_df)
        semantic_df = self._apply_scenario_features(semantic_df)
        semantic_df = self._add_family_interaction_features(semantic_df)
        semantic_prob = np.ones(len(semantic_df), dtype=np.float32)
        active_idx = np.flatnonzero(np.logical_not(hard_override))
        if len(active_idx):
            semantic_feature_matrix = self._float32_feature_matrix(semantic_df, bundle.semantic_feature_cols)
            semantic_prob[active_idx] = bundle.semantic_model.predict_proba(semantic_feature_matrix[active_idx])[:, 1].astype(np.float32)

        cat_prob: Optional[np.ndarray] = None
        if bundle.cat_model is not None and bundle.cat_feature_cols:
            cat_df = self._prepare_cat_frame(base_df)
            cat_prob = np.ones(len(cat_df), dtype=np.float32)
            if len(active_idx):
                cat_rows = cat_df.iloc[active_idx]
                cat_features = cat_rows.loc[:, bundle.cat_feature_cols]
                cat_prob[active_idx] = bundle.cat_model.predict_proba(cat_features)[:, 1].astype(np.float32)
        blend_prob = self._blend_probs(semantic_prob, cat_prob, bundle.blend_weight)
        blend_prob[hard_override] = 1.0
        pred = (blend_prob >= bundle.threshold).astype(np.int8)
        pred[hard_override] = 1
        return pred

    def predict_test(self, test_path: Path, out_csv: Path, row_limit: int = 0) -> None:
        if not self.family_bundles:
            raise RuntimeError("Model is not fitted.")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        positive_rows = 0
        LOGGER.info("[test] generating predictions from %s", test_path)
        with out_csv.open("w", encoding="utf-8") as fh:
            fh.write("Id,Label\n")
            with tqdm(
                self.iter_raw_chunks(test_path, USECOLS_TEST, row_limit),
                desc="test chunks",
                unit="chunk",
                dynamic_ncols=True,
            ) as progress:
                for chunk in progress:
                    feats = self.build_features(chunk)
                    pred = feats["hard_override_anomaly"].astype(np.int8).to_numpy()
                    for family in DEVICE_FAMILY_MAP:
                        if family not in self.family_bundles:
                            continue
                        family_mask = feats["device_family"] == family
                        if not family_mask.any():
                            continue
                        family_df = feats.loc[family_mask].reset_index(drop=True)
                        family_pred = self._predict_family_chunk(family, family_df)
                        pred[np.flatnonzero(family_mask.to_numpy())] = family_pred
                    out = pd.DataFrame(
                        {
                            "Id": feats["Id"].astype(np.int64),
                            "Label": pred.astype(np.int8),
                        }
                    )
                    out.to_csv(fh, index=False, header=False)
                    total_rows += len(out)
                    positive_rows += int(out["Label"].sum())
                    progress.set_postfix(
                        rows=f"{total_rows:,}",
                        positives=f"{positive_rows:,}",
                    )
        LOGGER.info(f"[test] done; total_rows={total_rows:,}, positive_rows={positive_rows:,}, positive_rate={positive_rows / max(total_rows, 1):.6f}")


class FeatureBuilder:
    _coerce_numeric = staticmethod(ResearchBaseline._coerce_numeric)
    _safe_div = staticmethod(ResearchBaseline._safe_div)
    _nanmin_rows = staticmethod(ResearchBaseline._nanmin_rows)
    _nanmax_rows = staticmethod(ResearchBaseline._nanmax_rows)
    _nanmean_rows = staticmethod(ResearchBaseline._nanmean_rows)
    _curve_index = staticmethod(ResearchBaseline._curve_index)
    _select_curve_scalar = staticmethod(ResearchBaseline._select_curve_scalar)
    _select_curve_points = staticmethod(ResearchBaseline._select_curve_points)
    _pair_point_count = staticmethod(ResearchBaseline._pair_point_count)
    _curve_reverse_steps = staticmethod(ResearchBaseline._curve_reverse_steps)
    _curve_slope_stats = staticmethod(ResearchBaseline._curve_slope_stats)
    _piecewise_interp = staticmethod(ResearchBaseline._piecewise_interp)
    _var_pct = staticmethod(ResearchBaseline._var_pct)
    _float_arrays = staticmethod(ResearchBaseline._float_arrays)
    _curve_scalars = staticmethod(ResearchBaseline._curve_scalars)
    _curve_points = staticmethod(ResearchBaseline._curve_points)

    def __init__(self, hard_override_names: Sequence[str]) -> None:
        self.hard_override_names = list(hard_override_names)

    @staticmethod
    def _add_block_health_features(data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
        block_missing_total = np.zeros(len(df), dtype=np.int16)
        block_missing_any = np.zeros(len(df), dtype=np.int16)
        for block_name, block_cols in BLOCK_SOURCE_COLUMNS.items():
            missing = df[block_cols].isna()
            missing_count = missing.sum(axis=1).astype(np.int16).to_numpy()
            data[f"missing_{block_name}_count"] = missing_count
            data[f"missing_{block_name}_any"] = (missing_count > 0).astype(np.int8)
            block_missing_total += missing_count
            block_missing_any += (missing_count > 0).astype(np.int16)
        data["missing_selected_total"] = block_missing_total
        data["missing_selected_blocks"] = block_missing_any.astype(np.int8)
        common_missing = df[[*COMMON_STR, "common[0].ID", "common[0].L"]].isna().to_numpy(dtype=np.uint16)
        common_weights = (1 << np.arange(common_missing.shape[1], dtype=np.uint16)).reshape(1, -1)
        data["common_missing_pattern"] = (common_missing * common_weights).sum(axis=1).astype(np.int16)
        enter_missing = df[ENTER_SERVICE_COLUMNS].isna().to_numpy(dtype=np.uint16)
        enter_weights = (1 << np.arange(enter_missing.shape[1], dtype=np.uint16)).reshape(1, -1)
        data["enter_service_missing_pattern"] = (enter_missing * enter_weights).sum(axis=1).astype(np.int16)
        anomaly_sum = np.zeros(len(df), dtype=np.int16)
        missing_sum = np.zeros(len(df), dtype=np.int16)
        for block_name, (
            id_col,
            len_col,
            expected_id,
            expected_len,
        ) in EXPECTED_MODEL_META.items():
            raw_id = df[id_col].to_numpy(float)
            raw_len = df[len_col].to_numpy(float)
            id_missing = ~np.isfinite(raw_id)
            len_missing = ~np.isfinite(raw_len)
            id_match = np.isclose(raw_id, expected_id, equal_nan=False)
            len_match = np.isclose(raw_len, expected_len, equal_nan=False)
            data[f"{block_name}_model_id_missing"] = id_missing.astype(np.int8)
            data[f"{block_name}_model_len_missing"] = len_missing.astype(np.int8)
            data[f"{block_name}_model_id_match"] = id_match.astype(np.int8)
            data[f"{block_name}_model_len_match"] = len_match.astype(np.int8)
            data[f"{block_name}_model_integrity_ok"] = (id_match & len_match).astype(np.int8)
            mismatch = (~id_missing & ~id_match) | (~len_missing & ~len_match)
            data[f"{block_name}_model_structure_anomaly"] = mismatch.astype(np.int8)
            anomaly_sum += mismatch.astype(np.int16)
            missing_sum += (id_missing | len_missing).astype(np.int16)
        data["model_structure_anomaly_count"] = anomaly_sum.astype(np.int8)
        data["model_structure_missing_count"] = missing_sum.astype(np.int8)
        data["model_structure_anomaly_any"] = (anomaly_sum > 0).astype(np.int8)

    def _add_temperature_features(self, data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
        temp_cols = [
            "DERMeasureAC[0].TmpAmb",
            "DERMeasureAC[0].TmpCab",
            "DERMeasureAC[0].TmpSnk",
            "DERMeasureAC[0].TmpTrns",
            "DERMeasureAC[0].TmpSw",
            "DERMeasureAC[0].TmpOt",
        ]
        temps = df[temp_cols].to_numpy(float)
        temp_min = self._nanmin_rows(temps)
        temp_max = self._nanmax_rows(temps)
        temp_mean = self._nanmean_rows(temps)
        amb = df["DERMeasureAC[0].TmpAmb"].to_numpy(float)
        data["temp_min"] = temp_min
        data["temp_max"] = temp_max
        data["temp_mean"] = temp_mean
        data["temp_spread"] = (temp_max - temp_min).astype(np.float32)
        data["temp_max_over_ambient"] = (temp_max - amb).astype(np.float32)

    def _add_power_capacity_features(self, data: Dict[str, np.ndarray], df: pd.DataFrame) -> Dict[str, np.ndarray]:
        ac = self._float_arrays(
            df,
            "DERMeasureAC[0]",
            ("ACType", "W", "VA", "Var", "PF", "A", "LLV", "LNV", "Hz"),
        )
        capacity = self._float_arrays(
            df,
            "DERCapacity[0]",
            (
                "WMaxRtg",
                "VAMaxRtg",
                "VarMaxInjRtg",
                "VarMaxAbsRtg",
                "WMax",
                "VAMax",
                "VarMaxInj",
                "VarMaxAbs",
                "AMax",
                "VNom",
                "VMax",
                "VMin",
                "VNomRtg",
                "VMaxRtg",
                "VMinRtg",
                "AMaxRtg",
                "WChaRteMaxRtg",
                "WDisChaRteMaxRtg",
                "VAChaRteMaxRtg",
                "VADisChaRteMaxRtg",
                "WChaRteMax",
                "WDisChaRteMax",
                "VAChaRteMax",
                "VADisChaRteMax",
                "PFOvrExtRtg",
                "PFOvrExt",
                "PFUndExtRtg",
                "PFUndExt",
            ),
        )
        ctl = self._float_arrays(
            df,
            "DERCtlAC[0]",
            (
                "WSetEna",
                "WSet",
                "WSetPct",
                "WMaxLimPctEna",
                "WMaxLimPct",
                "VarSetEna",
                "VarSet",
                "VarSetPct",
            ),
        )

        w = ac["W"]
        abs_w = np.abs(w)
        va = ac["VA"]
        var = ac["Var"]
        pf = ac["PF"]
        a = ac["A"]
        llv = ac["LLV"]
        lnv = ac["LNV"]
        hz = ac["Hz"]
        lnv_sqrt3 = lnv * SQRT3

        wmaxrtg = capacity["WMaxRtg"]
        vamaxrtg = capacity["VAMaxRtg"]
        varmaxinjrtg = capacity["VarMaxInjRtg"]
        varmaxabsrtg = capacity["VarMaxAbsRtg"]
        wmax = capacity["WMax"]
        vamax = capacity["VAMax"]
        varmaxinj = capacity["VarMaxInj"]
        varmaxabs = capacity["VarMaxAbs"]
        amax = capacity["AMax"]
        vnom = capacity["VNom"]
        vmax = capacity["VMax"]
        vmin = capacity["VMin"]

        for name, numerator, denominator in (
            ("w_over_wmaxrtg", w, wmaxrtg),
            ("w_over_wmax", w, wmax),
            ("va_over_vamax", va, vamax),
            ("va_over_vamaxrtg", va, vamaxrtg),
            ("var_over_injmax", var, varmaxinj),
            ("var_over_absmax", var, varmaxabs),
            ("a_over_amax", a, amax),
            ("llv_over_vnom", llv, vnom),
            ("lnv_over_vnom", lnv_sqrt3, vnom),
        ):
            data[name] = self._safe_div(numerator, denominator)

        for name, value in (
            ("w_minus_wmax", w - wmax),
            ("w_minus_wmaxrtg", w - wmaxrtg),
            ("va_minus_vamax", va - vamax),
            ("var_minus_injmax", var - varmaxinj),
            ("var_plus_absmax", var + varmaxabs),
            ("llv_minus_lnv_sqrt3", llv - lnv_sqrt3),
            ("hz_delta_60", hz - 60.0),
        ):
            data[name] = value.astype(np.float32)

        for name, left, right in (
            ("w_eq_wmaxrtg", w, wmaxrtg),
            ("w_eq_wmax", w, wmax),
            ("var_eq_varmaxinj", var, varmaxinj),
            ("var_eq_neg_varmaxabs", var, -varmaxabs),
        ):
            data[name] = np.isclose(left, right, equal_nan=False).astype(np.int8)
        data["pf_sign_mismatch"] = ((np.sign(np.nan_to_num(pf)) != np.sign(np.nan_to_num(w))) & (np.nan_to_num(pf) != 0) & (np.nan_to_num(w) != 0)).astype(np.int8)

        tolw = np.maximum(50.0, 0.02 * np.nan_to_num(wmaxrtg, nan=0.0)).astype(np.float32)
        tolva = np.maximum(50.0, 0.02 * np.nan_to_num(vamax, nan=0.0)).astype(np.float32)
        tolvi = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxinj, nan=0.0)).astype(np.float32)
        tolva2 = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxabs, nan=0.0)).astype(np.float32)
        for name, value, upper_bound in (
            ("w_gt_wmax_tol", w, wmax + tolw),
            ("w_gt_wmaxrtg_tol", w, wmaxrtg + tolw),
            ("va_gt_vamax_tol", va, vamax + tolva),
            ("var_gt_injmax_tol", var, varmaxinj + tolvi),
        ):
            data[name] = (value > upper_bound).astype(np.int8)
        data["var_lt_absmax_tol"] = (var < (-varmaxabs - tolva2)).astype(np.int8)

        pq_mag = np.sqrt(np.square(w.astype(np.float32)) + np.square(var.astype(np.float32)))
        data["va_minus_pqmag"] = (va - pq_mag).astype(np.float32)
        data["va_over_pqmag"] = self._safe_div(va, pq_mag)
        pf_from_w_va = self._safe_div(w, va)
        data["pf_from_w_va"] = pf_from_w_va
        data["pf_error"] = (pf - pf_from_w_va).astype(np.float32)

        for name, total, suffixes in (
            ("w_phase_sum_error", w, ("WL1", "WL2", "WL3")),
            ("va_phase_sum_error", va, ("VAL1", "VAL2", "VAL3")),
            ("var_phase_sum_error", var, ("VarL1", "VarL2", "VarL3")),
        ):
            phase_sum = sum(df[f"DERMeasureAC[0].{suffix}"].to_numpy(float) for suffix in suffixes)
            data[name] = (total - phase_sum).astype(np.float32)
        for name, suffixes in (
            ("phase_ll_spread", ("VL1L2", "VL2L3", "VL3L1")),
            ("phase_ln_spread", ("VL1", "VL2", "VL3")),
            ("phase_w_spread", ("WL1", "WL2", "WL3")),
            ("phase_var_spread", ("VarL1", "VarL2", "VarL3")),
        ):
            phase_values = df[[f"DERMeasureAC[0].{suffix}" for suffix in suffixes]].to_numpy(float)
            data[name] = (self._nanmax_rows(phase_values) - self._nanmin_rows(phase_values)).astype(np.float32)

        for name, numerator, denominator in (
            ("wmax_over_wmaxrtg", wmax, wmaxrtg),
            ("vamax_over_vamaxrtg", vamax, vamaxrtg),
            ("vmax_over_vnom", vmax, vnom),
            ("vmin_over_vnom", vmin, vnom),
        ):
            data[name] = self._safe_div(numerator, denominator)

        wsetena = np.nan_to_num(ctl["WSetEna"], nan=0.0)
        wset_abs_error = np.where(wsetena > 0, np.abs(w - ctl["WSet"]), np.nan)
        wsetpct_target = wmaxrtg * (ctl["WSetPct"] / 100.0)
        wsetpct_abs_error = np.where(wsetena > 0, np.abs(w - wsetpct_target), np.nan)
        wmaxlimena = np.nan_to_num(ctl["WMaxLimPctEna"], nan=0.0)
        wmaxlim_target = wmaxrtg * (ctl["WMaxLimPct"] / 100.0)
        wmaxlim_excess = np.where(wmaxlimena > 0, w - wmaxlim_target, np.nan)
        varsetena = np.nan_to_num(ctl["VarSetEna"], nan=0.0)
        varset_abs_error = np.where(varsetena > 0, np.abs(var - ctl["VarSet"]), np.nan)
        varsetpct_target = varmaxinj * (ctl["VarSetPct"] / 100.0)
        varsetpct_abs_error = np.where(varsetena > 0, np.abs(var - varsetpct_target), np.nan)

        data["wset_abs_error"] = wset_abs_error.astype(np.float32)
        data["wsetpct_target"] = wsetpct_target.astype(np.float32)
        data["wsetpct_abs_error"] = wsetpct_abs_error.astype(np.float32)
        data["wmaxlim_target"] = wmaxlim_target.astype(np.float32)
        data["wmaxlim_excess"] = wmaxlim_excess.astype(np.float32)
        data["varset_abs_error"] = varset_abs_error.astype(np.float32)
        data["varsetpct_target"] = varsetpct_target.astype(np.float32)
        data["varsetpct_abs_error"] = varsetpct_abs_error.astype(np.float32)
        data["wset_enabled_far"] = ((wsetena > 0) & (wset_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data["wsetpct_enabled_far"] = ((wsetena > 0) & (wsetpct_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data["wmaxlim_enabled_far"] = ((wmaxlimena > 0) & (wmaxlim_excess > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data["varsetpct_enabled_far"] = ((varsetena > 0) & (varsetpct_abs_error > np.maximum(20.0, 0.05 * np.nan_to_num(varmaxinj, nan=0.0)))).astype(np.int8)

        for name, setting, rating in (
            ("vnom_setting_delta", capacity["VNom"], capacity["VNomRtg"]),
            ("vmax_setting_delta", capacity["VMax"], capacity["VMaxRtg"]),
            ("vmin_setting_delta", capacity["VMin"], capacity["VMinRtg"]),
            ("amax_setting_delta", capacity["AMax"], capacity["AMaxRtg"]),
            ("pfover_setting_delta", capacity["PFOvrExt"], capacity["PFOvrExtRtg"]),
            ("pfunder_setting_delta", capacity["PFUndExt"], capacity["PFUndExtRtg"]),
        ):
            data[name] = (setting - rating).astype(np.float32)
        for name, numerator, denominator in (
            ("charge_rate_share_rtg", capacity["WChaRteMaxRtg"], wmaxrtg),
            ("discharge_rate_share_rtg", capacity["WDisChaRteMaxRtg"], wmaxrtg),
            ("charge_va_share_rtg", capacity["VAChaRteMaxRtg"], vamaxrtg),
            ("discharge_va_share_rtg", capacity["VADisChaRteMaxRtg"], vamaxrtg),
            ("charge_rate_share_setting", capacity["WChaRteMax"], wmax),
            ("discharge_rate_share_setting", capacity["WDisChaRteMax"], wmax),
            ("charge_va_share_setting", capacity["VAChaRteMax"], vamax),
            ("discharge_va_share_setting", capacity["VADisChaRteMax"], vamax),
        ):
            data[name] = self._safe_div(numerator, denominator)

        gap_count = np.zeros(len(df), dtype=np.int16)
        for rating, setting in (
            (wmaxrtg, wmax),
            (vamaxrtg, vamax),
            (varmaxinjrtg, varmaxinj),
            (varmaxabsrtg, varmaxabs),
            (capacity["VNomRtg"], capacity["VNom"]),
            (capacity["VMaxRtg"], capacity["VMax"]),
            (capacity["VMinRtg"], capacity["VMin"]),
            (capacity["AMaxRtg"], capacity["AMax"]),
        ):
            tol = np.maximum(1.0, 0.01 * np.nan_to_num(np.abs(rating), nan=0.0)).astype(np.float32)
            gap = np.isfinite(rating) & np.isfinite(setting) & (np.abs(setting - rating) > tol)
            gap_count += gap.astype(np.int16)
        data["rating_setting_gap_count"] = gap_count.astype(np.int8)

        voltage_pct = 100.0 * self._safe_div(llv, vnom)
        line_neutral_voltage_pct = 100.0 * self._safe_div(lnv_sqrt3, vnom)
        w_pct = 100.0 * self._safe_div(w, wmaxrtg)
        var_pct = self._var_pct(var, varmaxinj, varmaxabs)
        data["voltage_pct"] = voltage_pct.astype(np.float32)
        data["line_neutral_voltage_pct"] = line_neutral_voltage_pct.astype(np.float32)
        data["w_pct_of_rtg"] = w_pct.astype(np.float32)
        data["var_pct_of_limit"] = var_pct.astype(np.float32)
        return {
            "ac_type": ac["ACType"],
            "w": w,
            "abs_w": abs_w,
            "va": va,
            "var": var,
            "pf": pf,
            "a": a,
            "hz": hz,
            "amax": amax,
            "varmaxinj": varmaxinj,
            "varmaxabs": varmaxabs,
            "voltage_pct": voltage_pct,
            "w_pct": w_pct,
            "var_pct": var_pct,
            "tolw": tolw,
            "tolva": tolva,
        }

    def _add_gate_control_features(
        self,
        data: Dict[str, np.ndarray],
        df: pd.DataFrame,
        *,
        derived: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        enter = self._float_arrays(
            df,
            "DEREnterService[0]",
            (
                "ES",
                "ESVHi",
                "ESVLo",
                "ESHzHi",
                "ESHzLo",
                "ESDlyTms",
                "ESRndTms",
                "ESRmpTms",
                "ESDlyRemTms",
            ),
        )
        voltage_pct = derived["voltage_pct"]
        hz = derived["hz"]
        abs_w = derived["abs_w"]
        va = derived["va"]
        a = derived["a"]
        tolw = derived["tolw"]
        tolva = derived["tolva"]

        inside_v = np.isfinite(voltage_pct) & np.isfinite(enter["ESVHi"]) & np.isfinite(enter["ESVLo"]) & (voltage_pct >= enter["ESVLo"]) & (voltage_pct <= enter["ESVHi"])
        inside_hz = np.isfinite(hz) & np.isfinite(enter["ESHzHi"]) & np.isfinite(enter["ESHzLo"]) & (hz >= enter["ESHzLo"]) & (hz <= enter["ESHzHi"])
        inside_window = inside_v & inside_hz
        enabled = np.isfinite(enter["ES"]) & (enter["ES"] == 1.0)
        state_anomaly = np.isfinite(enter["ES"]) & (enter["ES"] >= 1.5)
        should_idle = (~enabled) | (~inside_window)
        current_tol = np.maximum(1.0, 0.02 * np.nan_to_num(derived["amax"], nan=0.0))

        data["enter_service_enabled"] = enabled.astype(np.int8)
        data["enter_service_state_anomaly"] = state_anomaly.astype(np.int8)
        data["enter_service_inside_window"] = inside_window.astype(np.int8)
        data["enter_service_outside_window"] = (~inside_window).astype(np.int8)
        data["enter_service_should_idle"] = should_idle.astype(np.int8)
        data["enter_service_v_window_width"] = (enter["ESVHi"] - enter["ESVLo"]).astype(np.float32)
        data["enter_service_hz_window_width"] = (enter["ESHzHi"] - enter["ESHzLo"]).astype(np.float32)
        data["enter_service_v_margin_low"] = (voltage_pct - enter["ESVLo"]).astype(np.float32)
        data["enter_service_v_margin_high"] = (enter["ESVHi"] - voltage_pct).astype(np.float32)
        data["enter_service_hz_margin_low"] = (hz - enter["ESHzLo"]).astype(np.float32)
        data["enter_service_hz_margin_high"] = (enter["ESHzHi"] - hz).astype(np.float32)
        data["enter_service_total_delay"] = (enter["ESDlyTms"] + enter["ESRndTms"]).astype(np.float32)
        data["enter_service_delay_remaining"] = enter["ESDlyRemTms"].astype(np.float32)
        data["enter_service_ramp_time"] = enter["ESRmpTms"].astype(np.float32)
        data["enter_service_delay_active"] = (np.nan_to_num(enter["ESDlyRemTms"], nan=0.0) > 0).astype(np.int8)

        blocked_power = should_idle & (abs_w > tolw)
        blocked_va = should_idle & (va > tolva)
        blocked_current = should_idle & (a > current_tol)
        data["enter_service_blocked_power"] = blocked_power.astype(np.int8)
        data["enter_service_blocked_va"] = blocked_va.astype(np.int8)
        data["enter_service_blocked_current"] = blocked_current.astype(np.int8)

        pf_ctl = self._float_arrays(
            df,
            "DERCtlAC[0]",
            (
                "PFWInjEna",
                "PFWInjEnaRvrt",
                "PFWAbsEna",
                "PFWAbsEnaRvrt",
                "PFWInj.PF",
                "PFWInjRvrt.PF",
                "PFWInj.Ext",
                "PFWInjRvrt.Ext",
                "PFWAbs.Ext",
                "PFWAbsRvrt.Ext",
            ),
        )
        pfinj_ena = np.nan_to_num(pf_ctl["PFWInjEna"], nan=0.0)
        pfinj_ena_rvrt = np.nan_to_num(pf_ctl["PFWInjEnaRvrt"], nan=0.0)
        pfabs_ena = np.nan_to_num(pf_ctl["PFWAbsEna"], nan=0.0)
        pfabs_ena_rvrt = np.nan_to_num(pf_ctl["PFWAbsEnaRvrt"], nan=0.0)
        inj_target_error = np.where(
            (pfinj_ena > 0) & np.isfinite(pf_ctl["PFWInj.PF"]),
            np.abs(np.abs(derived["pf"]) - pf_ctl["PFWInj.PF"]),
            np.nan,
        )
        inj_rvrt_error = np.where(
            (pfinj_ena_rvrt > 0) & np.isfinite(pf_ctl["PFWInjRvrt.PF"]),
            np.abs(np.abs(derived["pf"]) - pf_ctl["PFWInjRvrt.PF"]),
            np.nan,
        )
        observed_var_pct = self._var_pct(derived["var"], derived["varmaxinj"], derived["varmaxabs"])
        data["pf_control_any_enabled"] = ((pfinj_ena > 0) | (pfabs_ena > 0)).astype(np.int8)
        data["pf_control_any_reversion"] = ((pfinj_ena_rvrt > 0) | (pfabs_ena_rvrt > 0)).astype(np.int8)
        data["pf_inj_target_error"] = inj_target_error.astype(np.float32)
        data["pf_inj_reversion_error"] = inj_rvrt_error.astype(np.float32)
        data["pf_inj_ext_present"] = np.isfinite(pf_ctl["PFWInj.Ext"]).astype(np.int8)
        data["pf_inj_rvrt_ext_present"] = np.isfinite(pf_ctl["PFWInjRvrt.Ext"]).astype(np.int8)
        data["pf_abs_ext_present"] = np.isfinite(pf_ctl["PFWAbs.Ext"]).astype(np.int8)
        data["pf_abs_rvrt_ext_present"] = np.isfinite(pf_ctl["PFWAbsRvrt.Ext"]).astype(np.int8)
        data["pf_inj_enabled_missing_target"] = ((pfinj_ena > 0) & ~np.isfinite(pf_ctl["PFWInj.PF"])).astype(np.int8)
        data["pf_reactive_near_limit"] = (np.abs(observed_var_pct) >= 95.0).astype(np.int8)
        return {
            "enter_state": state_anomaly.astype(np.int8),
            "enter_blocked_power": blocked_power.astype(np.int8),
            "enter_blocked_current": blocked_current.astype(np.int8),
            "pf_abs": np.isfinite(pf_ctl["PFWAbs.Ext"]).astype(np.int8),
            "pf_abs_rvrt": np.isfinite(pf_ctl["PFWAbsRvrt.Ext"]).astype(np.int8),
        }

    def _add_trip_features(
        self,
        data: Dict[str, np.ndarray],
        df: pd.DataFrame,
        *,
        voltage_pct: np.ndarray,
        hz: np.ndarray,
        abs_w: np.ndarray,
        tolw: np.ndarray,
    ) -> np.ndarray:
        trip_outside_flags: List[np.ndarray] = []
        trip_power_flags: List[np.ndarray] = []

        for short_name, (prefix, axis_name, mode) in TRIP_SPECS.items():
            measure_value = voltage_pct if axis_name == "V" else hz
            adpt_idx = self._curve_index(df[f"{prefix}.AdptCrvRslt"].to_numpy(float), 2)
            must_actpt = self._select_curve_scalar(
                self._curve_scalars(df, prefix, "ActPt", count=2, group="MustTrip"),
                adpt_idx,
            )
            mom_actpt = self._select_curve_scalar(
                self._curve_scalars(df, prefix, "ActPt", count=2, group="MomCess"),
                adpt_idx,
            )
            must_x = self._select_curve_points(
                self._curve_points(df, prefix, axis_name, count=2, point_count=5, group="MustTrip"),
                adpt_idx,
            )
            must_t = self._select_curve_points(
                self._curve_points(df, prefix, "Tms", count=2, point_count=5, group="MustTrip"),
                adpt_idx,
            )
            mom_x = self._select_curve_points(
                self._curve_points(df, prefix, axis_name, count=2, point_count=5, group="MomCess"),
                adpt_idx,
            )
            mom_t = self._select_curve_points(
                self._curve_points(df, prefix, "Tms", count=2, point_count=5, group="MomCess"),
                adpt_idx,
            )
            may_present = np.column_stack([df[f"{prefix}.Crv[{curve}].MayTrip.Pt[{point}].{axis_name}"].to_numpy(float) for curve in range(2) for point in range(5)])

            enabled = np.nan_to_num(df[f"{prefix}.Ena"].to_numpy(float), nan=0.0) > 0
            must_count = self._pair_point_count(must_x, must_t)
            mom_count = self._pair_point_count(mom_x, mom_t)
            must_x_min = self._nanmin_rows(must_x)
            must_x_max = self._nanmax_rows(must_x)
            must_t_min = self._nanmin_rows(must_t)
            must_t_max = self._nanmax_rows(must_t)
            mom_x_min = self._nanmin_rows(mom_x)
            mom_x_max = self._nanmax_rows(mom_x)
            mom_t_min = self._nanmin_rows(mom_t)
            mom_t_max = self._nanmax_rows(mom_t)
            margin = measure_value - must_x_max if mode == "low" else must_x_min - measure_value
            outside = enabled & np.isfinite(margin) & (margin < 0)
            power_when_outside = outside & (abs_w > tolw)
            envelope_gap = np.where(
                np.isfinite(mom_x_min) & np.isfinite(must_x_max),
                np.abs(mom_x_min - must_x_max),
                np.nan,
            )

            data[f"trip_{short_name}_curve_idx"] = adpt_idx.astype(np.int8)
            data[f"trip_{short_name}_enabled"] = enabled.astype(np.int8)
            data[f"trip_{short_name}_curve_req_gap"] = (df[f"{prefix}.AdptCrvReq"].to_numpy(float) - df[f"{prefix}.AdptCrvRslt"].to_numpy(float)).astype(np.float32)
            data[f"trip_{short_name}_musttrip_count"] = must_count
            data[f"trip_{short_name}_musttrip_actpt_gap"] = (must_actpt - must_count).astype(np.float32)
            data[f"trip_{short_name}_musttrip_axis_min"] = must_x_min
            data[f"trip_{short_name}_musttrip_axis_max"] = must_x_max
            data[f"trip_{short_name}_musttrip_axis_span"] = (must_x_max - must_x_min).astype(np.float32)
            data[f"trip_{short_name}_musttrip_tms_span"] = (must_t_max - must_t_min).astype(np.float32)
            data[f"trip_{short_name}_musttrip_reverse_steps"] = self._curve_reverse_steps(must_x)
            data[f"trip_{short_name}_momcess_count"] = mom_count
            data[f"trip_{short_name}_momcess_actpt_gap"] = (mom_actpt - mom_count).astype(np.float32)
            data[f"trip_{short_name}_momcess_axis_span"] = (mom_x_max - mom_x_min).astype(np.float32)
            data[f"trip_{short_name}_momcess_tms_span"] = (mom_t_max - mom_t_min).astype(np.float32)
            data[f"trip_{short_name}_momcess_reverse_steps"] = self._curve_reverse_steps(mom_x)
            data[f"trip_{short_name}_maytrip_present_any"] = np.isfinite(may_present).any(axis=1).astype(np.int8)
            data[f"trip_{short_name}_musttrip_margin"] = margin.astype(np.float32)
            data[f"trip_{short_name}_outside_musttrip"] = outside.astype(np.int8)
            data[f"trip_{short_name}_power_when_outside"] = power_when_outside.astype(np.int8)
            data[f"trip_{short_name}_momcess_musttrip_gap"] = envelope_gap.astype(np.float32)
            trip_outside_flags.append(outside)
            trip_power_flags.append(power_when_outside)

        trip_any_outside = np.column_stack(trip_outside_flags).any(axis=1).astype(np.int8)
        trip_any_power_when_outside = np.column_stack(trip_power_flags).any(axis=1).astype(np.int8)
        data["trip_any_outside_musttrip"] = trip_any_outside
        data["trip_any_power_when_outside"] = trip_any_power_when_outside
        return trip_any_power_when_outside

    def _add_curve_control_features(
        self,
        data: Dict[str, np.ndarray],
        df: pd.DataFrame,
        *,
        voltage_pct: np.ndarray,
        w_pct: np.ndarray,
        var_pct: np.ndarray,
        hz: np.ndarray,
    ) -> None:
        def add_curve(
            name: str,
            prefix: str,
            *,
            point_count: int,
            x_field: str,
            y_field: str,
            meta_fields: Dict[str, str],
            measure_value: np.ndarray,
            observed_value: np.ndarray,
        ) -> None:
            adpt_idx = self._curve_index(df[f"{prefix}.AdptCrvRslt"].to_numpy(float), 3)
            selected_x = self._select_curve_points(
                self._curve_points(df, prefix, x_field, count=3, point_count=point_count),
                adpt_idx,
            )
            selected_y = self._select_curve_points(
                self._curve_points(df, prefix, y_field, count=3, point_count=point_count),
                adpt_idx,
            )
            selected_actpt = self._select_curve_scalar(self._curve_scalars(df, prefix, "ActPt", count=3), adpt_idx)
            data[f"{name}_curve_idx"] = adpt_idx.astype(np.int8)
            point_count_arr = self._pair_point_count(selected_x, selected_y)
            data[f"{name}_curve_point_count"] = point_count_arr
            data[f"{name}_curve_actpt_gap"] = (selected_actpt - point_count_arr).astype(np.float32)
            x_min = self._nanmin_rows(selected_x)
            x_max = self._nanmax_rows(selected_x)
            y_min = self._nanmin_rows(selected_y)
            y_max = self._nanmax_rows(selected_y)
            mean_slope, max_abs_slope = self._curve_slope_stats(selected_x, selected_y)
            data[f"{name}_curve_x_span"] = (x_max - x_min).astype(np.float32)
            data[f"{name}_curve_y_span"] = (y_max - y_min).astype(np.float32)
            data[f"{name}_curve_reverse_steps"] = self._curve_reverse_steps(selected_x)
            data[f"{name}_curve_mean_slope"] = mean_slope
            data[f"{name}_curve_max_abs_slope"] = max_abs_slope
            data[f"{name}_curve_measure_margin_low"] = (measure_value - x_min).astype(np.float32)
            data[f"{name}_curve_measure_margin_high"] = (x_max - measure_value).astype(np.float32)
            expected_value = self._piecewise_interp(measure_value, selected_x, selected_y)
            data[f"{name}_curve_expected"] = expected_value.astype(np.float32)
            data[f"{name}_curve_error"] = (observed_value - expected_value).astype(np.float32)
            for meta_name, field in meta_fields.items():
                data[f"{name}_curve_{meta_name}"] = self._select_curve_scalar(self._curve_scalars(df, prefix, field, count=3), adpt_idx).astype(np.float32)

        add_curve(
            "voltvar",
            "DERVoltVar[0]",
            point_count=4,
            x_field="V",
            y_field="Var",
            meta_fields={
                "deptref": "DeptRef",
                "pri": "Pri",
                "vref": "VRef",
                "vref_auto": "VRefAuto",
                "vref_auto_ena": "VRefAutoEna",
                "vref_auto_tms": "VRefAutoTms",
                "rsp": "RspTms",
                "readonly": "ReadOnly",
            },
            measure_value=voltage_pct - 100.0 + df["DERVoltVar[0].Crv[0].VRef"].fillna(100.0).to_numpy(float),
            observed_value=var_pct,
        )
        add_curve(
            "voltwatt",
            "DERVoltWatt[0]",
            point_count=2,
            x_field="V",
            y_field="W",
            meta_fields={
                "deptref": "DeptRef",
                "rsp": "RspTms",
                "readonly": "ReadOnly",
            },
            measure_value=voltage_pct,
            observed_value=w_pct,
        )
        add_curve(
            "wattvar",
            "DERWattVar[0]",
            point_count=6,
            x_field="W",
            y_field="Var",
            meta_fields={
                "deptref": "DeptRef",
                "pri": "Pri",
                "readonly": "ReadOnly",
            },
            measure_value=w_pct,
            observed_value=var_pct,
        )

        ctl_idx = self._curve_index(df["DERFreqDroop[0].AdptCtlRslt"].to_numpy(float), 3)
        dbof_curves = self._curve_scalars(df, "DERFreqDroop[0]", "DbOf", count=3, item_label="Ctl")
        dbuf_curves = self._curve_scalars(df, "DERFreqDroop[0]", "DbUf", count=3, item_label="Ctl")
        kof_curves = self._curve_scalars(df, "DERFreqDroop[0]", "KOf", count=3, item_label="Ctl")
        kuf_curves = self._curve_scalars(df, "DERFreqDroop[0]", "KUf", count=3, item_label="Ctl")
        rsp_curves = self._curve_scalars(df, "DERFreqDroop[0]", "RspTms", count=3, item_label="Ctl")
        pmin_curves = self._curve_scalars(df, "DERFreqDroop[0]", "PMin", count=3, item_label="Ctl")
        readonly_curves = self._curve_scalars(df, "DERFreqDroop[0]", "ReadOnly", count=3, item_label="Ctl")
        dbof = self._select_curve_scalar(dbof_curves, ctl_idx)
        dbuf = self._select_curve_scalar(dbuf_curves, ctl_idx)
        kof = self._select_curve_scalar(kof_curves, ctl_idx)
        kuf = self._select_curve_scalar(kuf_curves, ctl_idx)
        rsp = self._select_curve_scalar(rsp_curves, ctl_idx)
        pmin = self._select_curve_scalar(pmin_curves, ctl_idx)
        readonly = self._select_curve_scalar(readonly_curves, ctl_idx)

        over_activation = np.maximum(hz - (60.0 + dbof), 0.0)
        under_activation = np.maximum((60.0 - dbuf) - hz, 0.0)
        expected_delta_pct = 100.0 * self._safe_div(over_activation, kof) - 100.0 * self._safe_div(under_activation, kuf)
        dbof_stack = np.column_stack(dbof_curves)
        dbuf_stack = np.column_stack(dbuf_curves)
        k_stack = np.column_stack(kof_curves + kuf_curves)
        pmin_stack = np.column_stack(pmin_curves)

        data["freqdroop_ctl_idx"] = ctl_idx.astype(np.int8)
        data["freqdroop_dbof"] = dbof.astype(np.float32)
        data["freqdroop_dbuf"] = dbuf.astype(np.float32)
        data["freqdroop_kof"] = kof.astype(np.float32)
        data["freqdroop_kuf"] = kuf.astype(np.float32)
        data["freqdroop_rsp"] = rsp.astype(np.float32)
        data["freqdroop_pmin"] = pmin.astype(np.float32)
        data["freqdroop_readonly"] = readonly.astype(np.float32)
        data["freqdroop_deadband_width"] = (dbof + dbuf).astype(np.float32)
        data["freqdroop_over_activation"] = over_activation.astype(np.float32)
        data["freqdroop_under_activation"] = under_activation.astype(np.float32)
        data["freqdroop_expected_delta_pct"] = expected_delta_pct.astype(np.float32)
        data["freqdroop_outside_deadband"] = ((over_activation > 0) | (under_activation > 0)).astype(np.int8)
        data["freqdroop_w_over_pmin_pct"] = (w_pct - pmin).astype(np.float32)
        data["freqdroop_db_span"] = (self._nanmax_rows(np.column_stack([dbof_stack, dbuf_stack])) - self._nanmin_rows(np.column_stack([dbof_stack, dbuf_stack]))).astype(np.float32)
        data["freqdroop_k_span"] = (self._nanmax_rows(k_stack) - self._nanmin_rows(k_stack)).astype(np.float32)
        data["freqdroop_pmin_span"] = (self._nanmax_rows(pmin_stack) - self._nanmin_rows(pmin_stack)).astype(np.float32)

    def _add_dc_features(
        self,
        data: Dict[str, np.ndarray],
        df: pd.DataFrame,
        *,
        w: np.ndarray,
        abs_w: np.ndarray,
    ) -> np.ndarray:
        dcw = df["DERMeasureDC[0].DCW"].to_numpy(float)
        dca = df["DERMeasureDC[0].DCA"].to_numpy(float)
        prt0 = df["DERMeasureDC[0].Prt[0].DCW"].to_numpy(float)
        prt1 = df["DERMeasureDC[0].Prt[1].DCW"].to_numpy(float)
        prt0_v = df["DERMeasureDC[0].Prt[0].DCV"].to_numpy(float)
        prt1_v = df["DERMeasureDC[0].Prt[1].DCV"].to_numpy(float)
        prt0_a = df["DERMeasureDC[0].Prt[0].DCA"].to_numpy(float)
        prt1_a = df["DERMeasureDC[0].Prt[1].DCA"].to_numpy(float)
        prt0_t = df["DERMeasureDC[0].Prt[0].PrtTyp"].to_numpy(float)
        prt1_t = df["DERMeasureDC[0].Prt[1].PrtTyp"].to_numpy(float)

        data["dcw_over_w"] = self._safe_div(dcw, w)
        data["dcw_over_abs_w"] = self._safe_div(dcw, abs_w)
        data["dcw_minus_port_sum"] = (dcw - (prt0 + prt1)).astype(np.float32)
        data["dcv_spread"] = np.abs(prt0_v - prt1_v).astype(np.float32)
        data["dca_spread"] = np.abs(prt0_a - prt1_a).astype(np.float32)
        data["dc_port0_share"] = self._safe_div(prt0, prt0 + prt1)
        data["dc_port_type_mismatch"] = (np.isfinite(prt0_t) & np.isfinite(prt1_t) & (prt0_t != prt1_t)).astype(np.int8)
        rare_type = (prt0_t == 7) | (prt1_t == 7)
        data["dc_port_type_rare_any"] = rare_type.astype(np.int8)
        data["ac_zero_dc_positive"] = ((np.abs(w) <= 1e-6) & (dcw > 0)).astype(np.int8)
        data["ac_positive_dc_zero"] = ((w > 0) & (np.abs(dcw) <= 1e-6)).astype(np.int8)
        data["ac_dc_same_sign"] = (np.sign(np.nan_to_num(w, nan=0.0)) == np.sign(np.nan_to_num(dcw, nan=0.0))).astype(np.int8)
        data["dca_over_total"] = self._safe_div(dca, prt0_a + prt1_a)
        return rare_type.astype(np.int8)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        self._coerce_numeric(df)

        fingerprint = df[COMMON_STR].fillna("<NA>").agg("|".join, axis=1)
        data: Dict[str, np.ndarray] = {
            "Id": df["Id"].to_numpy(),
            "device_fingerprint": fingerprint.to_numpy(dtype=object),
            "device_family": np.where(
                fingerprint == CANON1,
                "canon10",
                np.where(fingerprint == CANON2, "canon100", "other"),
            ),
            "common_missing_any": df[COMMON_STR].isna().any(axis=1).astype(np.int8).to_numpy(),
            "common_missing_count": df[COMMON_STR].isna().sum(axis=1).astype(np.int16).to_numpy(),
            "common_sn_has_decimal_suffix": df["common[0].SN"].fillna("").astype(str).str.endswith(".0").astype(np.int8).to_numpy(),
        }
        data["noncanonical"] = (data["device_family"] == "other").astype(np.int8)

        for col in RAW_NUMERIC:
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            data[SAFE_RAW[col]] = arr
        for col in RAW_STRING_COLUMNS:
            data[SAFE_STR[col]] = df[col].fillna("<NA>").astype(str).to_numpy(dtype=object)

        self._add_block_health_features(data, df)
        self._add_temperature_features(data, df)
        derived = self._add_power_capacity_features(data, df)
        gate_flags = self._add_gate_control_features(data, df, derived=derived)
        trip_any_power_when_outside = self._add_trip_features(
            data,
            df,
            voltage_pct=derived["voltage_pct"],
            hz=derived["hz"],
            abs_w=derived["abs_w"],
            tolw=derived["tolw"],
        )
        self._add_curve_control_features(
            data,
            df,
            voltage_pct=derived["voltage_pct"],
            w_pct=derived["w_pct"],
            var_pct=derived["var_pct"],
            hz=derived["hz"],
        )
        dc_port_type_rare = self._add_dc_features(data, df, w=derived["w"], abs_w=derived["abs_w"])

        ac_type_is_rare = np.isfinite(derived["ac_type"]) & (derived["ac_type"] == 3.0)
        data["ac_type_is_rare"] = ac_type_is_rare.astype(np.int8)

        flag_map = {
            "noncanonical": data["noncanonical"] == 1,
            "common_missing": data["common_missing_any"] == 1,
            "w_gt_wmax": data["w_gt_wmax_tol"] == 1,
            "w_gt_wmaxrtg": data["w_gt_wmaxrtg_tol"] == 1,
            "va_gt_vamax": data["va_gt_vamax_tol"] == 1,
            "var_gt_injmax": data["var_gt_injmax_tol"] == 1,
            "var_lt_absmax": data["var_lt_absmax_tol"] == 1,
            "wset_far": data["wset_enabled_far"] == 1,
            "wsetpct_far": data["wsetpct_enabled_far"] == 1,
            "wmaxlim_far": data["wmaxlim_enabled_far"] == 1,
            "varsetpct_far": data["varsetpct_enabled_far"] == 1,
            "model_structure": data["model_structure_anomaly_any"] == 1,
            "ac_type_rare": ac_type_is_rare == 1,
            "dc_type_rare": dc_port_type_rare == 1,
            "enter_state": gate_flags["enter_state"] == 1,
            "enter_blocked_power": gate_flags["enter_blocked_power"] == 1,
            "enter_blocked_current": gate_flags["enter_blocked_current"] == 1,
            "pf_abs": gate_flags["pf_abs"] == 1,
            "pf_abs_rvrt": gate_flags["pf_abs_rvrt"] == 1,
            "trip_power": trip_any_power_when_outside == 1,
        }
        hard_rule_flags = np.column_stack([flag_map[name] for name in HARD_RULE_NAMES])
        hard_override_flags = np.column_stack([flag_map[name] for name in self.hard_override_names])
        float_flags = {name: flag.astype(np.float32) for name, flag in flag_map.items()}
        data["hard_rule_count"] = hard_rule_flags.sum(axis=1).astype(np.int8)
        data["hard_rule_score"] = (
            3.0 * float_flags["noncanonical"]
            + 2.5 * float_flags["common_missing"]
            + 2.0
            * (
                float_flags["w_gt_wmax"]
                + float_flags["w_gt_wmaxrtg"]
                + float_flags["va_gt_vamax"]
                + float_flags["var_gt_injmax"]
                + float_flags["var_lt_absmax"]
                + float_flags["model_structure"]
                + float_flags["enter_state"]
                + float_flags["trip_power"]
            )
            + 1.5
            * (
                float_flags["wset_far"]
                + float_flags["wsetpct_far"]
                + float_flags["ac_type_rare"]
                + float_flags["dc_type_rare"]
                + float_flags["pf_abs"]
                + float_flags["pf_abs_rvrt"]
            )
            + 1.0 * float_flags["varsetpct_far"]
            + 0.75 * float_flags["wmaxlim_far"]
            + 0.35 * (float_flags["enter_blocked_power"] + float_flags["enter_blocked_current"])
        )
        hard_rule_anomaly = hard_rule_flags.any(axis=1).astype(np.int8)
        data["hard_rule_anomaly"] = hard_rule_anomaly
        data["hard_override_anomaly"] = hard_override_flags.any(axis=1).astype(np.int8)
        return pd.DataFrame(data)


DEFAULT_RUN_CONFIG = RunConfig()


def run_pipeline(config: RunConfig = DEFAULT_RUN_CONFIG) -> None:
    seed_everything(config.seed)
    LOGGER.info(
        "[run] starting pipeline with train=%s test=%s submission=%s",
        config.train_path,
        config.test_path,
        config.submission_path,
    )
    baseline = config.create_baseline()
    baseline.fit(config.train_path, config.train_row_limit)
    if config.write_test_predictions:
        baseline.predict_test(config.test_path, config.submission_path, config.test_row_limit)


def main() -> None:
    configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
