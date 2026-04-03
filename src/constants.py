import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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

__all__ = [name for name in globals() if name.isupper() or name in {"build_block_columns", "dedupe", "prefixed"}]
