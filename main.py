#!/usr/bin/env python3
import gc
import json
import hashlib
import math
import random
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier, XGBRegressor
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

def dedupe(columns):
    return list(dict.fromkeys(columns))

def prefixed(prefix, fields):
    return [f'{prefix}.{field}' for field in fields]

def build_volt_var_columns(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrv', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCrv']
    for curve in range(3):
        curve_prefix = f'{prefix}.Crv[{curve}]'
        cols.extend([f'{curve_prefix}.ActPt', f'{curve_prefix}.DeptRef', f'{curve_prefix}.Pri', f'{curve_prefix}.VRef', f'{curve_prefix}.VRefAuto', f'{curve_prefix}.VRefAutoEna', f'{curve_prefix}.VRefAutoTms', f'{curve_prefix}.RspTms', f'{curve_prefix}.ReadOnly'])
        for point in range(4):
            cols.extend([f'{curve_prefix}.Pt[{point}].V', f'{curve_prefix}.Pt[{point}].Var'])
    return cols

def build_volt_watt_columns(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrv', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCrv']
    for curve in range(3):
        curve_prefix = f'{prefix}.Crv[{curve}]'
        cols.extend([f'{curve_prefix}.ActPt', f'{curve_prefix}.DeptRef', f'{curve_prefix}.RspTms', f'{curve_prefix}.ReadOnly'])
        for point in range(2):
            cols.extend([f'{curve_prefix}.Pt[{point}].V', f'{curve_prefix}.Pt[{point}].W'])
    return cols

def build_watt_var_columns(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrv', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCrv']
    for curve in range(3):
        curve_prefix = f'{prefix}.Crv[{curve}]'
        cols.extend([f'{curve_prefix}.ActPt', f'{curve_prefix}.DeptRef', f'{curve_prefix}.Pri', f'{curve_prefix}.ReadOnly'])
        for point in range(6):
            cols.extend([f'{curve_prefix}.Pt[{point}].W', f'{curve_prefix}.Pt[{point}].Var'])
    return cols

def build_freq_droop_columns(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCtlReq', f'{prefix}.AdptCtlRslt', f'{prefix}.NCtl', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCtl']
    for ctl in range(3):
        ctl_prefix = f'{prefix}.Ctl[{ctl}]'
        cols.extend([f'{ctl_prefix}.DbOf', f'{ctl_prefix}.DbUf', f'{ctl_prefix}.KOf', f'{ctl_prefix}.KUf', f'{ctl_prefix}.RspTms', f'{ctl_prefix}.PMin', f'{ctl_prefix}.ReadOnly'])
    return cols

def build_trip_columns(prefix, axis_name):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrvSet']
    for curve in range(2):
        curve_prefix = f'{prefix}.Crv[{curve}]'
        cols.append(f'{curve_prefix}.ReadOnly')
        for group in ['MustTrip', 'MayTrip', 'MomCess']:
            group_prefix = f'{curve_prefix}.{group}'
            cols.append(f'{group_prefix}.ActPt')
            for point in range(5):
                cols.extend([f'{group_prefix}.Pt[{point}].{axis_name}', f'{group_prefix}.Pt[{point}].Tms'])
    return cols
COMMON_FIELDS = 'Mn Md Opt Vr SN'.split()
COMMON_STR = prefixed('common[0]', COMMON_FIELDS)
COMMON_COLUMNS = prefixed('common[0]', ['ID', 'L', *COMMON_FIELDS, 'DA'])
MEASURE_AC_FIELDS = '\nID L ACType W VA Var PF A LLV LNV Hz TmpAmb TmpCab TmpSnk TmpTrns TmpSw TmpOt\nThrotPct ThrotSrc WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3\nAL1 AL2 AL3 VL1L2 VL2L3 VL3L1 VL1 VL2 VL3\n'.split()
MEASURE_AC_COLUMNS = prefixed('DERMeasureAC[0]', MEASURE_AC_FIELDS)
CAPACITY_FIELDS = '\nID L WMaxRtg VAMaxRtg VarMaxInjRtg VarMaxAbsRtg WChaRteMaxRtg WDisChaRteMaxRtg\nVAChaRteMaxRtg VADisChaRteMaxRtg VNomRtg VMaxRtg VMinRtg AMaxRtg PFOvrExtRtg\nPFUndExtRtg NorOpCatRtg AbnOpCatRtg IntIslandCatRtg WMax WMaxOvrExt WOvrExtPF\nWMaxUndExt WUndExtPF VAMax VarMaxInj VarMaxAbs WChaRteMax WDisChaRteMax\nVAChaRteMax VADisChaRteMax VNom VMax VMin AMax PFOvrExt PFUndExt CtrlModes\nIntIslandCat\n'.split()
CAPACITY_COLUMNS = prefixed('DERCapacity[0]', CAPACITY_FIELDS)
ENTER_SERVICE_FIELDS = 'ID L ES ESVHi ESVLo ESHzHi ESHzLo ESDlyTms ESRndTms ESRmpTms ESDlyRemTms'.split()
ENTER_SERVICE_COLUMNS = prefixed('DEREnterService[0]', ENTER_SERVICE_FIELDS)
CTL_AC_FIELDS = '\nID L PFWInjEna PFWInjEnaRvrt PFWInjRvrtTms PFWInjRvrtRem PFWAbsEna PFWAbsEnaRvrt\nPFWAbsRvrtTms PFWAbsRvrtRem WMaxLimPctEna WMaxLimPct WMaxLimPctRvrt\nWMaxLimPctEnaRvrt WMaxLimPctRvrtTms WMaxLimPctRvrtRem WSetEna WSetMod WSet\nWSetRvrt WSetPct WSetPctRvrt WSetEnaRvrt WSetRvrtTms WSetRvrtRem VarSetEna\nVarSetMod VarSetPri VarSet VarSetRvrt VarSetPct VarSetPctRvrt VarSetEnaRvrt\nVarSetRvrtTms VarSetRvrtRem WRmp WRmpRef VarRmp AntiIslEna PFWInj.PF PFWInj.Ext\nPFWInjRvrt.PF PFWInjRvrt.Ext PFWAbs.Ext PFWAbsRvrt.Ext\n'.split()
CTL_AC_COLUMNS = prefixed('DERCtlAC[0]', CTL_AC_FIELDS)
VOLT_VAR_COLUMNS = build_volt_var_columns('DERVoltVar[0]')
VOLT_WATT_COLUMNS = build_volt_watt_columns('DERVoltWatt[0]')
FREQ_DROOP_COLUMNS = build_freq_droop_columns('DERFreqDroop[0]')
WATT_VAR_COLUMNS = build_watt_var_columns('DERWattVar[0]')
TRIP_SPECS: Dict[str, Tuple[str, str, str]] = {'lv': ('DERTripLV[0]', 'V', 'low'), 'hv': ('DERTripHV[0]', 'V', 'high'), 'lf': ('DERTripLF[0]', 'Hz', 'low'), 'hf': ('DERTripHF[0]', 'Hz', 'high')}
TRIP_COLUMNS = {short_name: build_trip_columns(prefix, axis_name) for short_name, (prefix, axis_name, _) in TRIP_SPECS.items()}
MEASURE_DC_FIELDS = '\nID L NPrt DCA DCW Prt[0].PrtTyp Prt[0].ID Prt[0].DCA Prt[0].DCV Prt[0].DCW\nPrt[0].Tmp Prt[1].PrtTyp Prt[1].ID Prt[1].DCA Prt[1].DCV Prt[1].DCW Prt[1].Tmp\n'.split()
MEASURE_DC_COLUMNS = prefixed('DERMeasureDC[0]', MEASURE_DC_FIELDS)
BLOCK_SOURCE_COLUMNS: Dict[str, List[str]] = {'common': COMMON_COLUMNS, 'measure_ac': MEASURE_AC_COLUMNS, 'capacity': CAPACITY_COLUMNS, 'enter_service': ENTER_SERVICE_COLUMNS, 'ctl_ac': CTL_AC_COLUMNS, 'volt_var': VOLT_VAR_COLUMNS, 'volt_watt': VOLT_WATT_COLUMNS, 'freq_droop': FREQ_DROOP_COLUMNS, 'watt_var': WATT_VAR_COLUMNS, 'measure_dc': MEASURE_DC_COLUMNS}
for short_name, cols in TRIP_COLUMNS.items():
    BLOCK_SOURCE_COLUMNS[f'trip_{short_name}'] = cols
CURVE_BLOCK_META_FIELDS = 'Ena AdptCrvReq AdptCrvRslt NPt NCrv RvrtTms RvrtRem RvrtCrv'.split()
FREQ_DROOP_META_FIELDS = 'Ena AdptCtlReq AdptCtlRslt NCtl RvrtTms RvrtRem RvrtCtl'.split()
TRIP_META_FIELDS = 'Ena AdptCrvReq AdptCrvRslt NPt NCrvSet'.split()
RAW_NUMERIC = dedupe(['common[0].DA', *prefixed('DERMeasureAC[0]', MEASURE_AC_FIELDS[2:]), *prefixed('DERCapacity[0]', CAPACITY_FIELDS[2:]), *prefixed('DEREnterService[0]', ENTER_SERVICE_FIELDS[2:]), *prefixed('DERCtlAC[0]', CTL_AC_FIELDS[2:]), *prefixed('DERVoltVar[0]', CURVE_BLOCK_META_FIELDS), *prefixed('DERVoltWatt[0]', CURVE_BLOCK_META_FIELDS), *prefixed('DERFreqDroop[0]', FREQ_DROOP_META_FIELDS), *prefixed('DERWattVar[0]', CURVE_BLOCK_META_FIELDS), *prefixed('DERMeasureDC[0]', MEASURE_DC_FIELDS[2:])])
TRIP_META_COLUMNS = [f'{prefix}.{field}' for prefix, _, _ in TRIP_SPECS.values() for field in TRIP_META_FIELDS]
RAW_EXTRA_NUMERIC_COLUMNS = ['DERMeasureAC[0].A_SF', 'DERMeasureAC[0].V_SF', 'DERMeasureAC[0].Hz_SF', 'DERMeasureAC[0].W_SF', 'DERMeasureAC[0].PF_SF', 'DERMeasureAC[0].VA_SF', 'DERMeasureAC[0].Var_SF', 'DERCapacity[0].WOvrExtRtg', 'DERCapacity[0].WOvrExtRtgPF', 'DERCapacity[0].WUndExtRtg', 'DERCapacity[0].WUndExtRtgPF', 'DERCapacity[0].W_SF', 'DERCapacity[0].PF_SF', 'DERCapacity[0].VA_SF', 'DERCapacity[0].Var_SF', 'DERCapacity[0].V_SF', 'DERCapacity[0].A_SF', 'DERCtlAC[0].WSet_SF', 'DERMeasureDC[0].DCA_SF', 'DERMeasureDC[0].DCW_SF']
RAW_EXTRA_STRING_COLUMNS = ['DERMeasureDC[0].Prt[0].IDStr', 'DERMeasureDC[0].Prt[1].IDStr']
RAW_NUMERIC = dedupe([*RAW_NUMERIC, *TRIP_META_COLUMNS, *RAW_EXTRA_NUMERIC_COLUMNS])
RAW_STRING_COLUMNS = dedupe([*COMMON_STR, *RAW_EXTRA_STRING_COLUMNS])
TRIP_SOURCE_COLUMNS = [col for cols in TRIP_COLUMNS.values() for col in cols]
ALL_SOURCE_COLUMNS = dedupe([*COMMON_COLUMNS, *MEASURE_AC_COLUMNS, *CAPACITY_COLUMNS, *ENTER_SERVICE_COLUMNS, *CTL_AC_COLUMNS, *VOLT_VAR_COLUMNS, *VOLT_WATT_COLUMNS, *FREQ_DROOP_COLUMNS, *WATT_VAR_COLUMNS, *MEASURE_DC_COLUMNS, *TRIP_SOURCE_COLUMNS, *RAW_EXTRA_NUMERIC_COLUMNS, *RAW_EXTRA_STRING_COLUMNS])
NUMERIC_SOURCE_COLUMNS = [c for c in ALL_SOURCE_COLUMNS if c not in RAW_STRING_COLUMNS]
USECOLS_TRAIN = dedupe(['Id', 'Label', *ALL_SOURCE_COLUMNS])
USECOLS_TEST = dedupe(['Id', *ALL_SOURCE_COLUMNS])
CANON1 = 'DERSec|DER Simulator|10 kW DER|1.2.3|SN-Three-Phase'
CANON2 = 'DERSec|DER Simulator 100 kW|1.2.3.1|1.0.0|1100058974'
SAFE_RAW = {c: re.sub('[^0-9A-Za-z_]+', '_', c) for c in RAW_NUMERIC}
SAFE_STR = {c: re.sub('[^0-9A-Za-z_]+', '_', c) for c in RAW_STRING_COLUMNS}
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ARCHIVE_NAME = 'cyber-physical-anomaly-detection-for-der-systems.zip'
DEFAULT_SEED = 42
MODEL_FILENAME = 'semantic_full_data_xgb.json'
REPORT_FILENAME = 'semantic_full_data_validation.json'

def default_zip_path():
    local_zip_path = SCRIPT_DIR / DATASET_ARCHIVE_NAME
    if local_zip_path.exists():
        return local_zip_path
    return Path('/mnt/data') / DATASET_ARCHIVE_NAME
DEFAULT_ZIP_PATH = default_zip_path()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'full_data_hybrid'
SQRT3 = math.sqrt(3.0)
DEVICE_FAMILY_MAP = {'canon10': 0, 'canon100': 1}
RESIDUAL_TAIL_LEVELS = {'tail': 0.95, 'extreme': 0.99, 'ultra': 0.999}
RESIDUAL_TAIL_FALLBACKS = {'tail': 0.05, 'extreme': 0.1, 'ultra': 0.2}
FAMILY_THRESHOLD_FLOOR = 0.02
MAX_THRESHOLD = 0.6
CANON100_NEGATIVE_WEIGHT = 1.5
HARD_OVERRIDE_TRAIN_WEIGHT = 0.35
SCENARIO_SMOOTHING = 50.0
AUDIT_TOLERANCE = 0.003
MIN_OVERRIDE_PRECISION = 0.995
CANON100_INTERACTION_FEATURES = ['hard_rule_score', 'scenario_rate', 'scenario_output_rate', 'resid_quantile_score', 'mode_dispatch_w_resid']
SURROGATE_TARGETS = {'w': ('DERMeasureAC_0_W', 'DERCapacity_0_WMaxRtg'), 'va': ('DERMeasureAC_0_VA', 'DERCapacity_0_VAMaxRtg'), 'var': ('DERMeasureAC_0_Var', 'DERCapacity_0_VarMaxInjRtg'), 'pf': ('DERMeasureAC_0_PF', None), 'a': ('DERMeasureAC_0_A', 'DERCapacity_0_AMaxRtg')}
SURROGATE_LEAKY_FEATURES = {*(f'DERMeasureAC_0_{field}' for field in '\n    W VA Var PF A WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3\n    AL1 AL2 AL3\n    '.split()), *'\n    w_over_wmaxrtg w_over_wmax va_over_vamax va_over_vamaxrtg var_over_injmax\n    var_over_absmax a_over_amax w_minus_wmax w_minus_wmaxrtg va_minus_vamax\n    var_minus_injmax var_plus_absmax w_eq_wmaxrtg w_eq_wmax var_eq_varmaxinj\n    var_eq_neg_varmaxabs pf_sign_mismatch w_gt_wmax_tol w_gt_wmaxrtg_tol\n    va_gt_vamax_tol var_gt_injmax_tol var_lt_absmax_tol va_minus_pqmag\n    va_over_pqmag pf_from_w_va pf_error w_phase_sum_error va_phase_sum_error\n    var_phase_sum_error phase_w_spread phase_var_spread wset_abs_error\n    wsetpct_target wsetpct_abs_error wmaxlim_target wmaxlim_excess\n    varset_abs_error varsetpct_target varsetpct_abs_error wset_enabled_far\n    wsetpct_enabled_far wmaxlim_enabled_far varsetpct_enabled_far w_pct_of_rtg\n    var_pct_of_limit enter_service_blocked_power enter_service_blocked_va\n    enter_service_blocked_current pf_inj_target_error pf_inj_reversion_error\n    pf_reactive_near_limit trip_lv_power_when_outside trip_hv_power_when_outside\n    trip_lf_power_when_outside trip_hf_power_when_outside\n    trip_any_power_when_outside voltvar_curve_error voltwatt_curve_error\n    wattvar_curve_expected wattvar_curve_error freqdroop_w_over_pmin_pct\n    dcw_over_w dcw_over_abs_w ac_zero_dc_positive ac_positive_dc_zero\n    ac_dc_same_sign\n    '.split()}
HARD_RULE_NAMES = ['noncanonical', 'common_missing', 'w_gt_wmax', 'w_gt_wmaxrtg', 'va_gt_vamax', 'var_gt_injmax', 'var_lt_absmax', 'wset_far', 'wsetpct_far', 'wmaxlim_far', 'varsetpct_far', 'model_structure', 'ac_type_rare', 'dc_type_rare', 'enter_state', 'enter_blocked_power', 'enter_blocked_current', 'pf_abs', 'pf_abs_rvrt', 'trip_power']
DEFAULT_HARD_OVERRIDE_NAMES = ['noncanonical', 'common_missing', 'w_gt_wmax', 'w_gt_wmaxrtg', 'va_gt_vamax', 'var_gt_injmax', 'var_lt_absmax', 'wset_far', 'wsetpct_far', 'model_structure', 'ac_type_rare', 'dc_type_rare', 'enter_state', 'pf_abs', 'pf_abs_rvrt', 'trip_power']
RULE_COLUMN_MAP = {'noncanonical': 'noncanonical', 'common_missing': 'common_missing_any', 'w_gt_wmax': 'w_gt_wmax_tol', 'w_gt_wmaxrtg': 'w_gt_wmaxrtg_tol', 'va_gt_vamax': 'va_gt_vamax_tol', 'var_gt_injmax': 'var_gt_injmax_tol', 'var_lt_absmax': 'var_lt_absmax_tol', 'wset_far': 'wset_enabled_far', 'wsetpct_far': 'wsetpct_enabled_far', 'wmaxlim_far': 'wmaxlim_enabled_far', 'varsetpct_far': 'varsetpct_enabled_far', 'model_structure': 'model_structure_anomaly_any', 'ac_type_rare': 'ac_type_is_rare', 'dc_type_rare': 'dc_port_type_rare_any', 'enter_state': 'enter_service_state_anomaly', 'enter_blocked_power': 'enter_service_blocked_power', 'enter_blocked_current': 'enter_service_blocked_current', 'pf_abs': 'pf_abs_ext_present', 'pf_abs_rvrt': 'pf_abs_rvrt_ext_present', 'trip_power': 'trip_any_power_when_outside'}
CAT_ENGINEERED_COLUMNS = ['device_fingerprint', 'common_missing_pattern', 'enter_service_missing_pattern', 'missing_selected_total', 'missing_selected_blocks', 'common_missing_any', 'common_missing_count', 'common_sn_has_decimal_suffix']
EXPECTED_MODEL_META = {'common': ('common[0].ID', 'common[0].L', 1.0, 66.0), 'measure_ac': ('DERMeasureAC[0].ID', 'DERMeasureAC[0].L', 701.0, 153.0), 'capacity': ('DERCapacity[0].ID', 'DERCapacity[0].L', 702.0, 50.0), 'enter_service': ('DEREnterService[0].ID', 'DEREnterService[0].L', 703.0, 17.0), 'measure_dc': ('DERMeasureDC[0].ID', 'DERMeasureDC[0].L', 714.0, 68.0)}

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
class TrainingReport:
    primary_metrics: Dict[str, MetricSummary]
    audit_metrics: Dict[str, MetricSummary]
    family_thresholds: Dict[str, float]
    family_blend_weights: Dict[str, float]
    semantic_feature_counts: Dict[str, int]
    cat_feature_counts: Dict[str, int]
    active_hard_override_names: List[str]
    demoted_hard_override_names: List[str]
    hard_override_rule_stats: Dict[str, Dict[str, float]]
    artifact_row_counts: Dict[str, int]
    artifact_dir: str

    def as_dict(self):
        return {'primary_metrics': {name: asdict(metric) for name, metric in self.primary_metrics.items()}, 'audit_metrics': {name: asdict(metric) for name, metric in self.audit_metrics.items()}, 'family_thresholds': self.family_thresholds, 'family_blend_weights': self.family_blend_weights, 'semantic_feature_counts': self.semantic_feature_counts, 'cat_feature_counts': self.cat_feature_counts, 'active_hard_override_names': self.active_hard_override_names, 'demoted_hard_override_names': self.demoted_hard_override_names, 'hard_override_rule_stats': self.hard_override_rule_stats, 'artifact_row_counts': self.artifact_row_counts, 'artifact_dir': self.artifact_dir}

@dataclass(frozen=True)
class RunConfig:
    zip_path: Path = DEFAULT_ZIP_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    artifact_dir: Path = DEFAULT_OUTPUT_DIR / 'artifacts'
    write_test_predictions: bool = True
    rebuild_artifacts: bool = False
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

    def create_baseline(self):
        return ResearchBaseline(artifact_dir=self.artifact_dir, rebuild_artifacts=self.rebuild_artifacts, chunksize=self.chunksize, cv_folds=self.cv_folds, n_estimators=self.xgb_n_estimators, max_depth=self.xgb_max_depth, learning_rate=self.xgb_learning_rate, subsample=self.xgb_subsample, colsample_bytree=self.xgb_colsample_bytree, cat_iterations=self.cat_iterations, cat_depth=self.cat_depth, cat_learning_rate=self.cat_learning_rate, n_jobs=self.n_jobs, seed=self.seed)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

class ResearchBaseline:

    def __init__(self, *, artifact_dir=DEFAULT_OUTPUT_DIR / 'artifacts', rebuild_artifacts=False, chunksize=5000, cv_folds=5, n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, cat_iterations=400, cat_depth=8, cat_learning_rate=0.05, n_jobs=4, seed=DEFAULT_SEED):
        self.artifact_dir = artifact_dir
        self.rebuild_artifacts = rebuild_artifacts
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
        self.hard_override_rule_stats: Dict[str, Dict[str, float]] = {}
        self.row_counts_by_family: Dict[str, int] = {}
        self.model: Optional[XGBClassifier] = None
        self.semantic_models: Dict[str, XGBClassifier] = {}
        self.cat_models: Dict[str, Any] = {}
        self.semantic_contexts: Dict[str, FamilySemanticContext] = {}
        self.family_thresholds: Dict[str, float] = {'canon10': 0.5, 'canon100': 0.5}
        self.family_blend_weights: Dict[str, float] = {'canon10': 1.0, 'canon100': 1.0}
        self.feature_cols: Optional[List[str]] = None
        self.semantic_feature_cols_by_family: Dict[str, List[str]] = {}
        self.cat_feature_cols_by_family: Dict[str, List[str]] = {}
        self.cat_categorical_cols_by_family: Dict[str, List[str]] = {}
        self.surrogate_feature_cols: Optional[List[str]] = None
        self.surrogate_models: Dict[Tuple[str, str], XGBRegressor] = {}
        self.residual_quantiles: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.family_base_rates: Dict[str, float] = {}
        self.scenario_sum_map: Dict[int, float] = {}
        self.scenario_count_map: Dict[int, int] = {}
        self.scenario_output_sum_map: Dict[int, float] = {}
        self.scenario_output_count_map: Dict[int, int] = {}
        self.training_report: Optional[TrainingReport] = None

    @staticmethod
    def _safe_div(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        out = np.full_like(a, np.nan)
        mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-06)
        out[mask] = a[mask] / b[mask]
        return out

    @staticmethod
    def _nanmin_rows(arr):
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
    def _nanmax_rows(arr):
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
    def _nanmean_rows(arr):
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
    def _curve_index(raw_idx, num_options):
        idx = np.nan_to_num(np.asarray(raw_idx, dtype=np.float32), nan=1.0)
        idx = idx.astype(np.int16) - 1
        idx[(idx < 0) | (idx >= num_options)] = 0
        return idx.astype(np.int8)

    @staticmethod
    def _select_curve_scalar(curves, idx):
        stacked = np.stack(curves, axis=1)
        return np.take_along_axis(stacked, idx[:, None], axis=1)[:, 0]

    @staticmethod
    def _select_curve_points(curves, idx):
        stacked = np.stack(curves, axis=1)
        return np.take_along_axis(stacked, idx[:, None, None], axis=1)[:, 0, :]

    @staticmethod
    def _pair_point_count(x_points, y_points):
        return (np.isfinite(np.asarray(x_points, dtype=np.float32)) & np.isfinite(np.asarray(y_points, dtype=np.float32))).sum(axis=1).astype(np.int16)

    @staticmethod
    def _curve_reverse_steps(x_points):
        x_points = np.asarray(x_points, dtype=np.float32)
        finite_pair = np.isfinite(x_points[:, :-1]) & np.isfinite(x_points[:, 1:])
        return ((np.diff(x_points, axis=1) < -1e-06) & finite_pair).sum(axis=1).astype(np.int8)

    @staticmethod
    def _curve_slope_stats(x_points, y_points):
        x_points = np.asarray(x_points, dtype=np.float32)
        y_points = np.asarray(y_points, dtype=np.float32)
        dx = np.diff(x_points, axis=1)
        dy = np.diff(y_points, axis=1)
        valid = np.isfinite(x_points[:, :-1]) & np.isfinite(x_points[:, 1:]) & np.isfinite(y_points[:, :-1]) & np.isfinite(y_points[:, 1:]) & (np.abs(dx) > 1e-06)
        slopes = np.full(dx.shape, np.nan, dtype=np.float32)
        slopes[valid] = dy[valid] / dx[valid]
        return (ResearchBaseline._nanmean_rows(slopes), ResearchBaseline._nanmax_rows(np.abs(slopes)))

    @staticmethod
    def _piecewise_interp(x, x_points, y_points):
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
            valid_seg = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1) & (np.abs(x1 - x0) > 1e-06)
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
    def _var_pct(var, varmaxinj, varmaxabs):
        var = np.asarray(var, dtype=np.float32)
        denom = np.where(var >= 0, np.asarray(varmaxinj, dtype=np.float32), np.asarray(varmaxabs, dtype=np.float32))
        return 100.0 * ResearchBaseline._safe_div(var, denom)

    def _coerce_numeric(self, df):
        for col in NUMERIC_SOURCE_COLUMNS:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    def _add_block_missingness(self, data, df):
        block_missing_total = np.zeros(len(df), dtype=np.int16)
        block_missing_any = np.zeros(len(df), dtype=np.int16)
        for block_name, cols in BLOCK_SOURCE_COLUMNS.items():
            missing = df[cols].isna()
            missing_count = missing.sum(axis=1).astype(np.int16).to_numpy()
            data[f'missing_{block_name}_count'] = missing_count
            data[f'missing_{block_name}_any'] = (missing_count > 0).astype(np.int8)
            block_missing_total += missing_count
            block_missing_any += (missing_count > 0).astype(np.int16)
        data['missing_selected_total'] = block_missing_total
        data['missing_selected_blocks'] = block_missing_any.astype(np.int8)
        common_missing = df[[*COMMON_STR, 'common[0].ID', 'common[0].L']].isna().to_numpy(dtype=np.uint16)
        common_weights = (1 << np.arange(common_missing.shape[1], dtype=np.uint16)).reshape(1, -1)
        data['common_missing_pattern'] = (common_missing * common_weights).sum(axis=1).astype(np.int16)
        enter_missing = df[ENTER_SERVICE_COLUMNS].isna().to_numpy(dtype=np.uint16)
        enter_weights = (1 << np.arange(enter_missing.shape[1], dtype=np.uint16)).reshape(1, -1)
        data['enter_service_missing_pattern'] = (enter_missing * enter_weights).sum(axis=1).astype(np.int16)

    def _add_model_integrity_features(self, data, df):
        anomaly_sum = np.zeros(len(df), dtype=np.int16)
        missing_sum = np.zeros(len(df), dtype=np.int16)
        for block_name, (id_col, len_col, expected_id, expected_len) in EXPECTED_MODEL_META.items():
            raw_id = df[id_col].to_numpy(float)
            raw_len = df[len_col].to_numpy(float)
            id_missing = ~np.isfinite(raw_id)
            len_missing = ~np.isfinite(raw_len)
            id_match = np.isclose(raw_id, expected_id, equal_nan=False)
            len_match = np.isclose(raw_len, expected_len, equal_nan=False)
            data[f'{block_name}_model_id_missing'] = id_missing.astype(np.int8)
            data[f'{block_name}_model_len_missing'] = len_missing.astype(np.int8)
            data[f'{block_name}_model_id_match'] = id_match.astype(np.int8)
            data[f'{block_name}_model_len_match'] = len_match.astype(np.int8)
            data[f'{block_name}_model_integrity_ok'] = (id_match & len_match).astype(np.int8)
            mismatch = ~id_missing & ~id_match | ~len_missing & ~len_match
            data[f'{block_name}_model_structure_anomaly'] = mismatch.astype(np.int8)
            anomaly_sum += mismatch.astype(np.int16)
            missing_sum += (id_missing | len_missing).astype(np.int16)
        data['model_structure_anomaly_count'] = anomaly_sum.astype(np.int8)
        data['model_structure_missing_count'] = missing_sum.astype(np.int8)
        data['model_structure_anomaly_any'] = (anomaly_sum > 0).astype(np.int8)

    def _add_capacity_extension_features(self, data, *, wmaxrtg, wmax, vamaxrtg, vamax, varmaxinjrtg, varmaxinj, varmaxabsrtg, varmaxabs, vnomrtg, vnom, vmaxrtg, vmax, vminrtg, vmin, amaxrtg, amax, wcha_rtg, wdis_rtg, vacha_rtg, vadis_rtg, wcha, wdis, vacha, vadis, pfover_rtg, pfover, pfunder_rtg, pfunder):
        data['vnom_setting_delta'] = (vnom - vnomrtg).astype(np.float32)
        data['vmax_setting_delta'] = (vmax - vmaxrtg).astype(np.float32)
        data['vmin_setting_delta'] = (vmin - vminrtg).astype(np.float32)
        data['amax_setting_delta'] = (amax - amaxrtg).astype(np.float32)
        data['pfover_setting_delta'] = (pfover - pfover_rtg).astype(np.float32)
        data['pfunder_setting_delta'] = (pfunder - pfunder_rtg).astype(np.float32)
        data['charge_rate_share_rtg'] = self._safe_div(wcha_rtg, wmaxrtg)
        data['discharge_rate_share_rtg'] = self._safe_div(wdis_rtg, wmaxrtg)
        data['charge_va_share_rtg'] = self._safe_div(vacha_rtg, vamaxrtg)
        data['discharge_va_share_rtg'] = self._safe_div(vadis_rtg, vamaxrtg)
        data['charge_rate_share_setting'] = self._safe_div(wcha, wmax)
        data['discharge_rate_share_setting'] = self._safe_div(wdis, wmax)
        data['charge_va_share_setting'] = self._safe_div(vacha, vamax)
        data['discharge_va_share_setting'] = self._safe_div(vadis, vamax)
        rating_pairs = [(wmaxrtg, wmax), (vamaxrtg, vamax), (varmaxinjrtg, varmaxinj), (varmaxabsrtg, varmaxabs), (vnomrtg, vnom), (vmaxrtg, vmax), (vminrtg, vmin), (amaxrtg, amax)]
        gap_count = np.zeros(len(wmaxrtg), dtype=np.int16)
        for rating, setting in rating_pairs:
            tol = np.maximum(1.0, 0.01 * np.nan_to_num(np.abs(rating), nan=0.0)).astype(np.float32)
            gap = np.isfinite(rating) & np.isfinite(setting) & (np.abs(setting - rating) > tol)
            gap_count += gap.astype(np.int16)
        data['rating_setting_gap_count'] = gap_count.astype(np.int8)

    def _add_temperature_features(self, data, df):
        temp_cols = ['DERMeasureAC[0].TmpAmb', 'DERMeasureAC[0].TmpCab', 'DERMeasureAC[0].TmpSnk', 'DERMeasureAC[0].TmpTrns', 'DERMeasureAC[0].TmpSw', 'DERMeasureAC[0].TmpOt']
        temps = df[temp_cols].to_numpy(float)
        temp_min = self._nanmin_rows(temps)
        temp_max = self._nanmax_rows(temps)
        temp_mean = self._nanmean_rows(temps)
        amb = df['DERMeasureAC[0].TmpAmb'].to_numpy(float)
        data['temp_min'] = temp_min
        data['temp_max'] = temp_max
        data['temp_mean'] = temp_mean
        data['temp_spread'] = (temp_max - temp_min).astype(np.float32)
        data['temp_max_over_ambient'] = (temp_max - amb).astype(np.float32)

    def _add_enter_service_features(self, data, df, *, voltage_pct, hz, abs_w, va, a, tolw, tolva, amax):
        es = df['DEREnterService[0].ES'].to_numpy(float)
        es_v_hi = df['DEREnterService[0].ESVHi'].to_numpy(float)
        es_v_lo = df['DEREnterService[0].ESVLo'].to_numpy(float)
        es_hz_hi = df['DEREnterService[0].ESHzHi'].to_numpy(float)
        es_hz_lo = df['DEREnterService[0].ESHzLo'].to_numpy(float)
        es_delay = df['DEREnterService[0].ESDlyTms'].to_numpy(float)
        es_random = df['DEREnterService[0].ESRndTms'].to_numpy(float)
        es_ramp = df['DEREnterService[0].ESRmpTms'].to_numpy(float)
        es_delay_rem = df['DEREnterService[0].ESDlyRemTms'].to_numpy(float)
        inside_v = np.isfinite(voltage_pct) & np.isfinite(es_v_hi) & np.isfinite(es_v_lo) & (voltage_pct >= es_v_lo) & (voltage_pct <= es_v_hi)
        inside_hz = np.isfinite(hz) & np.isfinite(es_hz_hi) & np.isfinite(es_hz_lo) & (hz >= es_hz_lo) & (hz <= es_hz_hi)
        inside_window = inside_v & inside_hz
        enabled = np.isfinite(es) & (es == 1.0)
        state_anomaly = np.isfinite(es) & (es >= 1.5)
        should_idle = ~enabled | ~inside_window
        current_tol = np.maximum(1.0, 0.02 * np.nan_to_num(amax, nan=0.0))
        data['enter_service_enabled'] = enabled.astype(np.int8)
        data['enter_service_state_anomaly'] = state_anomaly.astype(np.int8)
        data['enter_service_inside_window'] = inside_window.astype(np.int8)
        data['enter_service_outside_window'] = (~inside_window).astype(np.int8)
        data['enter_service_should_idle'] = should_idle.astype(np.int8)
        data['enter_service_v_window_width'] = (es_v_hi - es_v_lo).astype(np.float32)
        data['enter_service_hz_window_width'] = (es_hz_hi - es_hz_lo).astype(np.float32)
        data['enter_service_v_margin_low'] = (voltage_pct - es_v_lo).astype(np.float32)
        data['enter_service_v_margin_high'] = (es_v_hi - voltage_pct).astype(np.float32)
        data['enter_service_hz_margin_low'] = (hz - es_hz_lo).astype(np.float32)
        data['enter_service_hz_margin_high'] = (es_hz_hi - hz).astype(np.float32)
        data['enter_service_total_delay'] = (es_delay + es_random).astype(np.float32)
        data['enter_service_delay_remaining'] = es_delay_rem.astype(np.float32)
        data['enter_service_ramp_time'] = es_ramp.astype(np.float32)
        data['enter_service_delay_active'] = (np.nan_to_num(es_delay_rem, nan=0.0) > 0).astype(np.int8)
        blocked_power = should_idle & (abs_w > tolw)
        blocked_va = should_idle & (va > tolva)
        blocked_current = should_idle & (a > current_tol)
        data['enter_service_blocked_power'] = blocked_power.astype(np.int8)
        data['enter_service_blocked_va'] = blocked_va.astype(np.int8)
        data['enter_service_blocked_current'] = blocked_current.astype(np.int8)
        return (state_anomaly.astype(np.int8), blocked_power.astype(np.int8), blocked_current.astype(np.int8))

    def _add_pf_control_features(self, data, df, *, pf, var, varmaxinj, varmaxabs):
        pfinj_ena = np.nan_to_num(df['DERCtlAC[0].PFWInjEna'].to_numpy(float), nan=0.0)
        pfinj_ena_rvrt = np.nan_to_num(df['DERCtlAC[0].PFWInjEnaRvrt'].to_numpy(float), nan=0.0)
        pfabs_ena = np.nan_to_num(df['DERCtlAC[0].PFWAbsEna'].to_numpy(float), nan=0.0)
        pfabs_ena_rvrt = np.nan_to_num(df['DERCtlAC[0].PFWAbsEnaRvrt'].to_numpy(float), nan=0.0)
        pfinj_target = df['DERCtlAC[0].PFWInj.PF'].to_numpy(float)
        pfinj_rvrt_target = df['DERCtlAC[0].PFWInjRvrt.PF'].to_numpy(float)
        pfinj_ext = df['DERCtlAC[0].PFWInj.Ext'].to_numpy(float)
        pfinj_rvrt_ext = df['DERCtlAC[0].PFWInjRvrt.Ext'].to_numpy(float)
        pfabs_ext = df['DERCtlAC[0].PFWAbs.Ext'].to_numpy(float)
        pfabs_rvrt_ext = df['DERCtlAC[0].PFWAbsRvrt.Ext'].to_numpy(float)
        observed_var_pct = self._var_pct(var, varmaxinj, varmaxabs)
        inj_target_error = np.where((pfinj_ena > 0) & np.isfinite(pfinj_target), np.abs(np.abs(pf) - pfinj_target), np.nan)
        inj_rvrt_error = np.where((pfinj_ena_rvrt > 0) & np.isfinite(pfinj_rvrt_target), np.abs(np.abs(pf) - pfinj_rvrt_target), np.nan)
        data['pf_control_any_enabled'] = ((pfinj_ena > 0) | (pfabs_ena > 0)).astype(np.int8)
        data['pf_control_any_reversion'] = ((pfinj_ena_rvrt > 0) | (pfabs_ena_rvrt > 0)).astype(np.int8)
        data['pf_inj_target_error'] = inj_target_error.astype(np.float32)
        data['pf_inj_reversion_error'] = inj_rvrt_error.astype(np.float32)
        data['pf_inj_ext_present'] = np.isfinite(pfinj_ext).astype(np.int8)
        data['pf_inj_rvrt_ext_present'] = np.isfinite(pfinj_rvrt_ext).astype(np.int8)
        data['pf_abs_ext_present'] = np.isfinite(pfabs_ext).astype(np.int8)
        data['pf_abs_rvrt_ext_present'] = np.isfinite(pfabs_rvrt_ext).astype(np.int8)
        data['pf_inj_enabled_missing_target'] = ((pfinj_ena > 0) & ~np.isfinite(pfinj_target)).astype(np.int8)
        data['pf_reactive_near_limit'] = (np.abs(observed_var_pct) >= 95.0).astype(np.int8)
        return (np.isfinite(pfabs_ext).astype(np.int8), np.isfinite(pfabs_rvrt_ext).astype(np.int8))

    def _add_trip_block_features(self, data, df, *, short_name, prefix, axis_name, mode, measure_value, abs_w, tolw):
        adpt_idx = self._curve_index(df[f'{prefix}.AdptCrvRslt'].to_numpy(float), 2)
        group_scalar = lambda group, field: self._select_curve_scalar([df[f'{prefix}.Crv[{curve}].{group}.{field}'].to_numpy(float) for curve in range(2)], adpt_idx)
        group_points = lambda group, field: self._select_curve_points([np.column_stack([df[f'{prefix}.Crv[{curve}].{group}.Pt[{i}].{field}'].to_numpy(float) for i in range(5)]) for curve in range(2)], adpt_idx)
        must_actpt = group_scalar('MustTrip', 'ActPt')
        mom_actpt = group_scalar('MomCess', 'ActPt')
        must_x = group_points('MustTrip', axis_name)
        must_t = group_points('MustTrip', 'Tms')
        mom_x = group_points('MomCess', axis_name)
        mom_t = group_points('MomCess', 'Tms')
        may_present = np.column_stack([df[f'{prefix}.Crv[{curve}].MayTrip.Pt[{point}].{axis_name}'].to_numpy(float) for curve in range(2) for point in range(5)])
        enabled = np.nan_to_num(df[f'{prefix}.Ena'].to_numpy(float), nan=0.0) > 0
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
        if mode == 'low':
            margin = measure_value - must_x_max
        else:
            margin = must_x_min - measure_value
        outside = enabled & np.isfinite(margin) & (margin < 0)
        power_when_outside = outside & (abs_w > tolw)
        envelope_gap = np.where(np.isfinite(mom_x_min) & np.isfinite(must_x_max), np.abs(mom_x_min - must_x_max), np.nan)
        data[f'trip_{short_name}_curve_idx'] = adpt_idx.astype(np.int8)
        data[f'trip_{short_name}_enabled'] = enabled.astype(np.int8)
        data[f'trip_{short_name}_curve_req_gap'] = (df[f'{prefix}.AdptCrvReq'].to_numpy(float) - df[f'{prefix}.AdptCrvRslt'].to_numpy(float)).astype(np.float32)
        data[f'trip_{short_name}_musttrip_count'] = must_count
        data[f'trip_{short_name}_musttrip_actpt_gap'] = (must_actpt - must_count).astype(np.float32)
        data[f'trip_{short_name}_musttrip_axis_min'] = must_x_min
        data[f'trip_{short_name}_musttrip_axis_max'] = must_x_max
        data[f'trip_{short_name}_musttrip_axis_span'] = (must_x_max - must_x_min).astype(np.float32)
        data[f'trip_{short_name}_musttrip_tms_span'] = (must_t_max - must_t_min).astype(np.float32)
        data[f'trip_{short_name}_musttrip_reverse_steps'] = self._curve_reverse_steps(must_x)
        data[f'trip_{short_name}_momcess_count'] = mom_count
        data[f'trip_{short_name}_momcess_actpt_gap'] = (mom_actpt - mom_count).astype(np.float32)
        data[f'trip_{short_name}_momcess_axis_span'] = (mom_x_max - mom_x_min).astype(np.float32)
        data[f'trip_{short_name}_momcess_tms_span'] = (mom_t_max - mom_t_min).astype(np.float32)
        data[f'trip_{short_name}_momcess_reverse_steps'] = self._curve_reverse_steps(mom_x)
        data[f'trip_{short_name}_maytrip_present_any'] = np.isfinite(may_present).any(axis=1).astype(np.int8)
        data[f'trip_{short_name}_musttrip_margin'] = margin.astype(np.float32)
        data[f'trip_{short_name}_outside_musttrip'] = outside.astype(np.int8)
        data[f'trip_{short_name}_power_when_outside'] = power_when_outside.astype(np.int8)
        data[f'trip_{short_name}_momcess_musttrip_gap'] = envelope_gap.astype(np.float32)
        return (outside.astype(np.int8), power_when_outside.astype(np.int8))

    def _add_curve_block_features(self, data, *, name, raw_idx, curve_x, curve_y, curve_actpt, curve_meta, measure_value, observed_value=None):
        adpt_idx = self._curve_index(raw_idx, len(curve_x))
        selected_x = self._select_curve_points(curve_x, adpt_idx)
        selected_y = self._select_curve_points(curve_y, adpt_idx)
        selected_actpt = self._select_curve_scalar(curve_actpt, adpt_idx)
        data[f'{name}_curve_idx'] = adpt_idx.astype(np.int8)
        point_count = self._pair_point_count(selected_x, selected_y)
        data[f'{name}_curve_point_count'] = point_count
        data[f'{name}_curve_actpt_gap'] = (selected_actpt - point_count).astype(np.float32)
        x_min = self._nanmin_rows(selected_x)
        x_max = self._nanmax_rows(selected_x)
        y_min = self._nanmin_rows(selected_y)
        y_max = self._nanmax_rows(selected_y)
        mean_slope, max_abs_slope = self._curve_slope_stats(selected_x, selected_y)
        data[f'{name}_curve_x_span'] = (x_max - x_min).astype(np.float32)
        data[f'{name}_curve_y_span'] = (y_max - y_min).astype(np.float32)
        data[f'{name}_curve_reverse_steps'] = self._curve_reverse_steps(selected_x)
        data[f'{name}_curve_mean_slope'] = mean_slope
        data[f'{name}_curve_max_abs_slope'] = max_abs_slope
        data[f'{name}_curve_measure_margin_low'] = (measure_value - x_min).astype(np.float32)
        data[f'{name}_curve_measure_margin_high'] = (x_max - measure_value).astype(np.float32)
        if observed_value is not None:
            expected_value = self._piecewise_interp(measure_value, selected_x, selected_y)
            data[f'{name}_curve_expected'] = expected_value.astype(np.float32)
            data[f'{name}_curve_error'] = (observed_value - expected_value).astype(np.float32)
        for meta_name, curves in curve_meta.items():
            data[f'{name}_curve_{meta_name}'] = self._select_curve_scalar(curves, adpt_idx).astype(np.float32)

    def _add_freq_droop_features(self, data, df, *, hz, w_pct):
        raw_idx = df['DERFreqDroop[0].AdptCtlRslt'].to_numpy(float)
        ctl_idx = self._curve_index(raw_idx, 3)
        dbof_curves = [df[f'DERFreqDroop[0].Ctl[{i}].DbOf'].to_numpy(float) for i in range(3)]
        dbuf_curves = [df[f'DERFreqDroop[0].Ctl[{i}].DbUf'].to_numpy(float) for i in range(3)]
        kof_curves = [df[f'DERFreqDroop[0].Ctl[{i}].KOf'].to_numpy(float) for i in range(3)]
        kuf_curves = [df[f'DERFreqDroop[0].Ctl[{i}].KUf'].to_numpy(float) for i in range(3)]
        rsp_curves = [df[f'DERFreqDroop[0].Ctl[{i}].RspTms'].to_numpy(float) for i in range(3)]
        pmin_curves = [df[f'DERFreqDroop[0].Ctl[{i}].PMin'].to_numpy(float) for i in range(3)]
        ro_curves = [df[f'DERFreqDroop[0].Ctl[{i}].ReadOnly'].to_numpy(float) for i in range(3)]
        dbof = self._select_curve_scalar(dbof_curves, ctl_idx)
        dbuf = self._select_curve_scalar(dbuf_curves, ctl_idx)
        kof = self._select_curve_scalar(kof_curves, ctl_idx)
        kuf = self._select_curve_scalar(kuf_curves, ctl_idx)
        rsp = self._select_curve_scalar(rsp_curves, ctl_idx)
        pmin = self._select_curve_scalar(pmin_curves, ctl_idx)
        readonly = self._select_curve_scalar(ro_curves, ctl_idx)
        over_activation = np.maximum(hz - (60.0 + dbof), 0.0)
        under_activation = np.maximum(60.0 - dbuf - hz, 0.0)
        expected_delta_pct = 100.0 * self._safe_div(over_activation, kof) - 100.0 * self._safe_div(under_activation, kuf)
        dbof_stack = np.column_stack(dbof_curves)
        dbuf_stack = np.column_stack(dbuf_curves)
        k_stack = np.column_stack(kof_curves + kuf_curves)
        pmin_stack = np.column_stack(pmin_curves)
        data['freqdroop_ctl_idx'] = ctl_idx.astype(np.int8)
        data['freqdroop_dbof'] = dbof.astype(np.float32)
        data['freqdroop_dbuf'] = dbuf.astype(np.float32)
        data['freqdroop_kof'] = kof.astype(np.float32)
        data['freqdroop_kuf'] = kuf.astype(np.float32)
        data['freqdroop_rsp'] = rsp.astype(np.float32)
        data['freqdroop_pmin'] = pmin.astype(np.float32)
        data['freqdroop_readonly'] = readonly.astype(np.float32)
        data['freqdroop_deadband_width'] = (dbof + dbuf).astype(np.float32)
        data['freqdroop_over_activation'] = over_activation.astype(np.float32)
        data['freqdroop_under_activation'] = under_activation.astype(np.float32)
        data['freqdroop_expected_delta_pct'] = expected_delta_pct.astype(np.float32)
        data['freqdroop_outside_deadband'] = ((over_activation > 0) | (under_activation > 0)).astype(np.int8)
        data['freqdroop_w_over_pmin_pct'] = (w_pct - pmin).astype(np.float32)
        data['freqdroop_db_span'] = (self._nanmax_rows(np.column_stack([dbof_stack, dbuf_stack])) - self._nanmin_rows(np.column_stack([dbof_stack, dbuf_stack]))).astype(np.float32)
        data['freqdroop_k_span'] = (self._nanmax_rows(k_stack) - self._nanmin_rows(k_stack)).astype(np.float32)
        data['freqdroop_pmin_span'] = (self._nanmax_rows(pmin_stack) - self._nanmin_rows(pmin_stack)).astype(np.float32)

    def _add_dc_features(self, data, df, *, w, abs_w):
        dcw = df['DERMeasureDC[0].DCW'].to_numpy(float)
        dca = df['DERMeasureDC[0].DCA'].to_numpy(float)
        prt0 = df['DERMeasureDC[0].Prt[0].DCW'].to_numpy(float)
        prt1 = df['DERMeasureDC[0].Prt[1].DCW'].to_numpy(float)
        prt0_v = df['DERMeasureDC[0].Prt[0].DCV'].to_numpy(float)
        prt1_v = df['DERMeasureDC[0].Prt[1].DCV'].to_numpy(float)
        prt0_a = df['DERMeasureDC[0].Prt[0].DCA'].to_numpy(float)
        prt1_a = df['DERMeasureDC[0].Prt[1].DCA'].to_numpy(float)
        prt0_t = df['DERMeasureDC[0].Prt[0].PrtTyp'].to_numpy(float)
        prt1_t = df['DERMeasureDC[0].Prt[1].PrtTyp'].to_numpy(float)
        data['dcw_over_w'] = self._safe_div(dcw, w)
        data['dcw_over_abs_w'] = self._safe_div(dcw, abs_w)
        data['dcw_minus_port_sum'] = (dcw - (prt0 + prt1)).astype(np.float32)
        data['dcv_spread'] = np.abs(prt0_v - prt1_v).astype(np.float32)
        data['dca_spread'] = np.abs(prt0_a - prt1_a).astype(np.float32)
        data['dc_port0_share'] = self._safe_div(prt0, prt0 + prt1)
        data['dc_port_type_mismatch'] = (np.isfinite(prt0_t) & np.isfinite(prt1_t) & (prt0_t != prt1_t)).astype(np.int8)
        rare_type = (prt0_t == 7) | (prt1_t == 7)
        data['dc_port_type_rare_any'] = rare_type.astype(np.int8)
        data['ac_zero_dc_positive'] = ((np.abs(w) <= 1e-06) & (dcw > 0)).astype(np.int8)
        data['ac_positive_dc_zero'] = ((w > 0) & (np.abs(dcw) <= 1e-06)).astype(np.int8)
        data['ac_dc_same_sign'] = (np.sign(np.nan_to_num(w, nan=0.0)) == np.sign(np.nan_to_num(dcw, nan=0.0))).astype(np.int8)
        data['dca_over_total'] = self._safe_div(dca, prt0_a + prt1_a)
        return rare_type.astype(np.int8)

    def build_features(self, df):
        self._coerce_numeric(df)
        fingerprint = df[COMMON_STR].fillna('<NA>').agg('|'.join, axis=1)
        data: Dict[str, np.ndarray] = {'Id': df['Id'].to_numpy(), 'device_fingerprint': fingerprint.to_numpy(dtype=object), 'device_family': np.where(fingerprint == CANON1, 'canon10', np.where(fingerprint == CANON2, 'canon100', 'other')), 'common_missing_any': df[COMMON_STR].isna().any(axis=1).astype(np.int8).to_numpy(), 'common_missing_count': df[COMMON_STR].isna().sum(axis=1).astype(np.int16).to_numpy(), 'common_sn_has_decimal_suffix': df['common[0].SN'].fillna('').astype(str).str.endswith('.0').astype(np.int8).to_numpy()}
        data['noncanonical'] = (data['device_family'] == 'other').astype(np.int8)
        for col in RAW_NUMERIC:
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            data[SAFE_RAW[col]] = arr
        for col in RAW_STRING_COLUMNS:
            data[SAFE_STR[col]] = df[col].fillna('<NA>').astype(str).to_numpy(dtype=object)
        self._add_block_missingness(data, df)
        self._add_model_integrity_features(data, df)
        self._add_temperature_features(data, df)
        w = df['DERMeasureAC[0].W'].to_numpy(float)
        abs_w = np.abs(w)
        va = df['DERMeasureAC[0].VA'].to_numpy(float)
        var = df['DERMeasureAC[0].Var'].to_numpy(float)
        pf = df['DERMeasureAC[0].PF'].to_numpy(float)
        a = df['DERMeasureAC[0].A'].to_numpy(float)
        llv = df['DERMeasureAC[0].LLV'].to_numpy(float)
        lnv = df['DERMeasureAC[0].LNV'].to_numpy(float)
        hz = df['DERMeasureAC[0].Hz'].to_numpy(float)
        wmaxrtg = df['DERCapacity[0].WMaxRtg'].to_numpy(float)
        vamaxrtg = df['DERCapacity[0].VAMaxRtg'].to_numpy(float)
        varmaxinjrtg = df['DERCapacity[0].VarMaxInjRtg'].to_numpy(float)
        varmaxabsrtg = df['DERCapacity[0].VarMaxAbsRtg'].to_numpy(float)
        wmax = df['DERCapacity[0].WMax'].to_numpy(float)
        vamax = df['DERCapacity[0].VAMax'].to_numpy(float)
        varmaxinj = df['DERCapacity[0].VarMaxInj'].to_numpy(float)
        varmaxabs = df['DERCapacity[0].VarMaxAbs'].to_numpy(float)
        amax = df['DERCapacity[0].AMax'].to_numpy(float)
        vnom = df['DERCapacity[0].VNom'].to_numpy(float)
        vmax = df['DERCapacity[0].VMax'].to_numpy(float)
        vmin = df['DERCapacity[0].VMin'].to_numpy(float)
        for name, numerator, denominator in [('w_over_wmaxrtg', w, wmaxrtg), ('w_over_wmax', w, wmax), ('va_over_vamax', va, vamax), ('va_over_vamaxrtg', va, vamaxrtg), ('var_over_injmax', var, varmaxinj), ('var_over_absmax', var, varmaxabs), ('a_over_amax', a, amax), ('llv_over_vnom', llv, vnom), ('lnv_over_vnom', lnv * SQRT3, vnom)]:
            data[name] = self._safe_div(numerator, denominator)
        for name, value in [('w_minus_wmax', w - wmax), ('w_minus_wmaxrtg', w - wmaxrtg), ('va_minus_vamax', va - vamax), ('var_minus_injmax', var - varmaxinj), ('var_plus_absmax', var + varmaxabs), ('llv_minus_lnv_sqrt3', llv - lnv * SQRT3), ('hz_delta_60', hz - 60.0)]:
            data[name] = value.astype(np.float32)
        for name, left, right in [('w_eq_wmaxrtg', w, wmaxrtg), ('w_eq_wmax', w, wmax), ('var_eq_varmaxinj', var, varmaxinj), ('var_eq_neg_varmaxabs', var, -varmaxabs)]:
            data[name] = np.isclose(left, right, equal_nan=False).astype(np.int8)
        data['pf_sign_mismatch'] = ((np.sign(np.nan_to_num(pf)) != np.sign(np.nan_to_num(w))) & (np.nan_to_num(pf) != 0) & (np.nan_to_num(w) != 0)).astype(np.int8)
        tolw = np.maximum(50.0, 0.02 * np.nan_to_num(wmaxrtg, nan=0.0)).astype(np.float32)
        tolva = np.maximum(50.0, 0.02 * np.nan_to_num(vamax, nan=0.0)).astype(np.float32)
        tolvi = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxinj, nan=0.0)).astype(np.float32)
        tolva2 = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxabs, nan=0.0)).astype(np.float32)
        for name, value, upper_bound in [('w_gt_wmax_tol', w, wmax + tolw), ('w_gt_wmaxrtg_tol', w, wmaxrtg + tolw), ('va_gt_vamax_tol', va, vamax + tolva), ('var_gt_injmax_tol', var, varmaxinj + tolvi)]:
            data[name] = (value > upper_bound).astype(np.int8)
        data['var_lt_absmax_tol'] = (var < -varmaxabs - tolva2).astype(np.int8)
        pq = np.sqrt(np.square(w.astype(np.float32)) + np.square(var.astype(np.float32)))
        data['va_minus_pqmag'] = (va - pq).astype(np.float32)
        data['va_over_pqmag'] = self._safe_div(va, pq)
        pf_from_w_va = self._safe_div(w, va)
        data['pf_from_w_va'] = pf_from_w_va
        data['pf_error'] = (pf - pf_from_w_va).astype(np.float32)
        for name, total, suffixes in [('w_phase_sum_error', w, ['WL1', 'WL2', 'WL3']), ('va_phase_sum_error', va, ['VAL1', 'VAL2', 'VAL3']), ('var_phase_sum_error', var, ['VarL1', 'VarL2', 'VarL3'])]:
            phase_sum = sum((df[f'DERMeasureAC[0].{suffix}'].to_numpy(float) for suffix in suffixes))
            data[name] = (total - phase_sum).astype(np.float32)
        for name, suffixes in [('phase_ll_spread', ['VL1L2', 'VL2L3', 'VL3L1']), ('phase_ln_spread', ['VL1', 'VL2', 'VL3']), ('phase_w_spread', ['WL1', 'WL2', 'WL3']), ('phase_var_spread', ['VarL1', 'VarL2', 'VarL3'])]:
            phase_values = df[[f'DERMeasureAC[0].{suffix}' for suffix in suffixes]].to_numpy(float)
            data[name] = (self._nanmax_rows(phase_values) - self._nanmin_rows(phase_values)).astype(np.float32)
        for name, numerator, denominator in [('wmax_over_wmaxrtg', wmax, wmaxrtg), ('vamax_over_vamaxrtg', vamax, vamaxrtg), ('vmax_over_vnom', vmax, vnom), ('vmin_over_vnom', vmin, vnom)]:
            data[name] = self._safe_div(numerator, denominator)
        wsetena = np.nan_to_num(df['DERCtlAC[0].WSetEna'].to_numpy(float), nan=0.0)
        wset = df['DERCtlAC[0].WSet'].to_numpy(float)
        wsetpct = df['DERCtlAC[0].WSetPct'].to_numpy(float)
        wmaxlimena = np.nan_to_num(df['DERCtlAC[0].WMaxLimPctEna'].to_numpy(float), nan=0.0)
        wmaxlimpct = df['DERCtlAC[0].WMaxLimPct'].to_numpy(float)
        varsetena = np.nan_to_num(df['DERCtlAC[0].VarSetEna'].to_numpy(float), nan=0.0)
        varset = df['DERCtlAC[0].VarSet'].to_numpy(float)
        varsetpct = df['DERCtlAC[0].VarSetPct'].to_numpy(float)
        wset_abs_error = np.where(wsetena > 0, np.abs(w - wset), np.nan)
        wsetpct_target = wmaxrtg * (wsetpct / 100.0)
        wsetpct_abs_error = np.where(wsetena > 0, np.abs(w - wsetpct_target), np.nan)
        wmaxlim_target = wmaxrtg * (wmaxlimpct / 100.0)
        wmaxlim_excess = np.where(wmaxlimena > 0, w - wmaxlim_target, np.nan)
        varset_abs_error = np.where(varsetena > 0, np.abs(var - varset), np.nan)
        varsetpct_target = varmaxinj * (varsetpct / 100.0)
        varsetpct_abs_error = np.where(varsetena > 0, np.abs(var - varsetpct_target), np.nan)
        data['wset_abs_error'] = wset_abs_error.astype(np.float32)
        data['wsetpct_target'] = wsetpct_target.astype(np.float32)
        data['wsetpct_abs_error'] = wsetpct_abs_error.astype(np.float32)
        data['wmaxlim_target'] = wmaxlim_target.astype(np.float32)
        data['wmaxlim_excess'] = wmaxlim_excess.astype(np.float32)
        data['varset_abs_error'] = varset_abs_error.astype(np.float32)
        data['varsetpct_target'] = varsetpct_target.astype(np.float32)
        data['varsetpct_abs_error'] = varsetpct_abs_error.astype(np.float32)
        data['wset_enabled_far'] = ((wsetena > 0) & (wset_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data['wsetpct_enabled_far'] = ((wsetena > 0) & (wsetpct_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data['wmaxlim_enabled_far'] = ((wmaxlimena > 0) & (wmaxlim_excess > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data['varsetpct_enabled_far'] = ((varsetena > 0) & (varsetpct_abs_error > np.maximum(20.0, 0.05 * np.nan_to_num(varmaxinj, nan=0.0)))).astype(np.int8)
        self._add_capacity_extension_features(data, wmaxrtg=wmaxrtg, wmax=wmax, vamaxrtg=vamaxrtg, vamax=vamax, varmaxinjrtg=varmaxinjrtg, varmaxinj=varmaxinj, varmaxabsrtg=varmaxabsrtg, varmaxabs=varmaxabs, vnomrtg=df['DERCapacity[0].VNomRtg'].to_numpy(float), vnom=vnom, vmaxrtg=df['DERCapacity[0].VMaxRtg'].to_numpy(float), vmax=vmax, vminrtg=df['DERCapacity[0].VMinRtg'].to_numpy(float), vmin=vmin, amaxrtg=df['DERCapacity[0].AMaxRtg'].to_numpy(float), amax=amax, wcha_rtg=df['DERCapacity[0].WChaRteMaxRtg'].to_numpy(float), wdis_rtg=df['DERCapacity[0].WDisChaRteMaxRtg'].to_numpy(float), vacha_rtg=df['DERCapacity[0].VAChaRteMaxRtg'].to_numpy(float), vadis_rtg=df['DERCapacity[0].VADisChaRteMaxRtg'].to_numpy(float), wcha=df['DERCapacity[0].WChaRteMax'].to_numpy(float), wdis=df['DERCapacity[0].WDisChaRteMax'].to_numpy(float), vacha=df['DERCapacity[0].VAChaRteMax'].to_numpy(float), vadis=df['DERCapacity[0].VADisChaRteMax'].to_numpy(float), pfover_rtg=df['DERCapacity[0].PFOvrExtRtg'].to_numpy(float), pfover=df['DERCapacity[0].PFOvrExt'].to_numpy(float), pfunder_rtg=df['DERCapacity[0].PFUndExtRtg'].to_numpy(float), pfunder=df['DERCapacity[0].PFUndExt'].to_numpy(float))
        voltage_pct = 100.0 * self._safe_div(llv, vnom)
        line_neutral_voltage_pct = 100.0 * self._safe_div(lnv * SQRT3, vnom)
        w_pct = 100.0 * self._safe_div(w, wmaxrtg)
        var_pct = self._var_pct(var, varmaxinj, varmaxabs)
        data['voltage_pct'] = voltage_pct.astype(np.float32)
        data['line_neutral_voltage_pct'] = line_neutral_voltage_pct.astype(np.float32)
        data['w_pct_of_rtg'] = w_pct.astype(np.float32)
        data['var_pct_of_limit'] = var_pct.astype(np.float32)
        enter_state_anomaly, enter_blocked_power, enter_blocked_current = self._add_enter_service_features(data, df, voltage_pct=voltage_pct, hz=hz, abs_w=abs_w, va=va, a=a, tolw=tolw, tolva=tolva, amax=amax)
        pf_abs_ext_present, pf_abs_rvrt_ext_present = self._add_pf_control_features(data, df, pf=pf, var=var, varmaxinj=varmaxinj, varmaxabs=varmaxabs)
        trip_outside_flags = []
        trip_power_flags = []
        for short_name, (prefix, axis_name, mode) in TRIP_SPECS.items():
            measure_value = voltage_pct if axis_name == 'V' else hz
            outside, power_when_outside = self._add_trip_block_features(data, df, short_name=short_name, prefix=prefix, axis_name=axis_name, mode=mode, measure_value=measure_value, abs_w=abs_w, tolw=tolw)
            trip_outside_flags.append(outside)
            trip_power_flags.append(power_when_outside)
        if trip_outside_flags:
            trip_any_outside = np.column_stack(trip_outside_flags).any(axis=1).astype(np.int8)
            trip_any_power_when_outside = np.column_stack(trip_power_flags).any(axis=1).astype(np.int8)
        else:
            trip_any_outside = np.zeros(len(df), dtype=np.int8)
            trip_any_power_when_outside = np.zeros(len(df), dtype=np.int8)
        data['trip_any_outside_musttrip'] = trip_any_outside
        data['trip_any_power_when_outside'] = trip_any_power_when_outside
        curve_scalars = lambda prefix, field: [df[f'{prefix}.Crv[{curve}].{field}'].to_numpy(float) for curve in range(3)]
        curve_points = lambda prefix, field, point_count: [np.column_stack([df[f'{prefix}.Crv[{curve}].Pt[{point}].{field}'].to_numpy(float) for point in range(point_count)]) for curve in range(3)]
        curve_specs = [('voltvar', 'DERVoltVar[0]', 4, 'V', 'Var', {'deptref': 'DeptRef', 'pri': 'Pri', 'vref': 'VRef', 'vref_auto': 'VRefAuto', 'vref_auto_ena': 'VRefAutoEna', 'vref_auto_tms': 'VRefAutoTms', 'rsp': 'RspTms', 'readonly': 'ReadOnly'}, voltage_pct - 100.0 + df['DERVoltVar[0].Crv[0].VRef'].fillna(100.0).to_numpy(float), var_pct), ('voltwatt', 'DERVoltWatt[0]', 2, 'V', 'W', {'deptref': 'DeptRef', 'rsp': 'RspTms', 'readonly': 'ReadOnly'}, voltage_pct, w_pct), ('wattvar', 'DERWattVar[0]', 6, 'W', 'Var', {'deptref': 'DeptRef', 'pri': 'Pri', 'readonly': 'ReadOnly'}, w_pct, var_pct)]
        for name, prefix, point_count, x_field, y_field, meta_fields, measure_value, observed_value in curve_specs:
            self._add_curve_block_features(data, name=name, raw_idx=df[f'{prefix}.AdptCrvRslt'].to_numpy(float), curve_x=curve_points(prefix, x_field, point_count), curve_y=curve_points(prefix, y_field, point_count), curve_actpt=curve_scalars(prefix, 'ActPt'), curve_meta={meta_name: curve_scalars(prefix, field) for meta_name, field in meta_fields.items()}, measure_value=measure_value, observed_value=observed_value)
        self._add_freq_droop_features(data, df, hz=hz, w_pct=w_pct)
        dc_port_type_rare = self._add_dc_features(data, df, w=w, abs_w=abs_w)
        ac_type = df['DERMeasureAC[0].ACType'].to_numpy(float)
        ac_type_is_rare = np.isfinite(ac_type) & (ac_type == 3.0)
        data['ac_type_is_rare'] = ac_type_is_rare.astype(np.int8)
        flag_map = {'noncanonical': data['noncanonical'] == 1, 'common_missing': data['common_missing_any'] == 1, 'w_gt_wmax': data['w_gt_wmax_tol'] == 1, 'w_gt_wmaxrtg': data['w_gt_wmaxrtg_tol'] == 1, 'va_gt_vamax': data['va_gt_vamax_tol'] == 1, 'var_gt_injmax': data['var_gt_injmax_tol'] == 1, 'var_lt_absmax': data['var_lt_absmax_tol'] == 1, 'wset_far': data['wset_enabled_far'] == 1, 'wsetpct_far': data['wsetpct_enabled_far'] == 1, 'wmaxlim_far': data['wmaxlim_enabled_far'] == 1, 'varsetpct_far': data['varsetpct_enabled_far'] == 1, 'model_structure': data['model_structure_anomaly_any'] == 1, 'ac_type_rare': ac_type_is_rare == 1, 'dc_type_rare': dc_port_type_rare == 1, 'enter_state': enter_state_anomaly == 1, 'enter_blocked_power': enter_blocked_power == 1, 'enter_blocked_current': enter_blocked_current == 1, 'pf_abs': pf_abs_ext_present == 1, 'pf_abs_rvrt': pf_abs_rvrt_ext_present == 1, 'trip_power': trip_any_power_when_outside == 1}
        hard_rule_flags = np.column_stack([flag_map[name] for name in HARD_RULE_NAMES])
        hard_override_flags = np.column_stack([flag_map[name] for name in self.hard_override_names])
        float_flags = {name: flag.astype(np.float32) for name, flag in flag_map.items()}
        data['hard_rule_count'] = hard_rule_flags.sum(axis=1).astype(np.int8)
        data['hard_rule_score'] = 3.0 * float_flags['noncanonical'] + 2.5 * float_flags['common_missing'] + 2.0 * (float_flags['w_gt_wmax'] + float_flags['w_gt_wmaxrtg'] + float_flags['va_gt_vamax'] + float_flags['var_gt_injmax'] + float_flags['var_lt_absmax'] + float_flags['model_structure'] + float_flags['enter_state'] + float_flags['trip_power']) + 1.5 * (float_flags['wset_far'] + float_flags['wsetpct_far'] + float_flags['ac_type_rare'] + float_flags['dc_type_rare'] + float_flags['pf_abs'] + float_flags['pf_abs_rvrt']) + 1.0 * float_flags['varsetpct_far'] + 0.75 * float_flags['wmaxlim_far'] + 0.35 * (float_flags['enter_blocked_power'] + float_flags['enter_blocked_current'])
        hard_rule_anomaly = hard_rule_flags.any(axis=1).astype(np.int8)
        data['hard_rule_anomaly'] = hard_rule_anomaly
        data['hard_override_anomaly'] = hard_override_flags.any(axis=1).astype(np.int8)
        return pd.DataFrame(data)

    def iter_raw_chunks(self, zip_path, member, usecols, limit_rows=0):
        yielded = 0
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as fh:
                for chunk in pd.read_csv(fh, usecols=list(usecols), chunksize=self.chunksize, low_memory=False):
                    if limit_rows > 0:
                        remaining = limit_rows - yielded
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining].copy()
                    yielded += len(chunk)
                    yield chunk
                    if limit_rows > 0 and yielded >= limit_rows:
                        break

    @staticmethod
    def tune_threshold(y_true, prob):
        best_thr, best_f2 = (0.5, -1.0)
        for thr in np.linspace(0.02, 0.98, 97):
            pred = (prob >= thr).astype(int)
            score = fbeta_score(y_true, pred, beta=2)
            if score > best_f2:
                best_thr, best_f2 = (float(thr), float(score))
        return (best_thr, best_f2)

    def _encode_device_family(self, df):
        out = df.copy()
        out['device_family'] = out['device_family'].map(DEVICE_FAMILY_MAP).fillna(-1).astype(np.int8)
        return out

    def _get_surrogate_feature_cols(self, columns):
        excluded = {'Id', 'Label', 'fold_id', 'audit_fold_id', 'device_fingerprint', 'hard_rule_anomaly', 'hard_rule_count', 'hard_rule_score', 'hard_override_anomaly'}
        excluded.update(SAFE_STR.values())
        return [col for col in columns if col not in excluded and col not in SURROGATE_LEAKY_FEATURES]

    def _build_sample_weights(self, x_df, y):
        weights = np.ones(len(x_df), dtype=np.float32)
        family = x_df['device_family'].to_numpy()
        hard_override = x_df['hard_override_anomaly'].to_numpy() == 1
        weights[(family == 'canon100') & (y == 0)] *= CANON100_NEGATIVE_WEIGHT
        weights[hard_override] *= HARD_OVERRIDE_TRAIN_WEIGHT
        return weights

    @staticmethod
    def _bucketize(values, *, fill_value, dtype, scale=1.0, round_values=True):
        out = pd.to_numeric(values, errors='coerce')
        if scale != 1.0:
            out = out / scale
        if round_values:
            out = out.round()
        return out.fillna(fill_value).astype(dtype)

    def _build_scenario_frame(self, x_df, *, include_output_bins):
        frame: Dict[str, pd.Series] = {'family': x_df['device_family'].astype(str), 'throt_src': self._bucketize(x_df['DERMeasureAC_0_ThrotSrc'], fill_value=-1, dtype=np.int16), 'throt_pct': self._bucketize(x_df['DERMeasureAC_0_ThrotPct'], scale=5.0, fill_value=-1, dtype=np.int16), 'wmaxlim_pct': self._bucketize(x_df['DERCtlAC_0_WMaxLimPct'], scale=5.0, fill_value=-1, dtype=np.int16), 'wset_pct': self._bucketize(x_df['DERCtlAC_0_WSetPct'], scale=5.0, fill_value=-1, dtype=np.int16), 'varset_pct': self._bucketize(x_df['DERCtlAC_0_VarSetPct'], scale=5.0, fill_value=-1, dtype=np.int16), 'pf_set': self._bucketize(x_df['DERCtlAC_0_PFWInj_PF'], scale=0.02, fill_value=-1, dtype=np.int16), 'fd_idx': self._bucketize(x_df['DERFreqDroop_0_AdptCtlRslt'], fill_value=-1, dtype=np.int16), 'vv_idx': self._bucketize(x_df['DERVoltVar_0_AdptCrvRslt'], fill_value=-1, dtype=np.int16), 'vw_idx': self._bucketize(x_df['DERVoltWatt_0_AdptCrvRslt'], fill_value=-1, dtype=np.int16), 'wv_idx': self._bucketize(x_df['DERWattVar_0_AdptCrvRslt'], fill_value=-1, dtype=np.int16), 'volt_bin': self._bucketize(x_df['voltage_pct'], fill_value=-999, dtype=np.int16), 'hz_bin': self._bucketize(x_df['DERMeasureAC_0_Hz'], scale=0.1, fill_value=-999, dtype=np.int16), 'enter_idle': self._bucketize(x_df['enter_service_should_idle'], fill_value=0, dtype=np.int8, round_values=False), 'droop_active': self._bucketize(x_df['freqdroop_outside_deadband'], fill_value=0, dtype=np.int8, round_values=False)}
        if include_output_bins:
            frame['w_bin'] = self._bucketize(x_df['w_pct_of_rtg'], scale=5.0, fill_value=-999, dtype=np.int16)
            frame['var_bin'] = self._bucketize(x_df['var_pct_of_limit'], scale=5.0, fill_value=-999, dtype=np.int16)
            frame['pf_mode'] = self._bucketize(x_df['pf_control_any_enabled'], fill_value=0, dtype=np.int8, round_values=False)
        return pd.DataFrame(frame)

    @staticmethod
    def _hash_frame(frame):
        return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)

    def _build_scenario_keys(self, x_df):
        return self._hash_frame(self._build_scenario_frame(x_df, include_output_bins=False))

    def _build_scenario_output_keys(self, x_df):
        return self._hash_frame(self._build_scenario_frame(x_df, include_output_bins=True))

    @staticmethod
    def _lookup_scenario_stats(keys, sum_map, count_map):
        key_series = pd.Series(keys)
        sum_values = key_series.map(sum_map).fillna(0.0).to_numpy(np.float32)
        count_values = key_series.map(count_map).fillna(0).to_numpy(np.int32)
        return (sum_values, count_values)

    @staticmethod
    def _assign_scenario_features(out, *, family_prior, scenario_rate, scenario_count, scenario_output_rate, scenario_output_count):
        out['scenario_rate'] = scenario_rate.astype(np.float32)
        out['scenario_rate_delta'] = (scenario_rate - family_prior).astype(np.float32)
        out['scenario_count'] = scenario_count.astype(np.int32)
        out['scenario_log_count'] = np.log1p(scenario_count).astype(np.float32)
        out['scenario_low_support'] = (scenario_count < 20).astype(np.int8)
        out['scenario_output_rate'] = scenario_output_rate.astype(np.float32)
        out['scenario_output_rate_delta'] = (scenario_output_rate - family_prior).astype(np.float32)
        out['scenario_output_count'] = scenario_output_count.astype(np.int32)
        out['scenario_output_log_count'] = np.log1p(scenario_output_count).astype(np.float32)
        out['scenario_output_low_support'] = (scenario_output_count < 20).astype(np.int8)
        return out

    def _fit_transform_scenario_features(self, x_train, y_train):
        out = x_train.copy()
        y_arr = y_train.to_numpy(np.float32)
        family_series = out['device_family'].astype(str)
        self.family_base_rates = pd.DataFrame({'family': family_series, 'y': y_arr}).groupby('family')['y'].mean().to_dict()
        keys = self._build_scenario_keys(out)
        output_keys = self._build_scenario_output_keys(out)
        fold_ids = (out['Id'].to_numpy(np.int64) % self.cv_folds).astype(np.int8)
        scenario_rate = np.zeros(len(out), dtype=np.float32)
        scenario_count = np.zeros(len(out), dtype=np.int32)
        scenario_output_rate = np.zeros(len(out), dtype=np.float32)
        scenario_output_count = np.zeros(len(out), dtype=np.int32)
        global_rate = float(np.mean(y_arr))
        for fold in range(self.cv_folds):
            train_mask = fold_ids != fold
            valid_mask = fold_ids == fold
            if not valid_mask.any():
                continue
            stats = pd.DataFrame({'key': keys[train_mask], 'y': y_arr[train_mask]}).groupby('key')['y'].agg(['sum', 'count'])
            output_stats = pd.DataFrame({'key': output_keys[train_mask], 'y': y_arr[train_mask]}).groupby('key')['y'].agg(['sum', 'count'])
            valid_keys = pd.Series(keys[valid_mask])
            valid_sum = valid_keys.map(stats['sum']).fillna(0.0).to_numpy(np.float32)
            valid_count = valid_keys.map(stats['count']).fillna(0).to_numpy(np.int32)
            valid_output_keys = pd.Series(output_keys[valid_mask])
            valid_output_sum = valid_output_keys.map(output_stats['sum']).fillna(0.0).to_numpy(np.float32)
            valid_output_count = valid_output_keys.map(output_stats['count']).fillna(0).to_numpy(np.int32)
            valid_family = family_series.loc[valid_mask].tolist()
            prior = np.array([self.family_base_rates.get(name, global_rate) for name in valid_family], dtype=np.float32)
            scenario_rate[valid_mask] = (valid_sum + SCENARIO_SMOOTHING * prior) / (valid_count + SCENARIO_SMOOTHING)
            scenario_count[valid_mask] = valid_count
            scenario_output_rate[valid_mask] = (valid_output_sum + SCENARIO_SMOOTHING * prior) / (valid_output_count + SCENARIO_SMOOTHING)
            scenario_output_count[valid_mask] = valid_output_count
        full_stats = pd.DataFrame({'key': keys, 'y': y_arr}).groupby('key')['y'].agg(['sum', 'count'])
        full_output_stats = pd.DataFrame({'key': output_keys, 'y': y_arr}).groupby('key')['y'].agg(['sum', 'count'])
        self.scenario_sum_map = {int(idx): float(val) for idx, val in full_stats['sum'].items()}
        self.scenario_count_map = {int(idx): int(val) for idx, val in full_stats['count'].items()}
        self.scenario_output_sum_map = {int(idx): float(val) for idx, val in full_output_stats['sum'].items()}
        self.scenario_output_count_map = {int(idx): int(val) for idx, val in full_output_stats['count'].items()}
        family_prior = family_series.map(self.family_base_rates).fillna(global_rate).to_numpy(np.float32)
        return self._assign_scenario_features(out, family_prior=family_prior, scenario_rate=scenario_rate, scenario_count=scenario_count, scenario_output_rate=scenario_output_rate, scenario_output_count=scenario_output_count)

    def _apply_scenario_features(self, x_df):
        if not self.scenario_count_map:
            return x_df
        out = x_df.copy()
        keys = self._build_scenario_keys(out)
        output_keys = self._build_scenario_output_keys(out)
        sum_values, count_values = self._lookup_scenario_stats(keys, self.scenario_sum_map, self.scenario_count_map)
        output_sum_values, output_count_values = self._lookup_scenario_stats(output_keys, self.scenario_output_sum_map, self.scenario_output_count_map)
        global_rate = float(np.mean(list(self.family_base_rates.values()))) if self.family_base_rates else 0.5
        family_prior = out['device_family'].astype(str).map(self.family_base_rates).fillna(global_rate).to_numpy(np.float32)
        scenario_rate = (sum_values + SCENARIO_SMOOTHING * family_prior) / (count_values + SCENARIO_SMOOTHING)
        scenario_output_rate = (output_sum_values + SCENARIO_SMOOTHING * family_prior) / (output_count_values + SCENARIO_SMOOTHING)
        return self._assign_scenario_features(out, family_prior=family_prior, scenario_rate=scenario_rate, scenario_count=count_values, scenario_output_rate=scenario_output_rate, scenario_output_count=output_count_values)

    def _add_family_interaction_features(self, x_df):
        out = x_df.copy()
        canon100_mask = out['device_family'].astype(str) == 'canon100'
        for feature_name in CANON100_INTERACTION_FEATURES:
            if feature_name not in out.columns:
                continue
            values = pd.to_numeric(out[feature_name], errors='coerce').to_numpy(np.float32)
            out[f'canon100_{feature_name}'] = np.where(canon100_mask.to_numpy(), values, 0.0).astype(np.float32)
        return out

    def _surrogate_partition_mask(self, ids, *, fit_partition):
        ids_arr = np.asarray(ids, dtype=np.int64)
        fit_mask = ids_arr % 2 == 0
        return fit_mask if fit_partition else ~fit_mask

    def _xgb_shared_params(self, *, eval_metric, verbosity):
        return {'subsample': self.subsample, 'colsample_bytree': self.colsample_bytree, 'eval_metric': eval_metric, 'tree_method': 'hist', 'n_jobs': self.n_jobs, 'random_state': self.seed, 'seed': self.seed, 'verbosity': verbosity}

    def _new_surrogate_model(self):
        return XGBRegressor(n_estimators=max(80, self.n_estimators // 2), max_depth=max(4, self.max_depth - 2), learning_rate=min(0.08, self.learning_rate * 1.2), objective='reg:squarederror', **self._xgb_shared_params(eval_metric='rmse', verbosity=0))

    def _new_classifier(self):
        return XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, objective='binary:logistic', **self._xgb_shared_params(eval_metric='logloss', verbosity=1))

    def _fit_surrogate_models(self, x_train, y_train, valid_mask):
        self.surrogate_feature_cols = self._get_surrogate_feature_cols(x_train.columns)
        fit_partition = self._surrogate_partition_mask(x_train['Id'], fit_partition=True)
        normal_mask = (y_train == 0) & (x_train['hard_override_anomaly'] == 0) & (x_train['device_family'] != 'other') & ~valid_mask.to_numpy() & fit_partition
        surrogate_df = x_train.loc[normal_mask].copy()
        if surrogate_df.empty:
            raise RuntimeError('No rows available to train surrogate models.')
        self.surrogate_models = {}
        for family in DEVICE_FAMILY_MAP:
            family_df = surrogate_df.loc[surrogate_df['device_family'] == family].copy()
            if family_df.empty:
                continue
            x_surrogate = self._encode_device_family(family_df[self.surrogate_feature_cols])
            for target_name, (target_col, _) in SURROGATE_TARGETS.items():
                model = self._new_surrogate_model()
                y_target = family_df[target_col].to_numpy(np.float32)
                print(f'[surrogate] training {family}/{target_name} on {len(family_df):,} normal rows')
                model.fit(x_surrogate, y_target)
                self.surrogate_models[family, target_name] = model

    def _augment_with_surrogates(self, x_df):
        if self.surrogate_feature_cols is None or not self.surrogate_models:
            return x_df
        out = x_df.copy()
        for target_name in SURROGATE_TARGETS:
            out[f'pred_{target_name}'] = np.nan
            out[f'resid_{target_name}'] = np.nan
            out[f'abs_resid_{target_name}'] = np.nan
            out[f'norm_resid_{target_name}'] = np.nan
            out[f'abs_norm_resid_{target_name}'] = np.nan
            out[f'tail_resid_{target_name}'] = 0
            out[f'extreme_resid_{target_name}'] = 0
            out[f'ultra_resid_{target_name}'] = 0
            out[f'q99_ratio_resid_{target_name}'] = np.nan
        x_surrogate = self._encode_device_family(out[self.surrogate_feature_cols])
        for family in DEVICE_FAMILY_MAP:
            family_mask = out['device_family'] == family
            if not family_mask.any():
                continue
            x_family = x_surrogate.loc[family_mask]
            for target_name, (target_col, scale_col) in SURROGATE_TARGETS.items():
                model = self.surrogate_models.get((family, target_name))
                if model is None:
                    continue
                pred = model.predict(x_family).astype(np.float32)
                actual = out.loc[family_mask, target_col].to_numpy(np.float32)
                resid = actual - pred
                out.loc[family_mask, f'pred_{target_name}'] = pred
                out.loc[family_mask, f'resid_{target_name}'] = resid
                out.loc[family_mask, f'abs_resid_{target_name}'] = np.abs(resid).astype(np.float32)
                if scale_col is not None:
                    scale = out.loc[family_mask, scale_col].to_numpy(np.float32)
                    norm_resid = self._safe_div(resid, scale)
                else:
                    scale = np.maximum(0.05, np.abs(actual))
                    norm_resid = (resid / scale).astype(np.float32)
                out.loc[family_mask, f'norm_resid_{target_name}'] = norm_resid.astype(np.float32)
                out.loc[family_mask, f'abs_norm_resid_{target_name}'] = np.abs(norm_resid).astype(np.float32)
        out['resid_energy_total'] = out[['abs_resid_w', 'abs_resid_va', 'abs_resid_var', 'abs_resid_pf', 'abs_resid_a']].sum(axis=1).astype(np.float32)
        out['resid_va_minus_pq'] = (out['pred_va'] - np.sqrt(np.square(out['pred_w']) + np.square(out['pred_var']))).astype(np.float32)
        out['resid_w_var_ratio'] = self._safe_div(out['abs_resid_w'].to_numpy(float), out['abs_resid_var'].to_numpy(float) + 0.001)
        return out

    def _compute_residual_quantiles(self, x_train, y_train, valid_mask):
        calibration_partition = self._surrogate_partition_mask(x_train['Id'], fit_partition=False)
        base_mask = (y_train == 0) & (x_train['hard_override_anomaly'] == 0) & (x_train['device_family'] != 'other') & ~valid_mask.to_numpy()
        self.residual_quantiles = {}
        for family in DEVICE_FAMILY_MAP:
            family_mask = base_mask & (x_train['device_family'] == family)
            family_calibration = family_mask & calibration_partition
            if not family_calibration.any():
                family_calibration = family_mask
            family_quantiles: Dict[str, Dict[str, float]] = {}
            for target_name in SURROGATE_TARGETS:
                series = x_train.loc[family_calibration, f'abs_norm_resid_{target_name}']
                values = pd.to_numeric(series, errors='coerce').to_numpy(np.float32)
                values = values[np.isfinite(values)]
                quantiles = RESIDUAL_TAIL_FALLBACKS.copy()
                if values.size > 0:
                    for level_name, q in RESIDUAL_TAIL_LEVELS.items():
                        quantiles[level_name] = float(np.quantile(values, q))
                family_quantiles[target_name] = {key: max(1e-06, value) for key, value in quantiles.items()}
            self.residual_quantiles[family] = family_quantiles

    def _apply_residual_calibration_features(self, x_df):
        if not self.residual_quantiles:
            return x_df
        out = x_df.copy()
        for target_name in SURROGATE_TARGETS:
            out[f'tail_resid_{target_name}'] = 0
            out[f'extreme_resid_{target_name}'] = 0
            out[f'ultra_resid_{target_name}'] = 0
            out[f'q99_ratio_resid_{target_name}'] = np.nan
        for family in DEVICE_FAMILY_MAP:
            family_mask = out['device_family'] == family
            if not family_mask.any():
                continue
            family_quantiles = self.residual_quantiles.get(family, {})
            for target_name in SURROGATE_TARGETS:
                abs_norm = out.loc[family_mask, f'abs_norm_resid_{target_name}'].to_numpy(np.float32)
                q = family_quantiles.get(target_name, RESIDUAL_TAIL_FALLBACKS)
                tail = abs_norm >= q['tail']
                extreme = abs_norm >= q['extreme']
                ultra = abs_norm >= q['ultra']
                q99_ratio = self._safe_div(abs_norm, np.full_like(abs_norm, q['extreme'], dtype=np.float32))
                out.loc[family_mask, f'tail_resid_{target_name}'] = tail.astype(np.int8)
                out.loc[family_mask, f'extreme_resid_{target_name}'] = extreme.astype(np.int8)
                out.loc[family_mask, f'ultra_resid_{target_name}'] = ultra.astype(np.int8)
                out.loc[family_mask, f'q99_ratio_resid_{target_name}'] = q99_ratio.astype(np.float32)
        abs_norm_w = np.nan_to_num(out['abs_norm_resid_w'].to_numpy(np.float32), nan=0.0)
        abs_norm_var = np.nan_to_num(out['abs_norm_resid_var'].to_numpy(np.float32), nan=0.0)
        abs_norm_pf = np.nan_to_num(out['abs_norm_resid_pf'].to_numpy(np.float32), nan=0.0)
        abs_norm_a = np.nan_to_num(out['abs_norm_resid_a'].to_numpy(np.float32), nan=0.0)
        pf_mode = np.nan_to_num(out['pf_control_any_enabled'].to_numpy(np.float32), nan=0.0) > 0
        voltvar_mode = (np.nan_to_num(out['DERVoltVar_0_Ena'].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(out['voltvar_curve_expected'].to_numpy(np.float32))
        voltwatt_mode = (np.nan_to_num(out['DERVoltWatt_0_Ena'].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(out['voltwatt_curve_expected'].to_numpy(np.float32))
        wattvar_mode = (np.nan_to_num(out['DERWattVar_0_Ena'].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(out['wattvar_curve_expected'].to_numpy(np.float32))
        droop_mode = np.nan_to_num(out['freqdroop_outside_deadband'].to_numpy(np.float32), nan=0.0) > 0
        enter_idle_mode = np.nan_to_num(out['enter_service_should_idle'].to_numpy(np.float32), nan=0.0) > 0
        out['mode_resid_pf_pf'] = (abs_norm_pf * pf_mode).astype(np.float32)
        out['mode_resid_var_pf'] = (abs_norm_var * pf_mode).astype(np.float32)
        out['mode_resid_var_voltvar'] = (abs_norm_var * voltvar_mode).astype(np.float32)
        out['mode_resid_w_voltwatt'] = (abs_norm_w * voltwatt_mode).astype(np.float32)
        out['mode_resid_var_wattvar'] = (abs_norm_var * wattvar_mode).astype(np.float32)
        out['mode_resid_w_droop'] = (abs_norm_w * droop_mode).astype(np.float32)
        out['mode_resid_w_enter_idle'] = (abs_norm_w * enter_idle_mode).astype(np.float32)
        out['mode_resid_a_enter_idle'] = (abs_norm_a * enter_idle_mode).astype(np.float32)
        out['mode_curve_var_resid'] = (abs_norm_var * (voltvar_mode | wattvar_mode | pf_mode)).astype(np.float32)
        out['mode_dispatch_w_resid'] = (abs_norm_w * (voltwatt_mode | droop_mode | enter_idle_mode)).astype(np.float32)
        out['mode_extreme_var_curve'] = (np.nan_to_num(out['extreme_resid_var'].to_numpy(np.float32), nan=0.0) * (voltvar_mode | wattvar_mode | pf_mode)).astype(np.int8)
        out['mode_extreme_w_dispatch'] = (np.nan_to_num(out['extreme_resid_w'].to_numpy(np.float32), nan=0.0) * (voltwatt_mode | droop_mode | enter_idle_mode)).astype(np.int8)
        out['mode_tail_count'] = out[['mode_extreme_var_curve', 'mode_extreme_w_dispatch']].sum(axis=1).astype(np.int8)
        out['resid_tail_count'] = out[['tail_resid_w', 'tail_resid_va', 'tail_resid_var', 'tail_resid_pf', 'tail_resid_a']].sum(axis=1).astype(np.int8)
        out['resid_extreme_count'] = out[['extreme_resid_w', 'extreme_resid_va', 'extreme_resid_var', 'extreme_resid_pf', 'extreme_resid_a']].sum(axis=1).astype(np.int8)
        out['resid_ultra_count'] = out[['ultra_resid_w', 'ultra_resid_va', 'ultra_resid_var', 'ultra_resid_pf', 'ultra_resid_a']].sum(axis=1).astype(np.int8)
        out['resid_quantile_score'] = out[['q99_ratio_resid_w', 'q99_ratio_resid_va', 'q99_ratio_resid_var', 'q99_ratio_resid_pf', 'q99_ratio_resid_a']].sum(axis=1).astype(np.float32)
        return out

    @staticmethod
    def tune_threshold(y_true, prob, *, low=FAMILY_THRESHOLD_FLOOR, high=MAX_THRESHOLD, step=0.01):
        if len(y_true) == 0:
            return (0.5, 0.0)
        best_thr, best_f2 = (0.5, -1.0)
        thresholds = np.arange(low, high + 1e-09, step, dtype=np.float32)
        for thr in thresholds:
            pred = (prob >= thr).astype(np.int8)
            score = fbeta_score(y_true, pred, beta=2)
            if score > best_f2:
                best_thr, best_f2 = (float(thr), float(score))
        return (best_thr, best_f2)

    @staticmethod
    def _metric_summary_from_pred(y_true, pred):
        if len(y_true) == 0:
            return MetricSummary(f2=0.0, precision=0.0, recall=0.0, positive_rate=0.0, rows=0)
        return MetricSummary(f2=float(fbeta_score(y_true, pred, beta=2)), precision=float(precision_score(y_true, pred, zero_division=0)), recall=float(recall_score(y_true, pred, zero_division=0)), positive_rate=float(np.mean(pred)), rows=int(len(y_true)))

    @staticmethod
    def _blend_probs(primary, secondary, weight):
        if secondary is None:
            return primary.astype(np.float32)
        return (weight * primary + (1.0 - weight) * secondary).astype(np.float32)

    @staticmethod
    def _select_nonconstant_columns(df, candidates):
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
    def _report_rows_to_metric(rows, pred_col):
        return ResearchBaseline._metric_summary_from_pred(rows['Label'].to_numpy(np.int8), rows[pred_col].to_numpy(np.int8))

    def _ensure_catboost_available(self):
        if CatBoostClassifier is None:
            raise RuntimeError('CatBoost is required for the full-data hybrid pipeline. Install dependencies from pyproject.toml before training.')

    def _new_cat_model(self):
        self._ensure_catboost_available()
        return CatBoostClassifier(iterations=self.cat_iterations, depth=self.cat_depth, learning_rate=self.cat_learning_rate, loss_function='Logloss', eval_metric='Logloss', random_seed=self.seed, thread_count=self.n_jobs, allow_writing_files=False, verbose=False)

    def _artifact_metadata_path(self):
        return self.artifact_dir / 'metadata.json'

    def _build_train_artifacts(self, zip_path):
        metadata_path = self._artifact_metadata_path()
        if metadata_path.exists() and (not self.rebuild_artifacts):
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            self.row_counts_by_family = {k: int(v) for k, v in metadata.get('row_counts', {}).items()}
            return metadata
        if self.artifact_dir.exists() and self.rebuild_artifacts:
            shutil.rmtree(self.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        train_root = self.artifact_dir / 'train'
        row_counts = {'canon10': 0, 'canon100': 0, 'other': 0}
        part_counts = {'canon10': 0, 'canon100': 0, 'other': 0}
        for chunk_idx, chunk in enumerate(self.iter_raw_chunks(zip_path, 'train.csv', USECOLS_TRAIN, 0)):
            labels = chunk['Label'].astype(np.int8).to_numpy()
            feats = self.build_features(chunk.drop(columns=['Label']))
            feats['Label'] = labels
            feats['fold_id'] = (feats['Id'].to_numpy(np.int64) % self.cv_folds).astype(np.int8)
            scenario_keys = self._build_scenario_keys(feats)
            feats['audit_fold_id'] = (scenario_keys % self.cv_folds).astype(np.int8)
            for family in row_counts:
                family_mask = feats['device_family'] == family
                if not family_mask.any():
                    continue
                family_dir = train_root / family
                family_dir.mkdir(parents=True, exist_ok=True)
                family_df = feats.loc[family_mask].copy()
                out_path = family_dir / f'part_{part_counts[family]:05d}.parquet'
                family_df.to_parquet(out_path, index=False)
                part_counts[family] += 1
                row_counts[family] += int(len(family_df))
            if chunk_idx % 10 == 0:
                built = sum(row_counts.values())
                print(f'[artifacts] materialized {built:,} training rows')
            del feats
            gc.collect()
        metadata = {'row_counts': row_counts, 'part_counts': part_counts, 'cv_folds': self.cv_folds}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        self.row_counts_by_family = row_counts
        return metadata

    def _load_family_artifact(self, family, columns=None):
        family_dir = self.artifact_dir / 'train' / family
        paths = sorted(family_dir.glob('*.parquet'))
        if not paths:
            return pd.DataFrame(columns=list(columns or []))
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        return pd.concat(frames, ignore_index=True)

    def _refresh_override_columns(self, df):
        out = df.copy()
        hard_rule_flags = np.column_stack([pd.to_numeric(out[RULE_COLUMN_MAP[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1 for name in HARD_RULE_NAMES])
        if self.hard_override_names:
            hard_override_flags = np.column_stack([pd.to_numeric(out[RULE_COLUMN_MAP[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1 for name in self.hard_override_names])
            out['hard_override_anomaly'] = hard_override_flags.any(axis=1).astype(np.int8)
        else:
            out['hard_override_anomaly'] = np.zeros(len(out), dtype=np.int8)
        out['hard_rule_anomaly'] = hard_rule_flags.any(axis=1).astype(np.int8)
        return out

    def _audit_hard_override_rules(self):
        unique_rule_cols = sorted({RULE_COLUMN_MAP[name] for name in DEFAULT_HARD_OVERRIDE_NAMES})
        total_positive = 0
        per_rule_counts = {name: {'count': 0, 'positives': 0} for name in DEFAULT_HARD_OVERRIDE_NAMES}
        for family in ['canon10', 'canon100', 'other']:
            frame = self._load_family_artifact(family, columns=['Label', *unique_rule_cols])
            if frame.empty:
                continue
            labels = frame['Label'].to_numpy(np.int8)
            total_positive += int(labels.sum())
            for rule_name in DEFAULT_HARD_OVERRIDE_NAMES:
                mask = pd.to_numeric(frame[RULE_COLUMN_MAP[rule_name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1
                if not mask.any():
                    continue
                per_rule_counts[rule_name]['count'] += int(mask.sum())
                per_rule_counts[rule_name]['positives'] += int(labels[mask].sum())
        stats: Dict[str, Dict[str, float]] = {}
        demoted: List[str] = []
        for rule_name, counts in per_rule_counts.items():
            count = counts['count']
            positives = counts['positives']
            precision = float(positives / count) if count else 1.0
            recall = float(positives / total_positive) if total_positive else 0.0
            stats[rule_name] = {'count': int(count), 'positives': int(positives), 'precision': precision, 'recall': recall}
            if count > 0 and precision < MIN_OVERRIDE_PRECISION:
                demoted.append(rule_name)
        self.hard_override_rule_stats = stats
        self.hard_override_names = [name for name in DEFAULT_HARD_OVERRIDE_NAMES if name not in demoted]
        return demoted

    def _capture_semantic_context(self, family):
        return FamilySemanticContext(family=family, surrogate_feature_cols=list(self.surrogate_feature_cols or []), surrogate_models=dict(self.surrogate_models), residual_quantiles=json.loads(json.dumps(self.residual_quantiles)), family_base_rates=dict(self.family_base_rates), scenario_sum_map=dict(self.scenario_sum_map), scenario_count_map=dict(self.scenario_count_map), scenario_output_sum_map=dict(self.scenario_output_sum_map), scenario_output_count_map=dict(self.scenario_output_count_map))

    def _activate_semantic_context(self, context):
        self.surrogate_feature_cols = list(context.surrogate_feature_cols)
        self.surrogate_models = dict(context.surrogate_models)
        self.residual_quantiles = json.loads(json.dumps(context.residual_quantiles))
        self.family_base_rates = dict(context.family_base_rates)
        self.scenario_sum_map = dict(context.scenario_sum_map)
        self.scenario_count_map = dict(context.scenario_count_map)
        self.scenario_output_sum_map = dict(context.scenario_output_sum_map)
        self.scenario_output_count_map = dict(context.scenario_output_count_map)

    def _prepare_family_semantic_frame(self, base_df, y, family):
        work = self._refresh_override_columns(base_df)
        no_valid = pd.Series(np.zeros(len(work), dtype=bool), index=work.index)
        self._fit_surrogate_models(work, y, no_valid)
        work = self._augment_with_surrogates(work)
        self._compute_residual_quantiles(work, y, no_valid)
        work = self._apply_residual_calibration_features(work)
        work = self._fit_transform_scenario_features(work, y)
        work = self._add_family_interaction_features(work)
        return (work, self._capture_semantic_context(family))

    def _semantic_feature_candidates(self, semantic_df):
        excluded = {'Id', 'Label', 'fold_id', 'audit_fold_id', 'hard_override_anomaly', 'device_fingerprint'}
        excluded.update(SAFE_STR.values())
        return [col for col in semantic_df.columns if col not in excluded and pd.api.types.is_numeric_dtype(semantic_df[col])]

    def _prepare_cat_frame(self, base_df):
        out = self._refresh_override_columns(base_df)
        for col in [*SAFE_STR.values(), 'device_fingerprint']:
            if col in out.columns:
                out[col] = out[col].fillna('<NA>').astype(str)
        return out

    def _cat_feature_candidates(self, cat_df):
        raw_numeric_cols = [SAFE_RAW[col] for col in RAW_NUMERIC if SAFE_RAW[col] in cat_df.columns]
        missing_cols = [col for col in cat_df.columns if col.startswith('missing_')]
        categorical_cols = [SAFE_STR[col] for col in RAW_STRING_COLUMNS if SAFE_STR[col] in cat_df.columns]
        candidates = dedupe([*raw_numeric_cols, *categorical_cols, 'device_fingerprint', *CAT_ENGINEERED_COLUMNS, *missing_cols])
        excluded = {'Id', 'Label', 'fold_id', 'audit_fold_id', 'hard_override_anomaly', 'hard_rule_anomaly'}
        return [col for col in candidates if col in cat_df.columns and col not in excluded]

    def _train_semantic_oof(self, semantic_df, y, feature_cols, *, fold_col, fit_final):
        probs = np.ones(len(semantic_df), dtype=np.float32)
        model_mask = semantic_df['hard_override_anomaly'].to_numpy(np.int8) == 0
        fold_ids = semantic_df[fold_col].to_numpy(np.int8)
        final_model: Optional[XGBClassifier] = None
        for fold in range(self.cv_folds):
            train_mask = model_mask & (fold_ids != fold)
            valid_mask = model_mask & (fold_ids == fold)
            if not valid_mask.any():
                continue
            model = self._new_classifier()
            x_train = semantic_df.loc[train_mask, feature_cols]
            y_train = y[train_mask]
            weights = self._build_sample_weights(semantic_df.loc[train_mask], y_train)
            model.fit(x_train, y_train, sample_weight=weights)
            probs[valid_mask] = model.predict_proba(semantic_df.loc[valid_mask, feature_cols])[:, 1].astype(np.float32)
        if fit_final and model_mask.any():
            final_model = self._new_classifier()
            weights = self._build_sample_weights(semantic_df.loc[model_mask], y[model_mask])
            final_model.fit(semantic_df.loc[model_mask, feature_cols], y[model_mask], sample_weight=weights)
        return (probs, final_model)

    def _train_cat_oof(self, cat_df, y, feature_cols, categorical_cols, *, fold_col, fit_final):
        probs = np.ones(len(cat_df), dtype=np.float32)
        model_mask = cat_df['hard_override_anomaly'].to_numpy(np.int8) == 0
        fold_ids = cat_df[fold_col].to_numpy(np.int8)
        final_model: Optional['CatBoostClassifier'] = None
        for fold in range(self.cv_folds):
            train_mask = model_mask & (fold_ids != fold)
            valid_mask = model_mask & (fold_ids == fold)
            if not valid_mask.any():
                continue
            model = self._new_cat_model()
            weights = self._build_sample_weights(cat_df.loc[train_mask], y[train_mask])
            model.fit(cat_df.loc[train_mask, feature_cols], y[train_mask], cat_features=list(categorical_cols), sample_weight=weights, verbose=False)
            probs[valid_mask] = model.predict_proba(cat_df.loc[valid_mask, feature_cols])[:, 1].astype(np.float32)
        if fit_final and model_mask.any():
            final_model = self._new_cat_model()
            weights = self._build_sample_weights(cat_df.loc[model_mask], y[model_mask])
            final_model.fit(cat_df.loc[model_mask, feature_cols], y[model_mask], cat_features=list(categorical_cols), sample_weight=weights, verbose=False)
        return (probs, final_model)

    def _select_family_blend(self, y, hard_override, semantic_primary, semantic_audit, cat_primary, cat_audit):
        baseline_primary_prob = semantic_primary.copy()
        baseline_primary_prob[hard_override] = 1.0
        baseline_thr, _ = self.tune_threshold(y, baseline_primary_prob)
        baseline_pred_primary = (baseline_primary_prob >= baseline_thr).astype(np.int8)
        baseline_audit_prob = semantic_audit.copy()
        baseline_audit_prob[hard_override] = 1.0
        baseline_pred_audit = (baseline_audit_prob >= baseline_thr).astype(np.int8)
        baseline_primary_score = float(fbeta_score(y, baseline_pred_primary, beta=2))
        baseline_audit_score = float(fbeta_score(y, baseline_pred_audit, beta=2))
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
            primary_score = float(fbeta_score(y, pred_primary, beta=2))
            audit_score = float(fbeta_score(y, pred_audit, beta=2))
            if audit_score < baseline_audit_score - AUDIT_TOLERANCE:
                continue
            if primary_score > best_primary_score + 1e-12 or (abs(primary_score - best_primary_score) <= 1e-12 and audit_score > best_audit_for_best):
                best_weight = weight
                best_thr = thr
                best_primary_pred = pred_primary
                best_audit_pred = pred_audit
                best_primary_score = primary_score
                best_audit_for_best = audit_score
        return (best_weight, best_thr, best_primary_pred.astype(np.int8), best_audit_pred.astype(np.int8))

    def fit(self, zip_path):
        metadata = self._build_train_artifacts(zip_path)
        demoted_rules = self._audit_hard_override_rules()
        primary_metrics: Dict[str, MetricSummary] = {}
        audit_metrics: Dict[str, MetricSummary] = {}
        semantic_feature_counts: Dict[str, int] = {}
        cat_feature_counts: Dict[str, int] = {}
        prediction_rows: List[pd.DataFrame] = []
        for family in DEVICE_FAMILY_MAP:
            base_df = self._load_family_artifact(family)
            if base_df.empty:
                continue
            y_series = base_df['Label'].astype(np.int8)
            semantic_df, context = self._prepare_family_semantic_frame(base_df.copy(), y_series, family)
            self.semantic_contexts[family] = context
            semantic_feature_cols = self._select_nonconstant_columns(semantic_df, self._semantic_feature_candidates(semantic_df))
            self.semantic_feature_cols_by_family[family] = semantic_feature_cols
            semantic_feature_counts[family] = len(semantic_feature_cols)
            cat_df = self._prepare_cat_frame(base_df.copy())
            cat_feature_cols = self._select_nonconstant_columns(cat_df, self._cat_feature_candidates(cat_df))
            cat_categorical_cols = [col for col in [*SAFE_STR.values(), 'device_fingerprint'] if col in cat_feature_cols]
            self.cat_feature_cols_by_family[family] = cat_feature_cols
            self.cat_categorical_cols_by_family[family] = cat_categorical_cols
            cat_feature_counts[family] = len(cat_feature_cols)
            y = y_series.to_numpy(np.int8)
            hard_override = semantic_df['hard_override_anomaly'].to_numpy(np.int8) == 1
            semantic_primary_prob, semantic_model = self._train_semantic_oof(semantic_df, y, semantic_feature_cols, fold_col='fold_id', fit_final=True)
            semantic_audit_prob, _ = self._train_semantic_oof(semantic_df, y, semantic_feature_cols, fold_col='audit_fold_id', fit_final=False)
            self.semantic_models[family] = semantic_model
            cat_primary_prob: Optional[np.ndarray] = None
            cat_audit_prob: Optional[np.ndarray] = None
            cat_model: Optional['CatBoostClassifier'] = None
            if cat_feature_cols:
                cat_primary_prob, cat_model = self._train_cat_oof(cat_df, y, cat_feature_cols, cat_categorical_cols, fold_col='fold_id', fit_final=True)
                cat_audit_prob, _ = self._train_cat_oof(cat_df, y, cat_feature_cols, cat_categorical_cols, fold_col='audit_fold_id', fit_final=False)
            self.cat_models[family] = cat_model
            weight, threshold, primary_pred, audit_pred = self._select_family_blend(y, hard_override, semantic_primary_prob, semantic_audit_prob, cat_primary_prob, cat_audit_prob)
            self.family_blend_weights[family] = weight
            self.family_thresholds[family] = threshold
            family_rows = pd.DataFrame({'Id': semantic_df['Id'].to_numpy(np.int64), 'Label': y, 'family': family, 'pred_primary': primary_pred, 'pred_audit': audit_pred})
            prediction_rows.append(family_rows)
            primary_metrics[family] = self._report_rows_to_metric(family_rows, 'pred_primary')
            audit_metrics[family] = self._report_rows_to_metric(family_rows, 'pred_audit')
            print(f'[fit] {family} primary F2={primary_metrics[family].f2:.6f}, audit F2={audit_metrics[family].f2:.6f}, threshold={threshold:.3f}, blend_weight={weight:.2f}')
            del base_df, semantic_df, cat_df, family_rows
            gc.collect()
        other_df = self._load_family_artifact('other', columns=['Id', 'Label'])
        if not other_df.empty:
            other_rows = other_df.copy()
            other_rows['family'] = 'other'
            other_rows['pred_primary'] = 1
            other_rows['pred_audit'] = 1
            prediction_rows.append(other_rows[['Id', 'Label', 'family', 'pred_primary', 'pred_audit']])
            primary_metrics['other'] = self._report_rows_to_metric(other_rows, 'pred_primary')
            audit_metrics['other'] = self._report_rows_to_metric(other_rows, 'pred_audit')
        if not prediction_rows:
            raise RuntimeError('No training artifacts were available for model fitting.')
        all_predictions = pd.concat(prediction_rows, ignore_index=True)
        primary_metrics['overall'] = self._report_rows_to_metric(all_predictions, 'pred_primary')
        audit_metrics['overall'] = self._report_rows_to_metric(all_predictions, 'pred_audit')
        report = TrainingReport(primary_metrics=primary_metrics, audit_metrics=audit_metrics, family_thresholds=dict(self.family_thresholds), family_blend_weights=dict(self.family_blend_weights), semantic_feature_counts=semantic_feature_counts, cat_feature_counts=cat_feature_counts, active_hard_override_names=list(self.hard_override_names), demoted_hard_override_names=demoted_rules, hard_override_rule_stats=dict(self.hard_override_rule_stats), artifact_row_counts={k: int(v) for k, v in metadata.get('row_counts', {}).items()}, artifact_dir=str(self.artifact_dir))
        self.training_report = report
        print(f"[fit] overall primary F2={report.primary_metrics['overall'].f2:.6f}, precision={report.primary_metrics['overall'].precision:.4f}, recall={report.primary_metrics['overall'].recall:.4f}")
        print(f"[fit] overall audit F2={report.audit_metrics['overall'].f2:.6f}, precision={report.audit_metrics['overall'].precision:.4f}, recall={report.audit_metrics['overall'].recall:.4f}")
        return report

    def _predict_family_chunk(self, family, base_df):
        if family not in self.semantic_contexts or family not in self.semantic_models:
            raise RuntimeError(f'Missing fitted semantic model bundle for family {family}.')
        context = self.semantic_contexts[family]
        self._activate_semantic_context(context)
        work = self._refresh_override_columns(base_df.copy())
        hard_override = work['hard_override_anomaly'].to_numpy(np.int8) == 1
        semantic_df = self._augment_with_surrogates(work.copy())
        semantic_df = self._apply_residual_calibration_features(semantic_df)
        semantic_df = self._apply_scenario_features(semantic_df)
        semantic_df = self._add_family_interaction_features(semantic_df)
        semantic_prob = np.ones(len(semantic_df), dtype=np.float32)
        if (~hard_override).any():
            semantic_model = self.semantic_models[family]
            semantic_prob[~hard_override] = semantic_model.predict_proba(semantic_df.loc[~hard_override, self.semantic_feature_cols_by_family[family]])[:, 1].astype(np.float32)
        cat_prob: Optional[np.ndarray] = None
        cat_model = self.cat_models.get(family)
        if cat_model is not None and self.cat_feature_cols_by_family.get(family):
            cat_df = self._prepare_cat_frame(base_df.copy())
            cat_prob = np.ones(len(cat_df), dtype=np.float32)
            if (~hard_override).any():
                cat_prob[~hard_override] = cat_model.predict_proba(cat_df.loc[~hard_override, self.cat_feature_cols_by_family[family]])[:, 1].astype(np.float32)
        blend_prob = self._blend_probs(semantic_prob, cat_prob, self.family_blend_weights.get(family, 1.0))
        blend_prob[hard_override] = 1.0
        pred = (blend_prob >= self.family_thresholds.get(family, 0.5)).astype(np.int8)
        pred[hard_override] = 1
        return pred

    def predict_test(self, zip_path, out_csv):
        if self.training_report is None:
            raise RuntimeError('Model is not fitted.')
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        positive_rows = 0
        with out_csv.open('w', encoding='utf-8') as fh:
            fh.write('Id,Label\n')
            for chunk_idx, chunk in enumerate(self.iter_raw_chunks(zip_path, 'test.csv', USECOLS_TEST, 0)):
                feats = self.build_features(chunk)
                pred = feats['hard_override_anomaly'].astype(np.int8).to_numpy()
                for family in DEVICE_FAMILY_MAP:
                    family_mask = feats['device_family'] == family
                    if not family_mask.any():
                        continue
                    family_df = feats.loc[family_mask].copy()
                    family_pred = self._predict_family_chunk(family, family_df)
                    pred[np.flatnonzero(family_mask.to_numpy())] = family_pred
                out = pd.DataFrame({'Id': feats['Id'].astype(np.int64), 'Label': pred.astype(np.int8)})
                out.to_csv(fh, index=False, header=False)
                total_rows += len(out)
                positive_rows += int(out['Label'].sum())
                if chunk_idx % 10 == 0:
                    print(f'[test] wrote {total_rows:,} predictions')
        print(f'[test] done; total_rows={total_rows:,}, positive_rows={positive_rows:,}, positive_rate={positive_rows / max(total_rows, 1):.6f}')

    def save(self, model_path, report_path):
        if self.training_report is None:
            raise RuntimeError('Nothing to save; fit the model first.')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_dir = model_path.parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {'semantic_models': {}, 'cat_models': {}, 'surrogates': {}, 'active_hard_override_names': self.hard_override_names}
        for family, model in self.semantic_models.items():
            if model is None:
                continue
            path = model_dir / f'semantic_{family}.json'
            model.save_model(path)
            manifest['semantic_models'][family] = path.name
        for family, model in self.cat_models.items():
            if model is None:
                continue
            path = model_dir / f'cat_{family}.cbm'
            model.save_model(path)
            manifest['cat_models'][family] = path.name
        surrogate_dir = model_dir / 'surrogates'
        surrogate_dir.mkdir(parents=True, exist_ok=True)
        for family, context in self.semantic_contexts.items():
            for (ctx_family, target_name), model in context.surrogate_models.items():
                path = surrogate_dir / f'{ctx_family}_{target_name}.json'
                model.save_model(path)
                manifest['surrogates'].setdefault(family, {})[target_name] = path.name
        model_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        payload = self.training_report.as_dict()
        payload['semantic_feature_cols_by_family'] = self.semantic_feature_cols_by_family
        payload['cat_feature_cols_by_family'] = self.cat_feature_cols_by_family
        payload['cat_categorical_cols_by_family'] = self.cat_categorical_cols_by_family
        payload['semantic_context_metadata'] = {family: {'surrogate_feature_cols': context.surrogate_feature_cols, 'residual_quantiles': context.residual_quantiles, 'family_base_rates': context.family_base_rates} for family, context in self.semantic_contexts.items()}
        report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
DEFAULT_RUN_CONFIG = RunConfig()

def run_pipeline(config=DEFAULT_RUN_CONFIG):
    seed_everything(config.seed)
    baseline = config.create_baseline()
    report = baseline.fit(config.zip_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / MODEL_FILENAME
    report_path = config.output_dir / REPORT_FILENAME
    baseline.save(model_path, report_path)
    final_solution_path = report_path
    final_solution_label = 'validation_report'
    if config.write_test_predictions:
        submission_path = config.output_dir / 'submission_full_data.csv'
        baseline.predict_test(config.zip_path, submission_path)
        final_solution_path = submission_path
        final_solution_label = 'submission'
    print(f'[solution] {final_solution_label}_sha256={file_sha256(final_solution_path)} path={final_solution_path}')
    return report

def main():
    run_pipeline()
if __name__ == '__main__':
    main()
