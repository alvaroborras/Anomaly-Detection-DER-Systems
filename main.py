#!/usr/bin/env python3
import gc
import math
import random
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from xgboost import XGBClassifier as X,XGBRegressor as G
try:
    from catboost import CatBoostClassifier as C
except ImportError:
    C = None

def dedupe(columns):
    return list(dict.fromkeys(columns))

def p(prefix, fields):
    return [f'{prefix}.{field}' for field in fields]

def bvv(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrv', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCrv']
    for curve in range(3):
        cp = f'{prefix}.Crv[{curve}]'
        cols.extend([f'{cp}.ActPt', f'{cp}.DeptRef', f'{cp}.Pri', f'{cp}.VRef', f'{cp}.VRefAuto', f'{cp}.VRefAutoEna', f'{cp}.VRefAutoTms', f'{cp}.RspTms', f'{cp}.ReadOnly'])
        for point in range(4):
            cols.extend([f'{cp}.Pt[{point}].V', f'{cp}.Pt[{point}].Var'])
    return cols

def bvw(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrv', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCrv']
    for curve in range(3):
        cp = f'{prefix}.Crv[{curve}]'
        cols.extend([f'{cp}.ActPt', f'{cp}.DeptRef', f'{cp}.RspTms', f'{cp}.ReadOnly'])
        for point in range(2):
            cols.extend([f'{cp}.Pt[{point}].V', f'{cp}.Pt[{point}].W'])
    return cols

def bwv(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrv', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCrv']
    for curve in range(3):
        cp = f'{prefix}.Crv[{curve}]'
        cols.extend([f'{cp}.ActPt', f'{cp}.DeptRef', f'{cp}.Pri', f'{cp}.ReadOnly'])
        for point in range(6):
            cols.extend([f'{cp}.Pt[{point}].W', f'{cp}.Pt[{point}].Var'])
    return cols

def bfd(prefix):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCtlReq', f'{prefix}.AdptCtlRslt', f'{prefix}.NCtl', f'{prefix}.RvrtTms', f'{prefix}.RvrtRem', f'{prefix}.RvrtCtl']
    for ctl in range(3):
        ctl_prefix = f'{prefix}.Ctl[{ctl}]'
        cols.extend([f'{ctl_prefix}.DbOf', f'{ctl_prefix}.DbUf', f'{ctl_prefix}.KOf', f'{ctl_prefix}.KUf', f'{ctl_prefix}.RspTms', f'{ctl_prefix}.PMin', f'{ctl_prefix}.ReadOnly'])
    return cols

def btc(prefix, axis_name):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrvSet']
    for curve in range(2):
        cp = f'{prefix}.Crv[{curve}]'
        cols.append(f'{cp}.ReadOnly')
        for group in ['MustTrip', 'MayTrip', 'MomCess']:
            group_prefix = f'{cp}.{group}'
            cols.append(f'{group_prefix}.ActPt')
            for point in range(5):
                cols.extend([f'{group_prefix}.Pt[{point}].{axis_name}', f'{group_prefix}.Pt[{point}].Tms'])
    return cols
COMMON_FIELDS = 'Mn Md Opt Vr SN'.split()
CS = p('common[0]', COMMON_FIELDS)
COMMON_COLUMNS = p('common[0]', ['ID', 'L', *COMMON_FIELDS, 'DA'])
MEASURE_AC_FIELDS = '\nID L ACType W VA Var PF A LLV LNV Hz TmpAmb TmpCab TmpSnk TmpTrns TmpSw TmpOt\nThrotPct ThrotSrc WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3\nAL1 AL2 AL3 VL1L2 VL2L3 VL3L1 VL1 VL2 VL3\n'.split()
MEASURE_AC_COLUMNS = p('DERMeasureAC[0]', MEASURE_AC_FIELDS)
CAPACITY_FIELDS = '\nID L WMaxRtg VAMaxRtg VarMaxInjRtg VarMaxAbsRtg WChaRteMaxRtg WDisChaRteMaxRtg\nVAChaRteMaxRtg VADisChaRteMaxRtg VNomRtg VMaxRtg VMinRtg AMaxRtg PFOvrExtRtg\nPFUndExtRtg NorOpCatRtg AbnOpCatRtg IntIslandCatRtg WMax WMaxOvrExt WOvrExtPF\nWMaxUndExt WUndExtPF VAMax VarMaxInj VarMaxAbs WChaRteMax WDisChaRteMax\nVAChaRteMax VADisChaRteMax VNom VMax VMin AMax PFOvrExt PFUndExt CtrlModes\nIntIslandCat\n'.split()
CAPACITY_COLUMNS = p('DERCapacity[0]', CAPACITY_FIELDS)
ENTER_SERVICE_FIELDS = 'ID L ES ESVHi ESVLo ESHzHi ESHzLo ESDlyTms ESRndTms ESRmpTms ESDlyRemTms'.split()
ENTER_SERVICE_COLUMNS = p('DEREnterService[0]', ENTER_SERVICE_FIELDS)
CTL_AC_FIELDS = '\nID L PFWInjEna PFWInjEnaRvrt PFWInjRvrtTms PFWInjRvrtRem PFWAbsEna PFWAbsEnaRvrt\nPFWAbsRvrtTms PFWAbsRvrtRem WMaxLimPctEna WMaxLimPct WMaxLimPctRvrt\nWMaxLimPctEnaRvrt WMaxLimPctRvrtTms WMaxLimPctRvrtRem WSetEna WSetMod WSet\nWSetRvrt WSetPct WSetPctRvrt WSetEnaRvrt WSetRvrtTms WSetRvrtRem VarSetEna\nVarSetMod VarSetPri VarSet VarSetRvrt VarSetPct VarSetPctRvrt VarSetEnaRvrt\nVarSetRvrtTms VarSetRvrtRem WRmp WRmpRef VarRmp AntiIslEna PFWInj.PF PFWInj.Ext\nPFWInjRvrt.PF PFWInjRvrt.Ext PFWAbs.Ext PFWAbsRvrt.Ext\n'.split()
CTL_AC_COLUMNS = p('DERCtlAC[0]', CTL_AC_FIELDS)
VOLT_VAR_COLUMNS = bvv('DERVoltVar[0]')
VOLT_WATT_COLUMNS = bvw('DERVoltWatt[0]')
FREQ_DROOP_COLUMNS = bfd('DERFreqDroop[0]')
WATT_VAR_COLUMNS = bwv('DERWattVar[0]')
TS = {'lv': ('DERTripLV[0]', 'V', 'low'), 'hv': ('DERTripHV[0]', 'V', 'high'), 'lf': ('DERTripLF[0]', 'Hz', 'low'), 'hf': ('DERTripHF[0]', 'Hz', 'high')}
TRIP_COLUMNS = {sn: btc(prefix, axis_name) for sn, (prefix, axis_name, _) in TS.items()}
MEASURE_DC_FIELDS = '\nID L NPrt DCA DCW Prt[0].PrtTyp Prt[0].ID Prt[0].DCA Prt[0].DCV Prt[0].DCW\nPrt[0].Tmp Prt[1].PrtTyp Prt[1].ID Prt[1].DCA Prt[1].DCV Prt[1].DCW Prt[1].Tmp\n'.split()
MEASURE_DC_COLUMNS = p('DERMeasureDC[0]', MEASURE_DC_FIELDS)
BSC = {'common': COMMON_COLUMNS, 'measure_ac': MEASURE_AC_COLUMNS, 'capacity': CAPACITY_COLUMNS, 'enter_service': ENTER_SERVICE_COLUMNS, 'ctl_ac': CTL_AC_COLUMNS, 'volt_var': VOLT_VAR_COLUMNS, 'volt_watt': VOLT_WATT_COLUMNS, 'freq_droop': FREQ_DROOP_COLUMNS, 'watt_var': WATT_VAR_COLUMNS, 'measure_dc': MEASURE_DC_COLUMNS}
for sn, cols in TRIP_COLUMNS.items():
    BSC[f'trip_{sn}'] = cols
CURVE_BLOCK_META_FIELDS = 'Ena AdptCrvReq AdptCrvRslt NPt NCrv RvrtTms RvrtRem RvrtCrv'.split()
FREQ_DROOP_META_FIELDS = 'Ena AdptCtlReq AdptCtlRslt NCtl RvrtTms RvrtRem RvrtCtl'.split()
TRIP_META_FIELDS = 'Ena AdptCrvReq AdptCrvRslt NPt NCrvSet'.split()
RN = dedupe(['common[0].DA', *p('DERMeasureAC[0]', MEASURE_AC_FIELDS[2:]), *p('DERCapacity[0]', CAPACITY_FIELDS[2:]), *p('DEREnterService[0]', ENTER_SERVICE_FIELDS[2:]), *p('DERCtlAC[0]', CTL_AC_FIELDS[2:]), *p('DERVoltVar[0]', CURVE_BLOCK_META_FIELDS), *p('DERVoltWatt[0]', CURVE_BLOCK_META_FIELDS), *p('DERFreqDroop[0]', FREQ_DROOP_META_FIELDS), *p('DERWattVar[0]', CURVE_BLOCK_META_FIELDS), *p('DERMeasureDC[0]', MEASURE_DC_FIELDS[2:])])
TRIP_META_COLUMNS = [f'{prefix}.{field}' for prefix, _, _ in TS.values() for field in TRIP_META_FIELDS]
RAW_EXTRA_NUMERIC_COLUMNS = ['DERMeasureAC[0].A_SF', 'DERMeasureAC[0].V_SF', 'DERMeasureAC[0].Hz_SF', 'DERMeasureAC[0].W_SF', 'DERMeasureAC[0].PF_SF', 'DERMeasureAC[0].VA_SF', 'DERMeasureAC[0].Var_SF', 'DERCapacity[0].WOvrExtRtg', 'DERCapacity[0].WOvrExtRtgPF', 'DERCapacity[0].WUndExtRtg', 'DERCapacity[0].WUndExtRtgPF', 'DERCapacity[0].W_SF', 'DERCapacity[0].PF_SF', 'DERCapacity[0].VA_SF', 'DERCapacity[0].Var_SF', 'DERCapacity[0].V_SF', 'DERCapacity[0].A_SF', 'DERCtlAC[0].WSet_SF', 'DERMeasureDC[0].DCA_SF', 'DERMeasureDC[0].DCW_SF']
RAW_EXTRA_STRING_COLUMNS = ['DERMeasureDC[0].Prt[0].IDStr', 'DERMeasureDC[0].Prt[1].IDStr']
RN = dedupe([*RN, *TRIP_META_COLUMNS, *RAW_EXTRA_NUMERIC_COLUMNS])
RSC = dedupe([*CS, *RAW_EXTRA_STRING_COLUMNS])
TRIP_SOURCE_COLUMNS = [col for cols in TRIP_COLUMNS.values() for col in cols]
ALL_SOURCE_COLUMNS = dedupe([*COMMON_COLUMNS, *MEASURE_AC_COLUMNS, *CAPACITY_COLUMNS, *ENTER_SERVICE_COLUMNS, *CTL_AC_COLUMNS, *VOLT_VAR_COLUMNS, *VOLT_WATT_COLUMNS, *FREQ_DROOP_COLUMNS, *WATT_VAR_COLUMNS, *MEASURE_DC_COLUMNS, *TRIP_SOURCE_COLUMNS, *RAW_EXTRA_NUMERIC_COLUMNS, *RAW_EXTRA_STRING_COLUMNS])
NSC = [c for c in ALL_SOURCE_COLUMNS if c not in RSC]
UTR = dedupe(['Id', 'Label', *ALL_SOURCE_COLUMNS])
UTE = dedupe(['Id', *ALL_SOURCE_COLUMNS])
CANON1 = 'DERSec|DER Simulator|10 kW DER|1.2.3|SN-Three-Phase'
CANON2 = 'DERSec|DER Simulator 100 kW|1.2.3.1|1.0.0|1100058974'
SR = {c: re.sub('[^0-9A-Za-z_]+', '_', c) for c in RN}
SS = {c: re.sub('[^0-9A-Za-z_]+', '_', c) for c in RSC}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'full_data_hybrid'
SQRT3 = math.sqrt(3.0)
FAM = {'canon10': 0, 'canon100': 1}
RTL = {'tail': 0.95, 'extreme': 0.99, 'ultra': 0.999}
RTF = {'tail': 0.05, 'extreme': 0.1, 'ultra': 0.2}
FTF = 0.02
MT = 0.6
CNW = 1.5
HARD_OVERRIDE_TRAIN_WEIGHT = 0.35
SM = 50.0
AT = 0.003
MOP = 0.995
CIF = ['hard_rule_score', 'scenario_rate', 'scenario_output_rate', 'resid_quantile_score', 'mode_dispatch_w_resid']
STG = {'w': ('DERMeasureAC_0_W', 'DERCapacity_0_WMaxRtg'), 'va': ('DERMeasureAC_0_VA', 'DERCapacity_0_VAMaxRtg'), 'var': ('DERMeasureAC_0_Var', 'DERCapacity_0_VarMaxInjRtg'), 'pf': ('DERMeasureAC_0_PF', None), 'a': ('DERMeasureAC_0_A', 'DERCapacity_0_AMaxRtg')}
SLF = {*(f'DERMeasureAC_0_{field}' for field in '\n    W VA Var PF A WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3\n    AL1 AL2 AL3\n    '.split()), *'\n    w_over_wmaxrtg w_over_wmax va_over_vamax va_over_vamaxrtg var_over_injmax\n    var_over_absmax a_over_amax w_minus_wmax w_minus_wmaxrtg va_minus_vamax\n    var_minus_injmax var_plus_absmax w_eq_wmaxrtg w_eq_wmax var_eq_varmaxinj\n    var_eq_neg_varmaxabs pf_sign_mismatch w_gt_wmax_tol w_gt_wmaxrtg_tol\n    va_gt_vamax_tol var_gt_injmax_tol var_lt_absmax_tol va_minus_pqmag\n    va_over_pqmag pf_from_w_va pf_error w_phase_sum_error va_phase_sum_error\n    var_phase_sum_error phase_w_spread phase_var_spread wset_abs_error\n    wsetpct_target wsetpct_abs_error wmaxlim_target wmaxlim_excess\n    varset_abs_error varsetpct_target varsetpct_abs_error wset_enabled_far\n    wsetpct_enabled_far wmaxlim_enabled_far varsetpct_enabled_far w_pct_of_rtg\n    var_pct_of_limit enter_service_blocked_power enter_service_blocked_va\n    enter_service_blocked_current pf_inj_target_error pf_inj_reversion_error\n    pf_reactive_near_limit trip_lv_power_when_outside trip_hv_power_when_outside\n    trip_lf_power_when_outside trip_hf_power_when_outside\n    trip_any_power_when_outside voltvar_curve_error voltwatt_curve_error\n    wattvar_curve_expected wattvar_curve_error freqdroop_w_over_pmin_pct\n    dcw_over_w dcw_over_abs_w ac_zero_dc_positive ac_positive_dc_zero\n    ac_dc_same_sign\n    '.split()}
HARD_RULE_NAMES = ['noncanonical', 'common_missing', 'w_gt_wmax', 'w_gt_wmaxrtg', 'va_gt_vamax', 'var_gt_injmax', 'var_lt_absmax', 'wset_far', 'wsetpct_far', 'wmaxlim_far', 'varsetpct_far', 'model_structure', 'ac_type_rare', 'dc_type_rare', 'enter_state', 'enter_blocked_power', 'enter_blocked_current', 'pf_abs', 'pf_abs_rvrt', 'trip_power']
OVR = ['noncanonical', 'common_missing', 'w_gt_wmax', 'w_gt_wmaxrtg', 'va_gt_vamax', 'var_gt_injmax', 'var_lt_absmax', 'wset_far', 'wsetpct_far', 'model_structure', 'ac_type_rare', 'dc_type_rare', 'enter_state', 'pf_abs', 'pf_abs_rvrt', 'trip_power']
RCM = {'noncanonical': 'noncanonical', 'common_missing': 'common_missing_any', 'w_gt_wmax': 'w_gt_wmax_tol', 'w_gt_wmaxrtg': 'w_gt_wmaxrtg_tol', 'va_gt_vamax': 'va_gt_vamax_tol', 'var_gt_injmax': 'var_gt_injmax_tol', 'var_lt_absmax': 'var_lt_absmax_tol', 'wset_far': 'wset_enabled_far', 'wsetpct_far': 'wsetpct_enabled_far', 'wmaxlim_far': 'wmaxlim_enabled_far', 'varsetpct_far': 'varsetpct_enabled_far', 'model_structure': 'model_structure_anomaly_any', 'ac_type_rare': 'ac_type_is_rare', 'dc_type_rare': 'dc_port_type_rare_any', 'enter_state': 'enter_service_state_anomaly', 'enter_blocked_power': 'enter_service_blocked_power', 'enter_blocked_current': 'enter_service_blocked_current', 'pf_abs': 'pf_abs_ext_present', 'pf_abs_rvrt': 'pf_abs_rvrt_ext_present', 'trip_power': 'trip_any_power_when_outside'}
CEC = ['device_fingerprint', 'common_missing_pattern', 'enter_service_missing_pattern', 'missing_selected_total', 'missing_selected_blocks', 'common_missing_any', 'common_missing_count', 'common_sn_has_decimal_suffix']
EMM = {'common': ('common[0].ID', 'common[0].L', 1.0, 66.0), 'measure_ac': ('DERMeasureAC[0].ID', 'DERMeasureAC[0].L', 701.0, 153.0), 'capacity': ('DERCapacity[0].ID', 'DERCapacity[0].L', 702.0, 50.0), 'enter_service': ('DEREnterService[0].ID', 'DEREnterService[0].L', 703.0, 17.0), 'measure_dc': ('DERMeasureDC[0].ID', 'DERMeasureDC[0].L', 714.0, 68.0)}

def s(seed):
    random.seed(seed)
    np.random.seed(seed)

class R:

    def __init__(self, *, artifact_dir=DEFAULT_OUTPUT_DIR / 'artifacts', chunksize=5000, cv_folds=5, n_estimators=180, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, cat_iterations=400, cat_depth=8, cat_learning_rate=0.05, n_jobs=4, seed=DEFAULT_SEED):
        self.artifact_dir = artifact_dir
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
        self.ovr = list(OVR)
        self.semantic_models = {}
        self.cat_models = {}
        self.ctx = {}
        self.thr = {'canon10': 0.5, 'canon100': 0.5}
        self.blend_w = {'canon10': 1.0, 'canon100': 1.0}
        self.sem_cols = {}
        self.cat_cols = {}
        self.sur_cols = None
        self.surrogate_models = {}
        self.res_q = {}
        self.family_base_rates = {}
        self.ssm = {}
        self.scm = {}
        self.sosm = {}
        self.socm = {}

    @staticmethod
    def _d(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        out = np.full_like(a, np.nan)
        mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-06)
        out[mask] = a[mask] / b[mask]
        return out

    @staticmethod
    def _n(arr):
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
    def _x(arr):
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
    def _cs(curves, idx):
        stacked = np.stack(curves, axis=1)
        return np.take_along_axis(stacked, idx[:, None], axis=1)[:, 0]

    @staticmethod
    def _cp(curves, idx):
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
        return (R._nanmean_rows(slopes), R._x(np.abs(slopes)))

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
        return 100.0 * R._d(var, denom)

    def _coerce_numeric(self, df):
        for col in NSC:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    def _add_block_missingness(self, data, df):
        block_missing_total = np.zeros(len(df), dtype=np.int16)
        block_missing_any = np.zeros(len(df), dtype=np.int16)
        for bn, cols in BSC.items():
            missing = df[cols].isna()
            missing_count = missing.sum(axis=1).astype(np.int16).to_numpy()
            data[f'missing_{bn}_count'] = missing_count
            data[f'missing_{bn}_any'] = (missing_count > 0).astype(np.int8)
            block_missing_total += missing_count
            block_missing_any += (missing_count > 0).astype(np.int16)
        data['missing_selected_total'] = block_missing_total
        data['missing_selected_blocks'] = block_missing_any.astype(np.int8)
        common_missing = df[[*CS, 'common[0].ID', 'common[0].L']].isna().to_numpy(dtype=np.uint16)
        common_weights = (1 << np.arange(common_missing.shape[1], dtype=np.uint16)).reshape(1, -1)
        data['common_missing_pattern'] = (common_missing * common_weights).sum(axis=1).astype(np.int16)
        enter_missing = df[ENTER_SERVICE_COLUMNS].isna().to_numpy(dtype=np.uint16)
        enter_weights = (1 << np.arange(enter_missing.shape[1], dtype=np.uint16)).reshape(1, -1)
        data['enter_service_missing_pattern'] = (enter_missing * enter_weights).sum(axis=1).astype(np.int16)

    def _add_model_integrity_features(self, data, df):
        anomaly_sum = np.zeros(len(df), dtype=np.int16)
        missing_sum = np.zeros(len(df), dtype=np.int16)
        for bn, (id_col, len_col, expected_id, expected_len) in EMM.items():
            raw_id = df[id_col].to_numpy(float)
            raw_len = df[len_col].to_numpy(float)
            id_missing = ~np.isfinite(raw_id)
            len_missing = ~np.isfinite(raw_len)
            id_match = np.isclose(raw_id, expected_id, equal_nan=False)
            len_match = np.isclose(raw_len, expected_len, equal_nan=False)
            data[f'{bn}_model_id_missing'] = id_missing.astype(np.int8)
            data[f'{bn}_model_len_missing'] = len_missing.astype(np.int8)
            data[f'{bn}_model_id_match'] = id_match.astype(np.int8)
            data[f'{bn}_model_len_match'] = len_match.astype(np.int8)
            data[f'{bn}_model_integrity_ok'] = (id_match & len_match).astype(np.int8)
            mismatch = ~id_missing & ~id_match | ~len_missing & ~len_match
            data[f'{bn}_model_structure_anomaly'] = mismatch.astype(np.int8)
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
        data['charge_rate_share_rtg'] = self._d(wcha_rtg, wmaxrtg)
        data['discharge_rate_share_rtg'] = self._d(wdis_rtg, wmaxrtg)
        data['charge_va_share_rtg'] = self._d(vacha_rtg, vamaxrtg)
        data['discharge_va_share_rtg'] = self._d(vadis_rtg, vamaxrtg)
        data['charge_rate_share_setting'] = self._d(wcha, wmax)
        data['discharge_rate_share_setting'] = self._d(wdis, wmax)
        data['charge_va_share_setting'] = self._d(vacha, vamax)
        data['discharge_va_share_setting'] = self._d(vadis, vamax)
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
        temp_min = self._n(temps)
        temp_max = self._x(temps)
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

    def _add_trip_block_features(self, data, df, *, sn, prefix, axis_name, mode, mv, abs_w, tolw):
        adpt_idx = self._curve_index(df[f'{prefix}.AdptCrvRslt'].to_numpy(float), 2)
        group_scalar = lambda group, field: self._cs([df[f'{prefix}.Crv[{curve}].{group}.{field}'].to_numpy(float) for curve in range(2)], adpt_idx)
        group_points = lambda group, field: self._cp([np.column_stack([df[f'{prefix}.Crv[{curve}].{group}.Pt[{i}].{field}'].to_numpy(float) for i in range(5)]) for curve in range(2)], adpt_idx)
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
        must_x_min = self._n(must_x)
        must_x_max = self._x(must_x)
        must_t_min = self._n(must_t)
        must_t_max = self._x(must_t)
        mom_x_min = self._n(mom_x)
        mom_x_max = self._x(mom_x)
        mom_t_min = self._n(mom_t)
        mom_t_max = self._x(mom_t)
        if mode == 'low':
            margin = mv - must_x_max
        else:
            margin = must_x_min - mv
        outside = enabled & np.isfinite(margin) & (margin < 0)
        power_when_outside = outside & (abs_w > tolw)
        envelope_gap = np.where(np.isfinite(mom_x_min) & np.isfinite(must_x_max), np.abs(mom_x_min - must_x_max), np.nan)
        data[f'trip_{sn}_curve_idx'] = adpt_idx.astype(np.int8)
        data[f'trip_{sn}_enabled'] = enabled.astype(np.int8)
        data[f'trip_{sn}_curve_req_gap'] = (df[f'{prefix}.AdptCrvReq'].to_numpy(float) - df[f'{prefix}.AdptCrvRslt'].to_numpy(float)).astype(np.float32)
        data[f'trip_{sn}_musttrip_count'] = must_count
        data[f'trip_{sn}_musttrip_actpt_gap'] = (must_actpt - must_count).astype(np.float32)
        data[f'trip_{sn}_musttrip_axis_min'] = must_x_min
        data[f'trip_{sn}_musttrip_axis_max'] = must_x_max
        data[f'trip_{sn}_musttrip_axis_span'] = (must_x_max - must_x_min).astype(np.float32)
        data[f'trip_{sn}_musttrip_tms_span'] = (must_t_max - must_t_min).astype(np.float32)
        data[f'trip_{sn}_musttrip_reverse_steps'] = self._curve_reverse_steps(must_x)
        data[f'trip_{sn}_momcess_count'] = mom_count
        data[f'trip_{sn}_momcess_actpt_gap'] = (mom_actpt - mom_count).astype(np.float32)
        data[f'trip_{sn}_momcess_axis_span'] = (mom_x_max - mom_x_min).astype(np.float32)
        data[f'trip_{sn}_momcess_tms_span'] = (mom_t_max - mom_t_min).astype(np.float32)
        data[f'trip_{sn}_momcess_reverse_steps'] = self._curve_reverse_steps(mom_x)
        data[f'trip_{sn}_maytrip_present_any'] = np.isfinite(may_present).any(axis=1).astype(np.int8)
        data[f'trip_{sn}_musttrip_margin'] = margin.astype(np.float32)
        data[f'trip_{sn}_outside_musttrip'] = outside.astype(np.int8)
        data[f'trip_{sn}_power_when_outside'] = power_when_outside.astype(np.int8)
        data[f'trip_{sn}_momcess_musttrip_gap'] = envelope_gap.astype(np.float32)
        return (outside.astype(np.int8), power_when_outside.astype(np.int8))

    def _add_curve_block_features(self, data, *, name, raw_idx, curve_x, curve_y, curve_actpt, curve_meta, mv, observed_value=None):
        adpt_idx = self._curve_index(raw_idx, len(curve_x))
        selected_x = self._cp(curve_x, adpt_idx)
        selected_y = self._cp(curve_y, adpt_idx)
        selected_actpt = self._cs(curve_actpt, adpt_idx)
        data[f'{name}_curve_idx'] = adpt_idx.astype(np.int8)
        point_count = self._pair_point_count(selected_x, selected_y)
        data[f'{name}_curve_point_count'] = point_count
        data[f'{name}_curve_actpt_gap'] = (selected_actpt - point_count).astype(np.float32)
        x_min = self._n(selected_x)
        x_max = self._x(selected_x)
        y_min = self._n(selected_y)
        y_max = self._x(selected_y)
        mean_slope, max_abs_slope = self._curve_slope_stats(selected_x, selected_y)
        data[f'{name}_curve_x_span'] = (x_max - x_min).astype(np.float32)
        data[f'{name}_curve_y_span'] = (y_max - y_min).astype(np.float32)
        data[f'{name}_curve_reverse_steps'] = self._curve_reverse_steps(selected_x)
        data[f'{name}_curve_mean_slope'] = mean_slope
        data[f'{name}_curve_max_abs_slope'] = max_abs_slope
        data[f'{name}_curve_measure_margin_low'] = (mv - x_min).astype(np.float32)
        data[f'{name}_curve_measure_margin_high'] = (x_max - mv).astype(np.float32)
        if observed_value is not None:
            expected_value = self._piecewise_interp(mv, selected_x, selected_y)
            data[f'{name}_curve_expected'] = expected_value.astype(np.float32)
            data[f'{name}_curve_error'] = (observed_value - expected_value).astype(np.float32)
        for meta_name, curves in curve_meta.items():
            data[f'{name}_curve_{meta_name}'] = self._cs(curves, adpt_idx).astype(np.float32)

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
        dbof = self._cs(dbof_curves, ctl_idx)
        dbuf = self._cs(dbuf_curves, ctl_idx)
        kof = self._cs(kof_curves, ctl_idx)
        kuf = self._cs(kuf_curves, ctl_idx)
        rsp = self._cs(rsp_curves, ctl_idx)
        pmin = self._cs(pmin_curves, ctl_idx)
        readonly = self._cs(ro_curves, ctl_idx)
        over_activation = np.maximum(hz - (60.0 + dbof), 0.0)
        under_activation = np.maximum(60.0 - dbuf - hz, 0.0)
        expected_delta_pct = 100.0 * self._d(over_activation, kof) - 100.0 * self._d(under_activation, kuf)
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
        data['freqdroop_db_span'] = (self._x(np.column_stack([dbof_stack, dbuf_stack])) - self._n(np.column_stack([dbof_stack, dbuf_stack]))).astype(np.float32)
        data['freqdroop_k_span'] = (self._x(k_stack) - self._n(k_stack)).astype(np.float32)
        data['freqdroop_pmin_span'] = (self._x(pmin_stack) - self._n(pmin_stack)).astype(np.float32)

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
        data['dcw_over_w'] = self._d(dcw, w)
        data['dcw_over_abs_w'] = self._d(dcw, abs_w)
        data['dcw_minus_port_sum'] = (dcw - (prt0 + prt1)).astype(np.float32)
        data['dcv_spread'] = np.abs(prt0_v - prt1_v).astype(np.float32)
        data['dca_spread'] = np.abs(prt0_a - prt1_a).astype(np.float32)
        data['dc_port0_share'] = self._d(prt0, prt0 + prt1)
        data['dc_port_type_mismatch'] = (np.isfinite(prt0_t) & np.isfinite(prt1_t) & (prt0_t != prt1_t)).astype(np.int8)
        rare_type = (prt0_t == 7) | (prt1_t == 7)
        data['dc_port_type_rare_any'] = rare_type.astype(np.int8)
        data['ac_zero_dc_positive'] = ((np.abs(w) <= 1e-06) & (dcw > 0)).astype(np.int8)
        data['ac_positive_dc_zero'] = ((w > 0) & (np.abs(dcw) <= 1e-06)).astype(np.int8)
        data['ac_dc_same_sign'] = (np.sign(np.nan_to_num(w, nan=0.0)) == np.sign(np.nan_to_num(dcw, nan=0.0))).astype(np.int8)
        data['dca_over_total'] = self._d(dca, prt0_a + prt1_a)
        return rare_type.astype(np.int8)

    def bf(self, df):
        self._coerce_numeric(df)
        fingerprint = df[CS].fillna('<NA>').agg('|'.join, axis=1)
        data = {'Id': df['Id'].to_numpy(), 'device_fingerprint': fingerprint.to_numpy(dtype=object), 'device_family': np.where(fingerprint == CANON1, 'canon10', np.where(fingerprint == CANON2, 'canon100', 'other')), 'common_missing_any': df[CS].isna().any(axis=1).astype(np.int8).to_numpy(), 'common_missing_count': df[CS].isna().sum(axis=1).astype(np.int16).to_numpy(), 'common_sn_has_decimal_suffix': df['common[0].SN'].fillna('').astype(str).str.endswith('.0').astype(np.int8).to_numpy()}
        data['noncanonical'] = (data['device_family'] == 'other').astype(np.int8)
        for col in RN:
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            data[SR[col]] = arr
        for col in RSC:
            data[SS[col]] = df[col].fillna('<NA>').astype(str).to_numpy(dtype=object)
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
            data[name] = self._d(numerator, denominator)
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
        data['va_over_pqmag'] = self._d(va, pq)
        pf_from_w_va = self._d(w, va)
        data['pf_from_w_va'] = pf_from_w_va
        data['pf_error'] = (pf - pf_from_w_va).astype(np.float32)
        for name, total, suffixes in [('w_phase_sum_error', w, ['WL1', 'WL2', 'WL3']), ('va_phase_sum_error', va, ['VAL1', 'VAL2', 'VAL3']), ('var_phase_sum_error', var, ['VarL1', 'VarL2', 'VarL3'])]:
            phase_sum = sum((df[f'DERMeasureAC[0].{suffix}'].to_numpy(float) for suffix in suffixes))
            data[name] = (total - phase_sum).astype(np.float32)
        for name, suffixes in [('phase_ll_spread', ['VL1L2', 'VL2L3', 'VL3L1']), ('phase_ln_spread', ['VL1', 'VL2', 'VL3']), ('phase_w_spread', ['WL1', 'WL2', 'WL3']), ('phase_var_spread', ['VarL1', 'VarL2', 'VarL3'])]:
            phase_values = df[[f'DERMeasureAC[0].{suffix}' for suffix in suffixes]].to_numpy(float)
            data[name] = (self._x(phase_values) - self._n(phase_values)).astype(np.float32)
        for name, numerator, denominator in [('wmax_over_wmaxrtg', wmax, wmaxrtg), ('vamax_over_vamaxrtg', vamax, vamaxrtg), ('vmax_over_vnom', vmax, vnom), ('vmin_over_vnom', vmin, vnom)]:
            data[name] = self._d(numerator, denominator)
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
        voltage_pct = 100.0 * self._d(llv, vnom)
        line_neutral_voltage_pct = 100.0 * self._d(lnv * SQRT3, vnom)
        w_pct = 100.0 * self._d(w, wmaxrtg)
        var_pct = self._var_pct(var, varmaxinj, varmaxabs)
        data['voltage_pct'] = voltage_pct.astype(np.float32)
        data['line_neutral_voltage_pct'] = line_neutral_voltage_pct.astype(np.float32)
        data['w_pct_of_rtg'] = w_pct.astype(np.float32)
        data['var_pct_of_limit'] = var_pct.astype(np.float32)
        enter_state_anomaly, enter_blocked_power, enter_blocked_current = self._add_enter_service_features(data, df, voltage_pct=voltage_pct, hz=hz, abs_w=abs_w, va=va, a=a, tolw=tolw, tolva=tolva, amax=amax)
        pf_abs_ext_present, pf_abs_rvrt_ext_present = self._add_pf_control_features(data, df, pf=pf, var=var, varmaxinj=varmaxinj, varmaxabs=varmaxabs)
        trip_outside_flags = []
        trip_power_flags = []
        for sn, (prefix, axis_name, mode) in TS.items():
            mv = voltage_pct if axis_name == 'V' else hz
            outside, power_when_outside = self._add_trip_block_features(data, df, sn=sn, prefix=prefix, axis_name=axis_name, mode=mode, mv=mv, abs_w=abs_w, tolw=tolw)
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
        for name, prefix, point_count, x_field, y_field, meta_fields, mv, observed_value in curve_specs:
            self._add_curve_block_features(data, name=name, raw_idx=df[f'{prefix}.AdptCrvRslt'].to_numpy(float), curve_x=curve_points(prefix, x_field, point_count), curve_y=curve_points(prefix, y_field, point_count), curve_actpt=curve_scalars(prefix, 'ActPt'), curve_meta={meta_name: curve_scalars(prefix, field) for meta_name, field in meta_fields.items()}, mv=mv, observed_value=observed_value)
        self._add_freq_droop_features(data, df, hz=hz, w_pct=w_pct)
        dc_port_type_rare = self._add_dc_features(data, df, w=w, abs_w=abs_w)
        ac_type = df['DERMeasureAC[0].ACType'].to_numpy(float)
        ac_type_is_rare = np.isfinite(ac_type) & (ac_type == 3.0)
        data['ac_type_is_rare'] = ac_type_is_rare.astype(np.int8)
        flag_map = {'noncanonical': data['noncanonical'] == 1, 'common_missing': data['common_missing_any'] == 1, 'w_gt_wmax': data['w_gt_wmax_tol'] == 1, 'w_gt_wmaxrtg': data['w_gt_wmaxrtg_tol'] == 1, 'va_gt_vamax': data['va_gt_vamax_tol'] == 1, 'var_gt_injmax': data['var_gt_injmax_tol'] == 1, 'var_lt_absmax': data['var_lt_absmax_tol'] == 1, 'wset_far': data['wset_enabled_far'] == 1, 'wsetpct_far': data['wsetpct_enabled_far'] == 1, 'wmaxlim_far': data['wmaxlim_enabled_far'] == 1, 'varsetpct_far': data['varsetpct_enabled_far'] == 1, 'model_structure': data['model_structure_anomaly_any'] == 1, 'ac_type_rare': ac_type_is_rare == 1, 'dc_type_rare': dc_port_type_rare == 1, 'enter_state': enter_state_anomaly == 1, 'enter_blocked_power': enter_blocked_power == 1, 'enter_blocked_current': enter_blocked_current == 1, 'pf_abs': pf_abs_ext_present == 1, 'pf_abs_rvrt': pf_abs_rvrt_ext_present == 1, 'trip_power': trip_any_power_when_outside == 1}
        hard_rule_flags = np.column_stack([flag_map[name] for name in HARD_RULE_NAMES])
        hard_override_flags = np.column_stack([flag_map[name] for name in self.ovr])
        ff = {name: flag.astype(np.float32) for name, flag in flag_map.items()}
        data['hard_rule_count'] = hard_rule_flags.sum(axis=1).astype(np.int8)
        data['hard_rule_score'] = 3.0 * ff['noncanonical'] + 2.5 * ff['common_missing'] + 2.0 * (ff['w_gt_wmax'] + ff['w_gt_wmaxrtg'] + ff['va_gt_vamax'] + ff['var_gt_injmax'] + ff['var_lt_absmax'] + ff['model_structure'] + ff['enter_state'] + ff['trip_power']) + 1.5 * (ff['wset_far'] + ff['wsetpct_far'] + ff['ac_type_rare'] + ff['dc_type_rare'] + ff['pf_abs'] + ff['pf_abs_rvrt']) + 1.0 * ff['varsetpct_far'] + 0.75 * ff['wmaxlim_far'] + 0.35 * (ff['enter_blocked_power'] + ff['enter_blocked_current'])
        hard_rule_anomaly = hard_rule_flags.any(axis=1).astype(np.int8)
        data['hard_rule_anomaly'] = hard_rule_anomaly
        data['hard_override_anomaly'] = hard_override_flags.any(axis=1).astype(np.int8)
        return pd.DataFrame(data)

    def irc(self, member, usecols, limit_rows=0):
        yielded = 0
        for chunk in pd.read_csv(SCRIPT_DIR / member, usecols=list(usecols), chunksize=self.chunksize, low_memory=False):
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
        out['device_family'] = out['device_family'].map(FAM).fillna(-1).astype(np.int8)
        return out

    def _gsf(self, columns):
        excluded = {'Id', 'Label', 'fold_id', 'audit_fold_id', 'device_fingerprint', 'hard_rule_anomaly', 'hard_rule_count', 'hard_rule_score', 'hard_override_anomaly'}
        excluded.update(SS.values())
        return [col for col in columns if col not in excluded and col not in SLF]

    def _build_sample_weights(self, x_df, y):
        weights = np.ones(len(x_df), dtype=np.float32)
        family = x_df['device_family'].to_numpy()
        ho = x_df['hard_override_anomaly'].to_numpy() == 1
        weights[(family == 'canon100') & (y == 0)] *= CNW
        weights[ho] *= HARD_OVERRIDE_TRAIN_WEIGHT
        return weights

    @staticmethod
    def _b(values, *, fv, dtype, scale=1.0, round_values=True):
        out = pd.to_numeric(values, errors='coerce')
        if scale != 1.0:
            out = out / scale
        if round_values:
            out = out.round()
        return out.fillna(fv).astype(dtype)

    def _bsf(self, x_df, *, include_output_bins):
        frame = {'family': x_df['device_family'].astype(str), 'throt_src': self._b(x_df['DERMeasureAC_0_ThrotSrc'], fv=-1, dtype=np.int16), 'throt_pct': self._b(x_df['DERMeasureAC_0_ThrotPct'], scale=5.0, fv=-1, dtype=np.int16), 'wmaxlim_pct': self._b(x_df['DERCtlAC_0_WMaxLimPct'], scale=5.0, fv=-1, dtype=np.int16), 'wset_pct': self._b(x_df['DERCtlAC_0_WSetPct'], scale=5.0, fv=-1, dtype=np.int16), 'varset_pct': self._b(x_df['DERCtlAC_0_VarSetPct'], scale=5.0, fv=-1, dtype=np.int16), 'pf_set': self._b(x_df['DERCtlAC_0_PFWInj_PF'], scale=0.02, fv=-1, dtype=np.int16), 'fd_idx': self._b(x_df['DERFreqDroop_0_AdptCtlRslt'], fv=-1, dtype=np.int16), 'vv_idx': self._b(x_df['DERVoltVar_0_AdptCrvRslt'], fv=-1, dtype=np.int16), 'vw_idx': self._b(x_df['DERVoltWatt_0_AdptCrvRslt'], fv=-1, dtype=np.int16), 'wv_idx': self._b(x_df['DERWattVar_0_AdptCrvRslt'], fv=-1, dtype=np.int16), 'volt_bin': self._b(x_df['voltage_pct'], fv=-999, dtype=np.int16), 'hz_bin': self._b(x_df['DERMeasureAC_0_Hz'], scale=0.1, fv=-999, dtype=np.int16), 'enter_idle': self._b(x_df['enter_service_should_idle'], fv=0, dtype=np.int8, round_values=False), 'droop_active': self._b(x_df['freqdroop_outside_deadband'], fv=0, dtype=np.int8, round_values=False)}
        if include_output_bins:
            frame['w_bin'] = self._b(x_df['w_pct_of_rtg'], scale=5.0, fv=-999, dtype=np.int16)
            frame['var_bin'] = self._b(x_df['var_pct_of_limit'], scale=5.0, fv=-999, dtype=np.int16)
            frame['pf_mode'] = self._b(x_df['pf_control_any_enabled'], fv=0, dtype=np.int8, round_values=False)
        return pd.DataFrame(frame)

    @staticmethod
    def _hash_frame(frame):
        return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)

    def _bsk(self, x_df):
        return self._hash_frame(self._bsf(x_df, include_output_bins=False))

    def _bok(self, x_df):
        return self._hash_frame(self._bsf(x_df, include_output_bins=True))

    @staticmethod
    def _lss(keys, sum_map, count_map):
        key_series = pd.Series(keys)
        sv = key_series.map(sum_map).fillna(0.0).to_numpy(np.float32)
        cv = key_series.map(count_map).fillna(0).to_numpy(np.int32)
        return (sv, cv)

    @staticmethod
    def _asf(out, *, fp, scenario_rate, scenario_count, scenario_output_rate, scenario_output_count):
        out['scenario_rate'] = scenario_rate.astype(np.float32)
        out['scenario_rate_delta'] = (scenario_rate - fp).astype(np.float32)
        out['scenario_count'] = scenario_count.astype(np.int32)
        out['scenario_log_count'] = np.log1p(scenario_count).astype(np.float32)
        out['scenario_low_support'] = (scenario_count < 20).astype(np.int8)
        out['scenario_output_rate'] = scenario_output_rate.astype(np.float32)
        out['scenario_output_rate_delta'] = (scenario_output_rate - fp).astype(np.float32)
        out['scenario_output_count'] = scenario_output_count.astype(np.int32)
        out['scenario_output_log_count'] = np.log1p(scenario_output_count).astype(np.float32)
        out['scenario_output_low_support'] = (scenario_output_count < 20).astype(np.int8)
        return out

    def _fts(self, x_train, y_train):
        out = x_train.copy()
        y_arr = y_train.to_numpy(np.float32)
        fs = out['device_family'].astype(str)
        self.family_base_rates = pd.DataFrame({'family': fs, 'y': y_arr}).groupby('family')['y'].mean().to_dict()
        keys = self._bsk(out)
        ok = self._bok(out)
        fold_ids = (out['Id'].to_numpy(np.int64) % self.cv_folds).astype(np.int8)
        scenario_rate = np.zeros(len(out), dtype=np.float32)
        scenario_count = np.zeros(len(out), dtype=np.int32)
        scenario_output_rate = np.zeros(len(out), dtype=np.float32)
        scenario_output_count = np.zeros(len(out), dtype=np.int32)
        gr = float(np.mean(y_arr))
        for fold in range(self.cv_folds):
            tm = fold_ids != fold
            vm = fold_ids == fold
            if not vm.any():
                continue
            stats = pd.DataFrame({'key': keys[tm], 'y': y_arr[tm]}).groupby('key')['y'].agg(['sum', 'count'])
            output_stats = pd.DataFrame({'key': ok[tm], 'y': y_arr[tm]}).groupby('key')['y'].agg(['sum', 'count'])
            valid_keys = pd.Series(keys[vm])
            valid_sum = valid_keys.map(stats['sum']).fillna(0.0).to_numpy(np.float32)
            valid_count = valid_keys.map(stats['count']).fillna(0).to_numpy(np.int32)
            valid_output_keys = pd.Series(ok[vm])
            valid_output_sum = valid_output_keys.map(output_stats['sum']).fillna(0.0).to_numpy(np.float32)
            valid_output_count = valid_output_keys.map(output_stats['count']).fillna(0).to_numpy(np.int32)
            valid_family = fs.loc[vm].tolist()
            prior = np.array([self.family_base_rates.get(name, gr) for name in valid_family], dtype=np.float32)
            scenario_rate[vm] = (valid_sum + SM * prior) / (valid_count + SM)
            scenario_count[vm] = valid_count
            scenario_output_rate[vm] = (valid_output_sum + SM * prior) / (valid_output_count + SM)
            scenario_output_count[vm] = valid_output_count
        full_stats = pd.DataFrame({'key': keys, 'y': y_arr}).groupby('key')['y'].agg(['sum', 'count'])
        full_output_stats = pd.DataFrame({'key': ok, 'y': y_arr}).groupby('key')['y'].agg(['sum', 'count'])
        self.ssm = {int(idx): float(val) for idx, val in full_stats['sum'].items()}
        self.scm = {int(idx): int(val) for idx, val in full_stats['count'].items()}
        self.sosm = {int(idx): float(val) for idx, val in full_output_stats['sum'].items()}
        self.socm = {int(idx): int(val) for idx, val in full_output_stats['count'].items()}
        fp = fs.map(self.family_base_rates).fillna(gr).to_numpy(np.float32)
        return self._asf(out, fp=fp, scenario_rate=scenario_rate, scenario_count=scenario_count, scenario_output_rate=scenario_output_rate, scenario_output_count=scenario_output_count)

    def _apf(self, x_df):
        if not self.scm:
            return x_df
        out = x_df.copy()
        keys = self._bsk(out)
        ok = self._bok(out)
        sv, cv = self._lss(keys, self.ssm, self.scm)
        output_sum_values, output_count_values = self._lss(ok, self.sosm, self.socm)
        gr = float(np.mean(list(self.family_base_rates.values()))) if self.family_base_rates else 0.5
        fp = out['device_family'].astype(str).map(self.family_base_rates).fillna(gr).to_numpy(np.float32)
        scenario_rate = (sv + SM * fp) / (cv + SM)
        scenario_output_rate = (output_sum_values + SM * fp) / (output_count_values + SM)
        return self._asf(out, fp=fp, scenario_rate=scenario_rate, scenario_count=cv, scenario_output_rate=scenario_output_rate, scenario_output_count=output_count_values)

    def _afi(self, x_df):
        out = x_df.copy()
        canon100_mask = out['device_family'].astype(str) == 'canon100'
        for feature_name in CIF:
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
        return G(n_estimators=max(80, self.n_estimators // 2), max_depth=max(4, self.max_depth - 2), learning_rate=min(0.08, self.learning_rate * 1.2), objective='reg:squarederror', **self._xgb_shared_params(eval_metric='rmse', verbosity=0))

    def _new_classifier(self):
        return X(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, objective='binary:logistic', **self._xgb_shared_params(eval_metric='logloss', verbosity=1))

    def _fsm(self, x_train, y_train, vm):
        self.sur_cols = self._gsf(x_train.columns)
        fit_partition = self._surrogate_partition_mask(x_train['Id'], fit_partition=True)
        normal_mask = (y_train == 0) & (x_train['hard_override_anomaly'] == 0) & (x_train['device_family'] != 'other') & ~vm.to_numpy() & fit_partition
        surrogate_df = x_train.loc[normal_mask].copy()
        if surrogate_df.empty:
            raise RuntimeError('No rows available to train surrogate models.')
        self.surrogate_models = {}
        for family in FAM:
            fd = surrogate_df.loc[surrogate_df['device_family'] == family].copy()
            if fd.empty:
                continue
            x_surrogate = self._encode_device_family(fd[self.sur_cols])
            for tg, (target_col, _) in STG.items():
                model = self._new_surrogate_model()
                y_target = fd[target_col].to_numpy(np.float32)
                print(f'[surrogate] training {family}/{tg} on {len(fd):,} normal rows')
                model.fit(x_surrogate, y_target)
                self.surrogate_models[family, tg] = model

    def _aws(self, x_df):
        if self.sur_cols is None or not self.surrogate_models:
            return x_df
        out = x_df.copy()
        for tg in STG:
            out[f'pred_{tg}'] = np.nan
            out[f'resid_{tg}'] = np.nan
            out[f'abs_resid_{tg}'] = np.nan
            out[f'norm_resid_{tg}'] = np.nan
            out[f'abs_norm_resid_{tg}'] = np.nan
            out[f'tail_resid_{tg}'] = 0
            out[f'extreme_resid_{tg}'] = 0
            out[f'ultra_resid_{tg}'] = 0
            out[f'q99_ratio_resid_{tg}'] = np.nan
        x_surrogate = self._encode_device_family(out[self.sur_cols])
        for family in FAM:
            fm = out['device_family'] == family
            if not fm.any():
                continue
            x_family = x_surrogate.loc[fm]
            for tg, (target_col, scale_col) in STG.items():
                model = self.surrogate_models.get((family, tg))
                if model is None:
                    continue
                pred = model.predict(x_family).astype(np.float32)
                actual = out.loc[fm, target_col].to_numpy(np.float32)
                resid = actual - pred
                out.loc[fm, f'pred_{tg}'] = pred
                out.loc[fm, f'resid_{tg}'] = resid
                out.loc[fm, f'abs_resid_{tg}'] = np.abs(resid).astype(np.float32)
                if scale_col is not None:
                    scale = out.loc[fm, scale_col].to_numpy(np.float32)
                    norm_resid = self._d(resid, scale)
                else:
                    scale = np.maximum(0.05, np.abs(actual))
                    norm_resid = (resid / scale).astype(np.float32)
                out.loc[fm, f'norm_resid_{tg}'] = norm_resid.astype(np.float32)
                out.loc[fm, f'abs_norm_resid_{tg}'] = np.abs(norm_resid).astype(np.float32)
        out['resid_energy_total'] = out[['abs_resid_w', 'abs_resid_va', 'abs_resid_var', 'abs_resid_pf', 'abs_resid_a']].sum(axis=1).astype(np.float32)
        out['resid_va_minus_pq'] = (out['pred_va'] - np.sqrt(np.square(out['pred_w']) + np.square(out['pred_var']))).astype(np.float32)
        out['resid_w_var_ratio'] = self._d(out['abs_resid_w'].to_numpy(float), out['abs_resid_var'].to_numpy(float) + 0.001)
        return out

    def _crq(self, x_train, y_train, vm):
        calibration_partition = self._surrogate_partition_mask(x_train['Id'], fit_partition=False)
        base_mask = (y_train == 0) & (x_train['hard_override_anomaly'] == 0) & (x_train['device_family'] != 'other') & ~vm.to_numpy()
        self.res_q = {}
        for family in FAM:
            fm = base_mask & (x_train['device_family'] == family)
            family_calibration = fm & calibration_partition
            if not family_calibration.any():
                family_calibration = fm
            family_quantiles = {}
            for tg in STG:
                series = x_train.loc[family_calibration, f'abs_norm_resid_{tg}']
                values = pd.to_numeric(series, errors='coerce').to_numpy(np.float32)
                values = values[np.isfinite(values)]
                quantiles = RTF.copy()
                if values.size > 0:
                    for level_name, q in RTL.items():
                        quantiles[level_name] = float(np.quantile(values, q))
                family_quantiles[tg] = {key: max(1e-06, value) for key, value in quantiles.items()}
            self.res_q[family] = family_quantiles

    def _arf(self, x_df):
        if not self.res_q:
            return x_df
        out = x_df.copy()
        for tg in STG:
            out[f'tail_resid_{tg}'] = 0
            out[f'extreme_resid_{tg}'] = 0
            out[f'ultra_resid_{tg}'] = 0
            out[f'q99_ratio_resid_{tg}'] = np.nan
        for family in FAM:
            fm = out['device_family'] == family
            if not fm.any():
                continue
            family_quantiles = self.res_q.get(family, {})
            for tg in STG:
                abs_norm = out.loc[fm, f'abs_norm_resid_{tg}'].to_numpy(np.float32)
                q = family_quantiles.get(tg, RTF)
                tail = abs_norm >= q['tail']
                extreme = abs_norm >= q['extreme']
                ultra = abs_norm >= q['ultra']
                q99_ratio = self._d(abs_norm, np.full_like(abs_norm, q['extreme'], dtype=np.float32))
                out.loc[fm, f'tail_resid_{tg}'] = tail.astype(np.int8)
                out.loc[fm, f'extreme_resid_{tg}'] = extreme.astype(np.int8)
                out.loc[fm, f'ultra_resid_{tg}'] = ultra.astype(np.int8)
                out.loc[fm, f'q99_ratio_resid_{tg}'] = q99_ratio.astype(np.float32)
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
    def tune_threshold(y_true, prob, *, low=FTF, high=MT, step=0.01):
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
    def _blend_probs(primary, secondary, weight):
        if secondary is None:
            return primary.astype(np.float32)
        return (weight * primary + (1.0 - weight) * secondary).astype(np.float32)

    @staticmethod
    def _select_nonconstant_columns(df, candidates):
        keep = []
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

    def _ensure_catboost_available(self):
        if C is None:
            raise RuntimeError('CatBoost is required for the full-data hybrid pipeline. Install dependencies from pyproject.toml before training.')

    def _new_cat_model(self):
        self._ensure_catboost_available()
        return C(iterations=self.cat_iterations, depth=self.cat_depth, learning_rate=self.cat_learning_rate, loss_function='Logloss', eval_metric='Logloss', random_seed=self.seed, thread_count=self.n_jobs, allow_writing_files=False, verbose=False)

    def _build_train_artifacts(self):
        shutil.rmtree(self.artifact_dir, ignore_errors=True)
        train_root = self.artifact_dir / 'train'
        built = 0
        for ci, chunk in enumerate(self.irc('train.csv', UTR)):
            labels = chunk['Label'].astype(np.int8).to_numpy()
            feats = self.bf(chunk.drop(columns=['Label']))
            feats['Label'] = labels
            feats['fold_id'] = (feats['Id'].to_numpy(np.int64) % self.cv_folds).astype(np.int8)
            feats['audit_fold_id'] = (self._bsk(feats) % self.cv_folds).astype(np.int8)
            for family in ('canon10', 'canon100', 'other'):
                fm = feats['device_family'] == family
                if fm.any():
                    family_dir = train_root / family
                    family_dir.mkdir(parents=True, exist_ok=True)
                    fd = feats.loc[fm].copy()
                    fd.to_parquet(family_dir / f'{ci:05d}.parquet', index=False)
                    built += len(fd)
            if ci % 10 == 0:
                print(f'[artifacts] materialized {built:,} training rows')
            del feats
            gc.collect()

    def _load_family_artifact(self, family, columns=None):
        family_dir = self.artifact_dir / 'train' / family
        paths = sorted(family_dir.glob('*.parquet'))
        if not paths:
            return pd.DataFrame(columns=list(columns or []))
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        return pd.concat(frames, ignore_index=True)

    def _refresh_override_columns(self, df):
        out = df.copy()
        hard_rule_flags = np.column_stack([pd.to_numeric(out[RCM[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1 for name in HARD_RULE_NAMES])
        if self.ovr:
            hard_override_flags = np.column_stack([pd.to_numeric(out[RCM[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1 for name in self.ovr])
            out['hard_override_anomaly'] = hard_override_flags.any(axis=1).astype(np.int8)
        else:
            out['hard_override_anomaly'] = np.zeros(len(out), dtype=np.int8)
        out['hard_rule_anomaly'] = hard_rule_flags.any(axis=1).astype(np.int8)
        return out

    def _audit_hard_override_rules(self):
        cols = sorted({RCM[name] for name in OVR})
        counts = {name: [0, 0] for name in OVR}
        for family in ['canon10', 'canon100', 'other']:
            frame = self._load_family_artifact(family, columns=['Label', *cols])
            if frame.empty:
                continue
            labels = frame['Label'].to_numpy(np.int8)
            for name in OVR:
                mask = pd.to_numeric(frame[RCM[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1
                if mask.any():
                    counts[name][0] += int(mask.sum())
                    counts[name][1] += int(labels[mask].sum())
        self.ovr = [name for name, (count, positives) in counts.items() if count == 0 or positives / count >= MOP]

    def _capture_semantic_context(self):
        return (self.sur_cols, self.surrogate_models, self.res_q, self.family_base_rates, self.ssm, self.scm, self.sosm, self.socm)

    def _activate_semantic_context(self, cx):
        self.sur_cols, self.surrogate_models, self.res_q, self.family_base_rates, self.ssm, self.scm, self.sosm, self.socm = cx

    def _psf(self, bdf, y):
        work = self._refresh_override_columns(bdf)
        no_valid = pd.Series(np.zeros(len(work), dtype=bool), index=work.index)
        self._fsm(work, y, no_valid)
        work = self._aws(work)
        self._crq(work, y, no_valid)
        work = self._arf(work)
        work = self._fts(work, y)
        work = self._afi(work)
        return (work, self._capture_semantic_context())

    def _semantic_feature_candidates(self, sdf):
        excluded = {'Id', 'Label', 'fold_id', 'audit_fold_id', 'hard_override_anomaly', 'device_fingerprint'}
        excluded.update(SS.values())
        return [col for col in sdf.columns if col not in excluded and pd.api.types.is_numeric_dtype(sdf[col])]

    def _prepare_cat_frame(self, bdf):
        out = self._refresh_override_columns(bdf)
        for col in [*SS.values(), 'device_fingerprint']:
            if col in out.columns:
                out[col] = out[col].fillna('<NA>').astype(str)
        return out

    def _cat_feature_candidates(self, cat_df):
        raw_numeric_cols = [SR[col] for col in RN if SR[col] in cat_df.columns]
        missing_cols = [col for col in cat_df.columns if col.startswith('missing_')]
        cc = [SS[col] for col in RSC if SS[col] in cat_df.columns]
        candidates = dedupe([*raw_numeric_cols, *cc, 'device_fingerprint', *CEC, *missing_cols])
        excluded = {'Id', 'Label', 'fold_id', 'audit_fold_id', 'hard_override_anomaly', 'hard_rule_anomaly'}
        return [col for col in candidates if col in cat_df.columns and col not in excluded]

    def _tso(self, sdf, y, fc, *, fold_col, fit_final):
        probs = np.ones(len(sdf), dtype=np.float32)
        mm = sdf['hard_override_anomaly'].to_numpy(np.int8) == 0
        fold_ids = sdf[fold_col].to_numpy(np.int8)
        final_model = None
        for fold in range(self.cv_folds):
            tm = mm & (fold_ids != fold)
            vm = mm & (fold_ids == fold)
            if not vm.any():
                continue
            model = self._new_classifier()
            x_train = sdf.loc[tm, fc]
            y_train = y[tm]
            weights = self._build_sample_weights(sdf.loc[tm], y_train)
            model.fit(x_train, y_train, sample_weight=weights)
            probs[vm] = model.predict_proba(sdf.loc[vm, fc])[:, 1].astype(np.float32)
        if fit_final and mm.any():
            final_model = self._new_classifier()
            weights = self._build_sample_weights(sdf.loc[mm], y[mm])
            final_model.fit(sdf.loc[mm, fc], y[mm], sample_weight=weights)
        return (probs, final_model)

    def _tco(self, cat_df, y, fc, cc, *, fold_col, fit_final):
        probs = np.ones(len(cat_df), dtype=np.float32)
        mm = cat_df['hard_override_anomaly'].to_numpy(np.int8) == 0
        fold_ids = cat_df[fold_col].to_numpy(np.int8)
        final_model = None
        for fold in range(self.cv_folds):
            tm = mm & (fold_ids != fold)
            vm = mm & (fold_ids == fold)
            if not vm.any():
                continue
            model = self._new_cat_model()
            weights = self._build_sample_weights(cat_df.loc[tm], y[tm])
            model.fit(cat_df.loc[tm, fc], y[tm], cat_features=list(cc), sample_weight=weights, verbose=False)
            probs[vm] = model.predict_proba(cat_df.loc[vm, fc])[:, 1].astype(np.float32)
        if fit_final and mm.any():
            final_model = self._new_cat_model()
            weights = self._build_sample_weights(cat_df.loc[mm], y[mm])
            final_model.fit(cat_df.loc[mm, fc], y[mm], cat_features=list(cc), sample_weight=weights, verbose=False)
        return (probs, final_model)

    def _sfb(self, y, ho, semantic_primary, semantic_audit, cat_primary, cat_audit):
        baseline_primary_prob = semantic_primary.copy()
        baseline_primary_prob[ho] = 1.0
        baseline_thr, _ = self.tune_threshold(y, baseline_primary_prob)
        baseline_pred_primary = (baseline_primary_prob >= baseline_thr).astype(np.int8)
        baseline_audit_prob = semantic_audit.copy()
        baseline_audit_prob[ho] = 1.0
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
            blended_primary[ho] = 1.0
            thr, _ = self.tune_threshold(y, blended_primary)
            pred_primary = (blended_primary >= thr).astype(np.int8)
            blended_audit = self._blend_probs(semantic_audit, cat_audit, weight)
            blended_audit[ho] = 1.0
            pred_audit = (blended_audit >= thr).astype(np.int8)
            ps = float(fbeta_score(y, pred_primary, beta=2))
            as_ = float(fbeta_score(y, pred_audit, beta=2))
            if as_ < baseline_audit_score - AT:
                continue
            if ps > best_primary_score + 1e-12 or (abs(ps - best_primary_score) <= 1e-12 and as_ > best_audit_for_best):
                best_weight = weight
                best_thr = thr
                best_primary_pred = pred_primary
                best_audit_pred = pred_audit
                best_primary_score = ps
                best_audit_for_best = as_
        return (best_weight, best_thr, best_primary_pred.astype(np.int8), best_audit_pred.astype(np.int8))

    def fit(self):
        self._build_train_artifacts()
        self._audit_hard_override_rules()
        trained = 0
        for family in FAM:
            bdf = self._load_family_artifact(family)
            if bdf.empty:
                continue
            y_series = bdf['Label'].astype(np.int8)
            sdf, cx = self._psf(bdf.copy(), y_series)
            self.ctx[family] = cx
            semf = self._select_nonconstant_columns(sdf, self._semantic_feature_candidates(sdf))
            self.sem_cols[family] = semf
            cat_df = self._prepare_cat_frame(bdf.copy())
            catf = self._select_nonconstant_columns(cat_df, self._cat_feature_candidates(cat_df))
            self.cat_cols[family] = catf
            catc = [col for col in [*SS.values(), 'device_fingerprint'] if col in catf]
            y = y_series.to_numpy(np.int8)
            ho = sdf['hard_override_anomaly'].to_numpy(np.int8) == 1
            sp1, sm = self._tso(sdf, y, semf, fold_col='fold_id', fit_final=True)
            sa1, _ = self._tso(sdf, y, semf, fold_col='audit_fold_id', fit_final=False)
            self.semantic_models[family] = sm
            cp1 = ca1 = cm = None
            if catf:
                cp1, cm = self._tco(cat_df, y, catf, catc, fold_col='fold_id', fit_final=True)
                ca1, _ = self._tco(cat_df, y, catf, catc, fold_col='audit_fold_id', fit_final=False)
            self.cat_models[family] = cm
            weight, threshold, _, _ = self._sfb(y, ho, sp1, sa1, cp1, ca1)
            self.blend_w[family] = weight
            self.thr[family] = threshold
            trained += 1
            print(f'[fit] {family} threshold={threshold:.3f}, blend_weight={weight:.2f}')
            del bdf, sdf, cat_df
            gc.collect()
        if not trained:
            raise RuntimeError('No training artifacts were available for model fitting.')

    def _predict_family_chunk(self, family, bdf):
        if family not in self.ctx or family not in self.semantic_models:
            raise RuntimeError(f'Missing fitted semantic model bundle for family {family}.')
        cx = self.ctx[family]
        self._activate_semantic_context(cx)
        work = self._refresh_override_columns(bdf.copy())
        ho = work['hard_override_anomaly'].to_numpy(np.int8) == 1
        sdf = self._aws(work.copy())
        sdf = self._arf(sdf)
        sdf = self._apf(sdf)
        sdf = self._afi(sdf)
        sp = np.ones(len(sdf), dtype=np.float32)
        if (~ho).any():
            sm = self.semantic_models[family]
            sp[~ho] = sm.predict_proba(sdf.loc[~ho, self.sem_cols[family]])[:, 1].astype(np.float32)
        cat_prob = None
        cm = self.cat_models.get(family)
        if cm is not None and self.cat_cols.get(family):
            cat_df = self._prepare_cat_frame(bdf.copy())
            cat_prob = np.ones(len(cat_df), dtype=np.float32)
            if (~ho).any():
                cat_prob[~ho] = cm.predict_proba(cat_df.loc[~ho, self.cat_cols[family]])[:, 1].astype(np.float32)
        bp = self._blend_probs(sp, cat_prob, self.blend_w.get(family, 1.0))
        bp[ho] = 1.0
        pred = (bp >= self.thr.get(family, 0.5)).astype(np.int8)
        pred[ho] = 1
        return pred

    def pt(self, out_csv):
        if not self.semantic_models:
            raise RuntimeError('Model is not fitted.')
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tr = 0
        pr = 0
        with out_csv.open('w', encoding='utf-8') as fh:
            fh.write('Id,Label\n')
            for ci, chunk in enumerate(self.irc('test.csv', UTE)):
                feats = self.bf(chunk)
                pred = feats['hard_override_anomaly'].astype(np.int8).to_numpy()
                for family in FAM:
                    fm = feats['device_family'] == family
                    if fm.any():
                        pred[np.flatnonzero(fm.to_numpy())] = self._predict_family_chunk(family, feats.loc[fm].copy())
                out = pd.DataFrame({'Id': feats['Id'].astype(np.int64), 'Label': pred.astype(np.int8)})
                out.to_csv(fh, index=False, header=False)
                tr += len(out)
                pr += int(out['Label'].sum())
                if ci % 10 == 0:
                    print(f'[test] wrote {tr:,} predictions')
        print(f'[test] done; tr={tr:,}, pr={pr:,}, positive_rate={pr / max(tr, 1):.6f}')

def rp():
    s(DEFAULT_SEED)
    baseline = R()
    baseline.fit()
    out = DEFAULT_OUTPUT_DIR / 'submission_full_data.csv'
    baseline.pt(out)
    print(f'[solution] path={out}')

def main():
    rp()
if __name__ == '__main__':
    main()
