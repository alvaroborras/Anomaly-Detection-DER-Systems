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
        tp = f'{prefix}.Ctl[{ctl}]'
        cols.extend([f'{tp}.DbOf', f'{tp}.DbUf', f'{tp}.KOf', f'{tp}.KUf', f'{tp}.RspTms', f'{tp}.PMin', f'{tp}.ReadOnly'])
    return cols

def btc(prefix, an):
    cols = [f'{prefix}.ID', f'{prefix}.L', f'{prefix}.Ena', f'{prefix}.AdptCrvReq', f'{prefix}.AdptCrvRslt', f'{prefix}.NPt', f'{prefix}.NCrvSet']
    for curve in range(2):
        cp = f'{prefix}.Crv[{curve}]'
        cols.append(f'{cp}.ReadOnly')
        for group in ['MustTrip', 'MayTrip', 'MomCess']:
            gpr = f'{cp}.{group}'
            cols.append(f'{gpr}.ActPt')
            for point in range(5):
                cols.extend([f'{gpr}.Pt[{point}].{an}', f'{gpr}.Pt[{point}].Tms'])
    return cols
COF = 'Mn Md Opt Vr SN'.split()
CS = p('common[0]', COF)
COC = p('common[0]', ['ID', 'L', *COF, 'DA'])
MAF = '\nID L ACType W VA Var PF A LLV LNV Hz TmpAmb TmpCab TmpSnk TmpTrns TmpSw TmpOt\nThrotPct ThrotSrc WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3\nAL1 AL2 AL3 VL1L2 VL2L3 VL3L1 VL1 VL2 VL3\n'.split()
MAC = p('DERMeasureAC[0]', MAF)
CPF = '\nID L WMaxRtg VAMaxRtg VarMaxInjRtg VarMaxAbsRtg WChaRteMaxRtg WDisChaRteMaxRtg\nVAChaRteMaxRtg VADisChaRteMaxRtg VNomRtg VMaxRtg VMinRtg AMaxRtg PFOvrExtRtg\nPFUndExtRtg NorOpCatRtg AbnOpCatRtg IntIslandCatRtg WMax WMaxOvrExt WOvrExtPF\nWMaxUndExt WUndExtPF VAMax VarMaxInj VarMaxAbs WChaRteMax WDisChaRteMax\nVAChaRteMax VADisChaRteMax VNom VMax VMin AMax PFOvrExt PFUndExt CtrlModes\nIntIslandCat\n'.split()
CAC = p('DERCapacity[0]', CPF)
ESF = 'ID L ES ESVHi ESVLo ESHzHi ESHzLo ESDlyTms ESRndTms ESRmpTms ESDlyRemTms'.split()
ESC = p('DEREnterService[0]', ESF)
CTF = '\nID L PFWInjEna PFWInjEnaRvrt PFWInjRvrtTms PFWInjRvrtRem PFWAbsEna PFWAbsEnaRvrt\nPFWAbsRvrtTms PFWAbsRvrtRem WMaxLimPctEna WMaxLimPct WMaxLimPctRvrt\nWMaxLimPctEnaRvrt WMaxLimPctRvrtTms WMaxLimPctRvrtRem WSetEna WSetMod WSet\nWSetRvrt WSetPct WSetPctRvrt WSetEnaRvrt WSetRvrtTms WSetRvrtRem VarSetEna\nVarSetMod VarSetPri VarSet VarSetRvrt VarSetPct VarSetPctRvrt VarSetEnaRvrt\nVarSetRvrtTms VarSetRvrtRem WRmp WRmpRef VarRmp AntiIslEna PFWInj.PF PFWInj.Ext\nPFWInjRvrt.PF PFWInjRvrt.Ext PFWAbs.Ext PFWAbsRvrt.Ext\n'.split()
TAC = p('DERCtlAC[0]', CTF)
VVC = bvv('DERVoltVar[0]')
VWC = bvw('DERVoltWatt[0]')
FDC = bfd('DERFreqDroop[0]')
WVC = bwv('DERWattVar[0]')
TS = {'lv': ('DERTripLV[0]', 'V', 'low'), 'hv': ('DERTripHV[0]', 'V', 'high'), 'lf': ('DERTripLF[0]', 'Hz', 'low'), 'hf': ('DERTripHF[0]', 'Hz', 'high')}
TRIP_COLUMNS = {sn: btc(prefix, an) for sn, (prefix, an, _) in TS.items()}
MDF = '\nID L NPrt DCA DCW Prt[0].PrtTyp Prt[0].ID Prt[0].DCA Prt[0].DCV Prt[0].DCW\nPrt[0].Tmp Prt[1].PrtTyp Prt[1].ID Prt[1].DCA Prt[1].DCV Prt[1].DCW Prt[1].Tmp\n'.split()
MDC = p('DERMeasureDC[0]', MDF)
BSC = {'common': COC, 'ma0': MAC, 'capacity': CAC, 'enter_service': ESC, 'ctl_ac': TAC, 'volt_var': VVC, 'volt_watt': VWC, 'freq_droop': FDC, 'watt_var': WVC, 'md0': MDC}
for sn, cols in TRIP_COLUMNS.items():
    BSC[f'trip_{sn}'] = cols
CBM = 'Ena AdptCrvReq AdptCrvRslt NPt NCrv RvrtTms RvrtRem RvrtCrv'.split()
FDM = 'Ena AdptCtlReq AdptCtlRslt NCtl RvrtTms RvrtRem RvrtCtl'.split()
TMF = 'Ena AdptCrvReq AdptCrvRslt NPt NCrvSet'.split()
RN = dedupe(['common[0].DA', *p('DERMeasureAC[0]', MAF[2:]), *p('DERCapacity[0]', CPF[2:]), *p('DEREnterService[0]', ESF[2:]), *p('DERCtlAC[0]', CTF[2:]), *p('DERVoltVar[0]', CBM), *p('DERVoltWatt[0]', CBM), *p('DERFreqDroop[0]', FDM), *p('DERWattVar[0]', CBM), *p('DERMeasureDC[0]', MDF[2:])])
TMC = [f'{prefix}.{field}' for prefix, _, _ in TS.values() for field in TMF]
RXN = ['DERMeasureAC[0].A_SF', 'DERMeasureAC[0].V_SF', 'DERMeasureAC[0].Hz_SF', 'DERMeasureAC[0].W_SF', 'DERMeasureAC[0].PF_SF', 'DERMeasureAC[0].VA_SF', 'DERMeasureAC[0].Var_SF', 'DERCapacity[0].WOvrExtRtg', 'DERCapacity[0].WOvrExtRtgPF', 'DERCapacity[0].WUndExtRtg', 'DERCapacity[0].WUndExtRtgPF', 'DERCapacity[0].W_SF', 'DERCapacity[0].PF_SF', 'DERCapacity[0].VA_SF', 'DERCapacity[0].Var_SF', 'DERCapacity[0].V_SF', 'DERCapacity[0].A_SF', 'DERCtlAC[0].WSet_SF', 'DERMeasureDC[0].DCA_SF', 'DERMeasureDC[0].DCW_SF']
RXS = ['DERMeasureDC[0].Prt[0].IDStr', 'DERMeasureDC[0].Prt[1].IDStr']
RN = dedupe([*RN, *TMC, *RXN])
RSC = dedupe([*CS, *RXS])
TSC = [col for cols in TRIP_COLUMNS.values() for col in cols]
ASC = dedupe([*COC, *MAC, *CAC, *ESC, *TAC, *VVC, *VWC, *FDC, *WVC, *MDC, *TSC, *RXN, *RXS])
NSC = [c for c in ASC if c not in RSC]
UTR = dedupe(['Id', 'Label', *ASC])
UTE = dedupe(['Id', *ASC])
CANON1 = 'DERSec|DER Simulator|10 kW DER|1.2.3|SN-Three-Phase'
CANON2 = 'DERSec|DER Simulator 100 kW|1.2.3.1|1.0.0|1100058974'
SR = {c: re.sub('[^0-9A-Za-z_]+', '_', c) for c in RN}
SS = {c: re.sub('[^0-9A-Za-z_]+', '_', c) for c in RSC}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 42
DOD = SCRIPT_DIR / 'outputs' / 'full_data_hybrid'
SQRT3 = math.sqrt(3.0)
FAM = {'c0': 0, 'c1': 1}
RTL = {'tail': 0.95, 'extreme': 0.99, 'ultra': 0.999}
RTF = {'tail': 0.05, 'extreme': 0.1, 'ultra': 0.2}
FTF = 0.02
MT = 0.6
CNW = 1.5
HOTW = 0.35
SM = 50.0
AT = 0.003
MOP = 0.995
CIF = ['rs', 'sr', 'so', 'rqs', 'mdwr']
STG = {'w': ('DERMeasureAC_0_W', 'DERCapacity_0_WMaxRtg'), 'va': ('DERMeasureAC_0_VA', 'DERCapacity_0_VAMaxRtg'), 'var': ('DERMeasureAC_0_Var', 'DERCapacity_0_VarMaxInjRtg'), 'pf': ('DERMeasureAC_0_PF', None), 'a': ('DERMeasureAC_0_A', 'DERCapacity_0_AMaxRtg')}
SLF = {*(f'DERMeasureAC_0_{field}' for field in '\n    W VA Var PF A WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3\n    AL1 AL2 AL3\n    '.split()), *'\n    wwr ww vav vvr voi\n    voa aoa wmx wmr vmv\n    vmi0 vpa wew w_eq_wmax vev\n    vena psm gw gwr\n    gva gvi vla vmp\n    vop pfv pf_error wpe ape\n    vpe pws pvs wsae\n    wspt wspe wltg wlex\n    vsae vspt vspe wf\n    wef wmf vef wpr\n    vpl ebp esbv\n    ebc pte pre2\n    prl trip_lv_pwo trip_hv_pwo\n    trip_lf_pwo trip_hf_pwo\n    tpo voltvar_cer voltwatt_cer\n    wattvar_ce wattvar_cer fdwp\n    dcw_over_w dwwa azdp apdz\n    adss\n    '.split()}
HRN = ['nc', 'cmi', 'gww', 'gwrt', 'vgv', 'vgi', 'vlm', 'wsf', 'wpf', 'wlf', 'vpf', 'ms0', 'atr2', 'dtr2', 'es0', 'eip', 'eic', 'pa0', 'pr0', 'tp0']
OVR = ['nc', 'cmi', 'gww', 'gwrt', 'vgv', 'vgi', 'vlm', 'wsf', 'wpf', 'ms0', 'atr2', 'dtr2', 'es0', 'pa0', 'pr0', 'tp0']
RCM = {'nc': 'nc', 'cmi': 'cma', 'gww': 'gw', 'gwrt': 'gwr', 'vgv': 'gva', 'vgi': 'gvi', 'vlm': 'vla', 'wsf': 'wf', 'wpf': 'wef', 'wlf': 'wmf', 'vpf': 'vef', 'ms0': 'msa', 'atr2': 'atr', 'dtr2': 'dtr', 'es0': 'esa', 'eip': 'ebp', 'eic': 'ebc', 'pa0': 'pae', 'pr0': 'pre', 'tp0': 'tpo'}
CEC = ['dg', 'cmp', 'emp', 'mst', 'msb', 'cma', 'cmc', 'csd']
EMM = {'common': ('common[0].ID', 'common[0].L', 1.0, 66.0), 'ma0': ('DERMeasureAC[0].ID', 'DERMeasureAC[0].L', 701.0, 153.0), 'capacity': ('DERCapacity[0].ID', 'DERCapacity[0].L', 702.0, 50.0), 'enter_service': ('DEREnterService[0].ID', 'DEREnterService[0].L', 703.0, 17.0), 'md0': ('DERMeasureDC[0].ID', 'DERMeasureDC[0].L', 714.0, 68.0)}

def s(seed):
    random.seed(seed)
    np.random.seed(seed)

def _d(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    out = np.full_like(a, np.nan)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-06)
    out[mask] = a[mask] / b[mask]
    return out

def _n(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    if arr.shape[1] == 0:
        return out
    reduced = np.where(mask, arr, np.inf).min(axis=1)
    vr = mask.any(axis=1)
    out[vr] = reduced[vr]
    return out

def _x(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    if arr.shape[1] == 0:
        return out
    reduced = np.where(mask, arr, -np.inf).max(axis=1)
    vr = mask.any(axis=1)
    out[vr] = reduced[vr]
    return out

def _nmr(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    counts = mask.sum(axis=1)
    vr = counts > 0
    if vr.any():
        totals = np.where(mask, arr, 0.0).sum(axis=1)
        out[vr] = totals[vr] / counts[vr]
    return out

def _ci(raw_idx, num_options):
    idx = np.nan_to_num(np.asarray(raw_idx, dtype=np.float32), nan=1.0)
    idx = idx.astype(np.int16) - 1
    idx[(idx < 0) | (idx >= num_options)] = 0
    return idx.astype(np.int8)

def _cs(curves, idx):
    stacked = np.stack(curves, axis=1)
    return np.take_along_axis(stacked, idx[:, None], axis=1)[:, 0]

def _cp(curves, idx):
    stacked = np.stack(curves, axis=1)
    return np.take_along_axis(stacked, idx[:, None, None], axis=1)[:, 0, :]

def _ppc(xp, yp):
    return (np.isfinite(np.asarray(xp, dtype=np.float32)) & np.isfinite(np.asarray(yp, dtype=np.float32))).sum(axis=1).astype(np.int16)

def _crs(xp):
    xp = np.asarray(xp, dtype=np.float32)
    finite_pair = np.isfinite(xp[:, :-1]) & np.isfinite(xp[:, 1:])
    return ((np.diff(xp, axis=1) < -1e-06) & finite_pair).sum(axis=1).astype(np.int8)

def _css(xp, yp):
    xp = np.asarray(xp, dtype=np.float32)
    yp = np.asarray(yp, dtype=np.float32)
    dx = np.diff(xp, axis=1)
    dy = np.diff(yp, axis=1)
    valid = np.isfinite(xp[:, :-1]) & np.isfinite(xp[:, 1:]) & np.isfinite(yp[:, :-1]) & np.isfinite(yp[:, 1:]) & (np.abs(dx) > 1e-06)
    slopes = np.full(dx.shape, np.nan, dtype=np.float32)
    slopes[valid] = dy[valid] / dx[valid]
    return (_nmr(slopes), _x(np.abs(slopes)))

def _pwi(x, xp, yp):
    x = np.asarray(x, dtype=np.float32)
    xp = np.asarray(xp, dtype=np.float32)
    yp = np.asarray(yp, dtype=np.float32)
    n_rows, n_points = xp.shape
    result = np.full(n_rows, np.nan, dtype=np.float32)
    vpts = np.isfinite(xp) & np.isfinite(yp)
    hv = vpts.any(axis=1)
    if n_points == 0:
        return result
    row_idx = np.arange(n_rows)
    fv = np.argmax(vpts, axis=1)
    lv = n_points - 1 - np.argmax(vpts[:, ::-1], axis=1)
    first_x = np.full(n_rows, np.nan, dtype=np.float32)
    first_y = np.full(n_rows, np.nan, dtype=np.float32)
    last_x = np.full(n_rows, np.nan, dtype=np.float32)
    last_y = np.full(n_rows, np.nan, dtype=np.float32)
    first_x[hv] = xp[row_idx[hv], fv[hv]]
    first_y[hv] = yp[row_idx[hv], fv[hv]]
    last_x[hv] = xp[row_idx[hv], lv[hv]]
    last_y[hv] = yp[row_idx[hv], lv[hv]]
    for seg in range(n_points - 1):
        x0 = xp[:, seg]
        x1 = xp[:, seg + 1]
        y0 = yp[:, seg]
        y1 = yp[:, seg + 1]
        valid_seg = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1) & (np.abs(x1 - x0) > 1e-06)
        lo = np.minimum(x0, x1)
        hi = np.maximum(x0, x1)
        mask = valid_seg & np.isfinite(x) & np.isnan(result) & (x >= lo) & (x <= hi)
        if mask.any():
            frac = (x[mask] - x0[mask]) / (x1[mask] - x0[mask])
            result[mask] = y0[mask] + frac * (y1[mask] - y0[mask])
    low_mask = hv & np.isfinite(x) & np.isnan(result) & (x <= np.minimum(first_x, last_x))
    result[low_mask] = first_y[low_mask]
    high_mask = hv & np.isfinite(x) & np.isnan(result) & (x >= np.maximum(first_x, last_x))
    result[high_mask] = last_y[high_mask]
    return result

def _var_pct(var, vmi, vma):
    var = np.asarray(var, dtype=np.float32)
    denom = np.where(var >= 0, np.asarray(vmi, dtype=np.float32), np.asarray(vma, dtype=np.float32))
    return 100.0 * _d(var, denom)

def _tt(y_true, prob, *, l=FTF, h=MT, s=0.01):
    if len(y_true) == 0:
        return (0.5, 0.0)
    bt, best_f2 = (0.5, -1.0)
    ths = np.arange(l, h + 1e-09, s, dtype=np.float32)
    for thr in ths:
        pred = (prob >= thr).astype(np.int8)
        score = fbeta_score(y_true, pred, beta=2)
        if score > best_f2:
            bt, best_f2 = (float(thr), float(score))
    return (bt, best_f2)

def _bp(primary, secondary, weight):
    if secondary is None:
        return primary.astype(np.float32)
    return (weight * primary + (1.0 - weight) * secondary).astype(np.float32)

def _snc(df, cds):
    keep = []
    for col in cds:
        if col not in df.columns:
            continue
        series = df[col]
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return keep

def _edf(df):
    out = df.copy()
    out['df'] = out['df'].map(FAM).fillna(-1).astype(np.int8)
    return out

def _gsf(columns):
    excluded = {'Id', 'Label', 'fold_id', 'af', 'dg', 'ra', 'hrc', 'rs', 'oa'}
    excluded.update(SS.values())
    return [col for col in columns if col not in excluded and col not in SLF]

def _b(values, *, fv, dtype, scale=1.0, round_values=True):
    out = pd.to_numeric(values, errors='coerce')
    if scale != 1.0:
        out = out / scale
    if round_values:
        out = out.round()
    return out.fillna(fv).astype(dtype)

def _hf(frame):
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)

def _lss(keys, sum_map, cm0):
    ks = pd.Series(keys)
    sv = ks.map(sum_map).fillna(0.0).to_numpy(np.float32)
    cv = ks.map(cm0).fillna(0).to_numpy(np.int32)
    return (sv, cv)

def _asf(out, *, fp, sr, sc, so, oc):
    out['sr'] = sr.astype(np.float32)
    out['srd'] = (sr - fp).astype(np.float32)
    out['sc'] = sc.astype(np.int32)
    out['slc'] = np.log1p(sc).astype(np.float32)
    out['sls'] = (sc < 20).astype(np.int8)
    out['so'] = so.astype(np.float32)
    out['sord'] = (so - fp).astype(np.float32)
    out['oc'] = oc.astype(np.int32)
    out['solc'] = np.log1p(oc).astype(np.float32)
    out['sols'] = (oc < 20).astype(np.int8)
    return out

def _spm(ids, *, fp):
    ids_arr = np.asarray(ids, dtype=np.int64)
    fit_mask = ids_arr % 2 == 0
    return fit_mask if fp else ~fit_mask

def _sfc(sdf):
    excluded = {'Id', 'Label', 'fold_id', 'af', 'oa', 'dg'}
    excluded.update(SS.values())
    return [col for col in sdf.columns if col not in excluded and pd.api.types.is_numeric_dtype(sdf[col])]

def _cfc(cat_df):
    rnc = [SR[col] for col in RN if SR[col] in cat_df.columns]
    missing_cols = [col for col in cat_df.columns if col.startswith('missing_')]
    cc = [SS[col] for col in RSC if SS[col] in cat_df.columns]
    cds = dedupe([*rnc, *cc, 'dg', *CEC, *missing_cols])
    excluded = {'Id', 'Label', 'fold_id', 'af', 'oa', 'ra'}
    return [col for col in cds if col in cat_df.columns and col not in excluded]

def _cn(df):
    for col in NSC:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

def _bsw(x_df, y):
    weights = np.ones(len(x_df), dtype=np.float32)
    family = x_df['df'].to_numpy()
    ho = x_df['oa'].to_numpy() == 1
    weights[(family == 'c1') & (y == 0)] *= CNW
    weights[ho] *= HOTW
    return weights

def _bsf(x_df, *, iob):
    frame = {'family': x_df['df'].astype(str), 'throt_src': _b(x_df['DERMeasureAC_0_ThrotSrc'], fv=-1, dtype=np.int16), 'throt_pct': _b(x_df['DERMeasureAC_0_ThrotPct'], scale=5.0, fv=-1, dtype=np.int16), 'wmaxlim_pct': _b(x_df['DERCtlAC_0_WMaxLimPct'], scale=5.0, fv=-1, dtype=np.int16), 'wset_pct': _b(x_df['DERCtlAC_0_WSetPct'], scale=5.0, fv=-1, dtype=np.int16), 'varset_pct': _b(x_df['DERCtlAC_0_VarSetPct'], scale=5.0, fv=-1, dtype=np.int16), 'pf_set': _b(x_df['DERCtlAC_0_PFWInj_PF'], scale=0.02, fv=-1, dtype=np.int16), 'fd_idx': _b(x_df['DERFreqDroop_0_AdptCtlRslt'], fv=-1, dtype=np.int16), 'vv_idx': _b(x_df['DERVoltVar_0_AdptCrvRslt'], fv=-1, dtype=np.int16), 'vw_idx': _b(x_df['DERVoltWatt_0_AdptCrvRslt'], fv=-1, dtype=np.int16), 'wv_idx': _b(x_df['DERWattVar_0_AdptCrvRslt'], fv=-1, dtype=np.int16), 'volt_bin': _b(x_df['vp'], fv=-999, dtype=np.int16), 'hz_bin': _b(x_df['DERMeasureAC_0_Hz'], scale=0.1, fv=-999, dtype=np.int16), 'enter_idle': _b(x_df['esi'], fv=0, dtype=np.int8, round_values=False), 'droop_active': _b(x_df['fod'], fv=0, dtype=np.int8, round_values=False)}
    if iob:
        frame['w_bin'] = _b(x_df['wpr'], scale=5.0, fv=-999, dtype=np.int16)
        frame['var_bin'] = _b(x_df['vpl'], scale=5.0, fv=-999, dtype=np.int16)
        frame['pf_mode'] = _b(x_df['pcae'], fv=0, dtype=np.int8, round_values=False)
    return pd.DataFrame(frame)

def _bsk(x_df):
    return _hf(_bsf(x_df, iob=False))

def _bok(x_df):
    return _hf(_bsf(x_df, iob=True))

def _afi(x_df):
    out = x_df.copy()
    c1m = out['df'].astype(str) == 'c1'
    for fname in CIF:
        if fname not in out.columns:
            continue
        values = pd.to_numeric(out[fname], errors='coerce').to_numpy(np.float32)
        out[f'c1_{fname}'] = np.where(c1m.to_numpy(), values, 0.0).astype(np.float32)
    return out

class R:

    def __init__(self, *, ad=DOD / 'artifacts', chunksize=5000, kf=5, ne=180, md=8, lr=0.05, su=0.8, cb=0.8, cti=400, cd=8, clr=0.05, nj=4, seed=DEFAULT_SEED):
        self.ad = ad
        self.chunksize = chunksize
        self.kf = kf
        self.ne = ne
        self.md = md
        self.lr = lr
        self.su = su
        self.cb = cb
        self.cti = cti
        self.cd = cd
        self.clr = clr
        self.nj = nj
        self.seed = seed
        self.ovr = list(OVR)
        self.smd = {}
        self.cmd = {}
        self.ctx = {}
        self.thr = {'c0': 0.5, 'c1': 0.5}
        self.blend_w = {'c0': 1.0, 'c1': 1.0}
        self.sm0 = {}
        self.ccs = {}
        self.sc0 = None
        self.sgm = {}
        self.res_q = {}
        self.fbr = {}
        self.ssm = {}
        self.scm = {}
        self.sosm = {}
        self.socm = {}

    def _abm(self, data, df):
        bmt = np.zeros(len(df), dtype=np.int16)
        bma = np.zeros(len(df), dtype=np.int16)
        for bn, cols in BSC.items():
            missing = df[cols].isna()
            mct = missing.sum(axis=1).astype(np.int16).to_numpy()
            data[f'missing_{bn}_count'] = mct
            data[f'missing_{bn}_any'] = (mct > 0).astype(np.int8)
            bmt += mct
            bma += (mct > 0).astype(np.int16)
        data['mst'] = bmt
        data['msb'] = bma.astype(np.int8)
        cmi = df[[*CS, 'common[0].ID', 'common[0].L']].isna().to_numpy(dtype=np.uint16)
        cwt = (1 << np.arange(cmi.shape[1], dtype=np.uint16)).reshape(1, -1)
        data['cmp'] = (cmi * cwt).sum(axis=1).astype(np.int16)
        ems = df[ESC].isna().to_numpy(dtype=np.uint16)
        ewt = (1 << np.arange(ems.shape[1], dtype=np.uint16)).reshape(1, -1)
        data['emp'] = (ems * ewt).sum(axis=1).astype(np.int16)

    def _ami(self, data, df):
        as0 = np.zeros(len(df), dtype=np.int16)
        msm = np.zeros(len(df), dtype=np.int16)
        for bn, (id_col, len_col, expected_id, expected_len) in EMM.items():
            raw_id = df[id_col].to_numpy(float)
            raw_len = df[len_col].to_numpy(float)
            im0 = ~np.isfinite(raw_id)
            lms = ~np.isfinite(raw_len)
            id_match = np.isclose(raw_id, expected_id, equal_nan=False)
            lmt = np.isclose(raw_len, expected_len, equal_nan=False)
            data[f'{bn}_model_id_missing'] = im0.astype(np.int8)
            data[f'{bn}_model_len_missing'] = lms.astype(np.int8)
            data[f'{bn}_model_id_match'] = id_match.astype(np.int8)
            data[f'{bn}_model_len_match'] = lmt.astype(np.int8)
            data[f'{bn}_model_integrity_ok'] = (id_match & lmt).astype(np.int8)
            mismatch = ~im0 & ~id_match | ~lms & ~lmt
            data[f'{bn}_ms0_anomaly'] = mismatch.astype(np.int8)
            as0 += mismatch.astype(np.int16)
            msm += (im0 | lms).astype(np.int16)
        data['msac'] = as0.astype(np.int8)
        data['msmc'] = msm.astype(np.int8)
        data['msa'] = (as0 > 0).astype(np.int8)

    def _ace(self, data, *, wmaxrtg, wmax, vamaxrtg, vamax, viri, vmi, vart, vma, vnomrtg, vnom, vmaxrtg, vmax, vminrtg, vmin, amaxrtg, amax, wcha_rtg, wdis_rtg, vcr, vdr, wcha, wdis, vacha, vadis, por, pfover, pur, pfunder):
        data['vnsd'] = (vnom - vnomrtg).astype(np.float32)
        data['vxsd'] = (vmax - vmaxrtg).astype(np.float32)
        data['vmsd'] = (vmin - vminrtg).astype(np.float32)
        data['amsd'] = (amax - amaxrtg).astype(np.float32)
        data['pfod'] = (pfover - por).astype(np.float32)
        data['pfud'] = (pfunder - pur).astype(np.float32)
        data['crsr'] = _d(wcha_rtg, wmaxrtg)
        data['drsr'] = _d(wdis_rtg, wmaxrtg)
        data['cvsr'] = _d(vcr, vamaxrtg)
        data['dvsr'] = _d(vdr, vamaxrtg)
        data['crss'] = _d(wcha, wmax)
        data['drss'] = _d(wdis, wmax)
        data['cvss'] = _d(vacha, vamax)
        data['dvss'] = _d(vadis, vamax)
        rating_pairs = [(wmaxrtg, wmax), (vamaxrtg, vamax), (viri, vmi), (vart, vma), (vnomrtg, vnom), (vmaxrtg, vmax), (vminrtg, vmin), (amaxrtg, amax)]
        gct = np.zeros(len(wmaxrtg), dtype=np.int16)
        for rating, setting in rating_pairs:
            tol = np.maximum(1.0, 0.01 * np.nan_to_num(np.abs(rating), nan=0.0)).astype(np.float32)
            gap = np.isfinite(rating) & np.isfinite(setting) & (np.abs(setting - rating) > tol)
            gct += gap.astype(np.int16)
        data['rsgc'] = gct.astype(np.int8)

    def _atf(self, data, df):
        temp_cols = ['DERMeasureAC[0].TmpAmb', 'DERMeasureAC[0].TmpCab', 'DERMeasureAC[0].TmpSnk', 'DERMeasureAC[0].TmpTrns', 'DERMeasureAC[0].TmpSw', 'DERMeasureAC[0].TmpOt']
        temps = df[temp_cols].to_numpy(float)
        tmn = _n(temps)
        tmx = _x(temps)
        tme = _nmr(temps)
        amb = df['DERMeasureAC[0].TmpAmb'].to_numpy(float)
        data['tmn'] = tmn
        data['tmx'] = tmx
        data['tme'] = tme
        data['temp_spread'] = (tmx - tmn).astype(np.float32)
        data['tmoa'] = (tmx - amb).astype(np.float32)

    def _aes(self, data, df, *, vp, hz, abs_w, va, a, tolw, tolva, amax):
        es = df['DEREnterService[0].ES'].to_numpy(float)
        es_v_hi = df['DEREnterService[0].ESVHi'].to_numpy(float)
        es_v_lo = df['DEREnterService[0].ESVLo'].to_numpy(float)
        es_hz_hi = df['DEREnterService[0].ESHzHi'].to_numpy(float)
        es_hz_lo = df['DEREnterService[0].ESHzLo'].to_numpy(float)
        es_delay = df['DEREnterService[0].ESDlyTms'].to_numpy(float)
        er0 = df['DEREnterService[0].ESRndTms'].to_numpy(float)
        es_ramp = df['DEREnterService[0].ESRmpTms'].to_numpy(float)
        edr = df['DEREnterService[0].ESDlyRemTms'].to_numpy(float)
        inside_v = np.isfinite(vp) & np.isfinite(es_v_hi) & np.isfinite(es_v_lo) & (vp >= es_v_lo) & (vp <= es_v_hi)
        ihz = np.isfinite(hz) & np.isfinite(es_hz_hi) & np.isfinite(es_hz_lo) & (hz >= es_hz_lo) & (hz <= es_hz_hi)
        iwin = inside_v & ihz
        enabled = np.isfinite(es) & (es == 1.0)
        saz = np.isfinite(es) & (es >= 1.5)
        sidl = ~enabled | ~iwin
        ct0 = np.maximum(1.0, 0.02 * np.nan_to_num(amax, nan=0.0))
        data['ese'] = enabled.astype(np.int8)
        data['esa'] = saz.astype(np.int8)
        data['esiw'] = iwin.astype(np.int8)
        data['esow'] = (~iwin).astype(np.int8)
        data['esi'] = sidl.astype(np.int8)
        data['esvw'] = (es_v_hi - es_v_lo).astype(np.float32)
        data['eshw'] = (es_hz_hi - es_hz_lo).astype(np.float32)
        data['esvl'] = (vp - es_v_lo).astype(np.float32)
        data['esvh'] = (es_v_hi - vp).astype(np.float32)
        data['eshl'] = (hz - es_hz_lo).astype(np.float32)
        data['eshh'] = (es_hz_hi - hz).astype(np.float32)
        data['estd'] = (es_delay + er0).astype(np.float32)
        data['esdr'] = edr.astype(np.float32)
        data['esrt'] = es_ramp.astype(np.float32)
        data['esda'] = (np.nan_to_num(edr, nan=0.0) > 0).astype(np.int8)
        bpwr = sidl & (abs_w > tolw)
        blocked_va = sidl & (va > tolva)
        bcur = sidl & (a > ct0)
        data['ebp'] = bpwr.astype(np.int8)
        data['esbv'] = blocked_va.astype(np.int8)
        data['ebc'] = bcur.astype(np.int8)
        return (saz.astype(np.int8), bpwr.astype(np.int8), bcur.astype(np.int8))

    def _apc(self, data, df, *, pf, var, vmi, vma):
        pie = np.nan_to_num(df['DERCtlAC[0].PFWInjEna'].to_numpy(float), nan=0.0)
        per = np.nan_to_num(df['DERCtlAC[0].PFWInjEnaRvrt'].to_numpy(float), nan=0.0)
        pfabs_ena = np.nan_to_num(df['DERCtlAC[0].PFWAbsEna'].to_numpy(float), nan=0.0)
        par0 = np.nan_to_num(df['DERCtlAC[0].PFWAbsEnaRvrt'].to_numpy(float), nan=0.0)
        ptg = df['DERCtlAC[0].PFWInj.PF'].to_numpy(float)
        prt = df['DERCtlAC[0].PFWInjRvrt.PF'].to_numpy(float)
        pfinj_ext = df['DERCtlAC[0].PFWInj.Ext'].to_numpy(float)
        prx = df['DERCtlAC[0].PFWInjRvrt.Ext'].to_numpy(float)
        pfabs_ext = df['DERCtlAC[0].PFWAbs.Ext'].to_numpy(float)
        pare = df['DERCtlAC[0].PFWAbsRvrt.Ext'].to_numpy(float)
        ovp = _var_pct(var, vmi, vma)
        ite = np.where((pie > 0) & np.isfinite(ptg), np.abs(np.abs(pf) - ptg), np.nan)
        ire = np.where((per > 0) & np.isfinite(prt), np.abs(np.abs(pf) - prt), np.nan)
        data['pcae'] = ((pie > 0) | (pfabs_ena > 0)).astype(np.int8)
        data['pcar'] = ((per > 0) | (par0 > 0)).astype(np.int8)
        data['pte'] = ite.astype(np.float32)
        data['pre2'] = ire.astype(np.float32)
        data['piep'] = np.isfinite(pfinj_ext).astype(np.int8)
        data['pire'] = np.isfinite(prx).astype(np.int8)
        data['pae'] = np.isfinite(pfabs_ext).astype(np.int8)
        data['pre'] = np.isfinite(pare).astype(np.int8)
        data['piem'] = ((pie > 0) & ~np.isfinite(ptg)).astype(np.int8)
        data['prl'] = (np.abs(ovp) >= 95.0).astype(np.int8)
        return (np.isfinite(pfabs_ext).astype(np.int8), np.isfinite(pare).astype(np.int8))

    def _atb(self, data, df, *, sn, prefix, an, mode, mv, abs_w, tolw):
        adpt_idx = _ci(df[f'{prefix}.AdptCrvRslt'].to_numpy(float), 2)
        gs = lambda group, field: _cs([df[f'{prefix}.Crv[{curve}].{group}.{field}'].to_numpy(float) for curve in range(2)], adpt_idx)
        gp = lambda group, field: _cp([np.column_stack([df[f'{prefix}.Crv[{curve}].{group}.Pt[{i}].{field}'].to_numpy(float) for i in range(5)]) for curve in range(2)], adpt_idx)
        must_actpt = gs('MustTrip', 'ActPt')
        mom_actpt = gs('MomCess', 'ActPt')
        must_x = gp('MustTrip', an)
        must_t = gp('MustTrip', 'Tms')
        mom_x = gp('MomCess', an)
        mom_t = gp('MomCess', 'Tms')
        may_present = np.column_stack([df[f'{prefix}.Crv[{curve}].MayTrip.Pt[{point}].{an}'].to_numpy(float) for curve in range(2) for point in range(5)])
        enabled = np.nan_to_num(df[f'{prefix}.Ena'].to_numpy(float), nan=0.0) > 0
        mc0 = _ppc(must_x, must_t)
        mom_count = _ppc(mom_x, mom_t)
        mxn = _n(must_x)
        mxx = _x(must_x)
        must_t_min = _n(must_t)
        must_t_max = _x(must_t)
        mxm = _n(mom_x)
        mom_x_max = _x(mom_x)
        mom_t_min = _n(mom_t)
        mom_t_max = _x(mom_t)
        if mode == 'low':
            margin = mv - mxx
        else:
            margin = mxn - mv
        outside = enabled & np.isfinite(margin) & (margin < 0)
        pwo = outside & (abs_w > tolw)
        envelope_gap = np.where(np.isfinite(mxm) & np.isfinite(mxx), np.abs(mxm - mxx), np.nan)
        data[f'trip_{sn}_ci'] = adpt_idx.astype(np.int8)
        data[f'trip_{sn}_enabled'] = enabled.astype(np.int8)
        data[f'trip_{sn}_crg'] = (df[f'{prefix}.AdptCrvReq'].to_numpy(float) - df[f'{prefix}.AdptCrvRslt'].to_numpy(float)).astype(np.float32)
        data[f'trip_{sn}_mt_count'] = mc0
        data[f'trip_{sn}_mt_actpt_gap'] = (must_actpt - mc0).astype(np.float32)
        data[f'trip_{sn}_mt_axis_min'] = mxn
        data[f'trip_{sn}_mt_axis_max'] = mxx
        data[f'trip_{sn}_mt_axis_span'] = (mxx - mxn).astype(np.float32)
        data[f'trip_{sn}_mt_tms_span'] = (must_t_max - must_t_min).astype(np.float32)
        data[f'trip_{sn}_mt_reverse_steps'] = _crs(must_x)
        data[f'trip_{sn}_mc_count'] = mom_count
        data[f'trip_{sn}_mc_actpt_gap'] = (mom_actpt - mom_count).astype(np.float32)
        data[f'trip_{sn}_mc_axis_span'] = (mom_x_max - mxm).astype(np.float32)
        data[f'trip_{sn}_mc_tms_span'] = (mom_t_max - mom_t_min).astype(np.float32)
        data[f'trip_{sn}_mc_reverse_steps'] = _crs(mom_x)
        data[f'trip_{sn}_mpa'] = np.isfinite(may_present).any(axis=1).astype(np.int8)
        data[f'trip_{sn}_mt_margin'] = margin.astype(np.float32)
        data[f'trip_{sn}_om'] = outside.astype(np.int8)
        data[f'trip_{sn}_pwo'] = pwo.astype(np.int8)
        data[f'trip_{sn}_mc_mt_gap'] = envelope_gap.astype(np.float32)
        return (outside.astype(np.int8), pwo.astype(np.int8))

    def _acb(self, data, *, name, raw_idx, curve_x, curve_y, capt, cmx, mv, ov=None):
        adpt_idx = _ci(raw_idx, len(curve_x))
        sx = _cp(curve_x, adpt_idx)
        sy = _cp(curve_y, adpt_idx)
        sa = _cs(capt, adpt_idx)
        data[f'{name}_ci'] = adpt_idx.astype(np.int8)
        pc = _ppc(sx, sy)
        data[f'{name}_cpc'] = pc
        data[f'{name}_cag'] = (sa - pc).astype(np.float32)
        x_min = _n(sx)
        x_max = _x(sx)
        y_min = _n(sy)
        y_max = _x(sy)
        mean_slope, max_abs_slope = _css(sx, sy)
        data[f'{name}_cxs'] = (x_max - x_min).astype(np.float32)
        data[f'{name}_cys'] = (y_max - y_min).astype(np.float32)
        data[f'{name}_crs'] = _crs(sx)
        data[f'{name}_cms'] = mean_slope
        data[f'{name}_cas'] = max_abs_slope
        data[f'{name}_cml'] = (mv - x_min).astype(np.float32)
        data[f'{name}_cmh'] = (x_max - mv).astype(np.float32)
        if ov is not None:
            ev = _pwi(mv, sx, sy)
            data[f'{name}_ce'] = ev.astype(np.float32)
            data[f'{name}_cer'] = (ov - ev).astype(np.float32)
        for mn, curves in cmx.items():
            data[f'{name}_curve_{mn}'] = _cs(curves, adpt_idx).astype(np.float32)

    def _afd(self, data, df, *, hz, w_pct):
        raw_idx = df['DERFreqDroop[0].AdptCtlRslt'].to_numpy(float)
        ctl_idx = _ci(raw_idx, 3)
        dbc = [df[f'DERFreqDroop[0].Ctl[{i}].DbOf'].to_numpy(float) for i in range(3)]
        duc = [df[f'DERFreqDroop[0].Ctl[{i}].DbUf'].to_numpy(float) for i in range(3)]
        kof_curves = [df[f'DERFreqDroop[0].Ctl[{i}].KOf'].to_numpy(float) for i in range(3)]
        kuf_curves = [df[f'DERFreqDroop[0].Ctl[{i}].KUf'].to_numpy(float) for i in range(3)]
        rsp_curves = [df[f'DERFreqDroop[0].Ctl[{i}].RspTms'].to_numpy(float) for i in range(3)]
        pmc = [df[f'DERFreqDroop[0].Ctl[{i}].PMin'].to_numpy(float) for i in range(3)]
        ro_curves = [df[f'DERFreqDroop[0].Ctl[{i}].ReadOnly'].to_numpy(float) for i in range(3)]
        dbof = _cs(dbc, ctl_idx)
        dbuf = _cs(duc, ctl_idx)
        kof = _cs(kof_curves, ctl_idx)
        kuf = _cs(kuf_curves, ctl_idx)
        rsp = _cs(rsp_curves, ctl_idx)
        pmin = _cs(pmc, ctl_idx)
        rdo = _cs(ro_curves, ctl_idx)
        oa2 = np.maximum(hz - (60.0 + dbof), 0.0)
        ua = np.maximum(60.0 - dbuf - hz, 0.0)
        edp = 100.0 * _d(oa2, kof) - 100.0 * _d(ua, kuf)
        dbof_stack = np.column_stack(dbc)
        dbuf_stack = np.column_stack(duc)
        k_stack = np.column_stack(kof_curves + kuf_curves)
        pmin_stack = np.column_stack(pmc)
        data['fdci'] = ctl_idx.astype(np.int8)
        data['fdof'] = dbof.astype(np.float32)
        data['fduf'] = dbuf.astype(np.float32)
        data['fdko'] = kof.astype(np.float32)
        data['fdku'] = kuf.astype(np.float32)
        data['fdrs'] = rsp.astype(np.float32)
        data['fdpm'] = pmin.astype(np.float32)
        data['fdro'] = rdo.astype(np.float32)
        data['fdbw'] = (dbof + dbuf).astype(np.float32)
        data['fdoa'] = oa2.astype(np.float32)
        data['fdua'] = ua.astype(np.float32)
        data['fdep'] = edp.astype(np.float32)
        data['fod'] = ((oa2 > 0) | (ua > 0)).astype(np.int8)
        data['fdwp'] = (w_pct - pmin).astype(np.float32)
        data['fdbs'] = (_x(np.column_stack([dbof_stack, dbuf_stack])) - _n(np.column_stack([dbof_stack, dbuf_stack]))).astype(np.float32)
        data['fdks'] = (_x(k_stack) - _n(k_stack)).astype(np.float32)
        data['fdps'] = (_x(pmin_stack) - _n(pmin_stack)).astype(np.float32)

    def _adc(self, data, df, *, w, abs_w):
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
        data['dcw_over_w'] = _d(dcw, w)
        data['dwwa'] = _d(dcw, abs_w)
        data['dmps'] = (dcw - (prt0 + prt1)).astype(np.float32)
        data['dcv_spread'] = np.abs(prt0_v - prt1_v).astype(np.float32)
        data['dca_spread'] = np.abs(prt0_a - prt1_a).astype(np.float32)
        data['dps'] = _d(prt0, prt0 + prt1)
        data['dptm'] = (np.isfinite(prt0_t) & np.isfinite(prt1_t) & (prt0_t != prt1_t)).astype(np.int8)
        rare_type = (prt0_t == 7) | (prt1_t == 7)
        data['dtr'] = rare_type.astype(np.int8)
        data['azdp'] = ((np.abs(w) <= 1e-06) & (dcw > 0)).astype(np.int8)
        data['apdz'] = ((w > 0) & (np.abs(dcw) <= 1e-06)).astype(np.int8)
        data['adss'] = (np.sign(np.nan_to_num(w, nan=0.0)) == np.sign(np.nan_to_num(dcw, nan=0.0))).astype(np.int8)
        data['dot'] = _d(dca, prt0_a + prt1_a)
        return rare_type.astype(np.int8)

    def bf(self, df):
        _cn(df)
        fpg = df[CS].fillna('<NA>').agg('|'.join, axis=1)
        data = {'Id': df['Id'].to_numpy(), 'dg': fpg.to_numpy(dtype=object), 'df': np.where(fpg == CANON1, 'c0', np.where(fpg == CANON2, 'c1', 'o')), 'cma': df[CS].isna().any(axis=1).astype(np.int8).to_numpy(), 'cmc': df[CS].isna().sum(axis=1).astype(np.int16).to_numpy(), 'csd': df['common[0].SN'].fillna('').astype(str).str.endswith('.0').astype(np.int8).to_numpy()}
        data['nc'] = (data['df'] == 'o').astype(np.int8)
        for col in RN:
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            data[SR[col]] = arr
        for col in RSC:
            data[SS[col]] = df[col].fillna('<NA>').astype(str).to_numpy(dtype=object)
        self._abm(data, df)
        self._ami(data, df)
        self._atf(data, df)
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
        viri = df['DERCapacity[0].VarMaxInjRtg'].to_numpy(float)
        vart = df['DERCapacity[0].VarMaxAbsRtg'].to_numpy(float)
        wmax = df['DERCapacity[0].WMax'].to_numpy(float)
        vamax = df['DERCapacity[0].VAMax'].to_numpy(float)
        vmi = df['DERCapacity[0].VarMaxInj'].to_numpy(float)
        vma = df['DERCapacity[0].VarMaxAbs'].to_numpy(float)
        amax = df['DERCapacity[0].AMax'].to_numpy(float)
        vnom = df['DERCapacity[0].VNom'].to_numpy(float)
        vmax = df['DERCapacity[0].VMax'].to_numpy(float)
        vmin = df['DERCapacity[0].VMin'].to_numpy(float)
        for name, numerator, dnr in [('wwr', w, wmaxrtg), ('ww', w, wmax), ('vav', va, vamax), ('vvr', va, vamaxrtg), ('voi', var, vmi), ('voa', var, vma), ('aoa', a, amax), ('llv_over_vnom', llv, vnom), ('lnv_over_vnom', lnv * SQRT3, vnom)]:
            data[name] = _d(numerator, dnr)
        for name, value in [('wmx', w - wmax), ('wmr', w - wmaxrtg), ('vmv', va - vamax), ('vmi0', var - vmi), ('vpa', var + vma), ('llv_minus_lnv_sqrt3', llv - lnv * SQRT3), ('hz_delta_60', hz - 60.0)]:
            data[name] = value.astype(np.float32)
        for name, left, right in [('wew', w, wmaxrtg), ('w_eq_wmax', w, wmax), ('vev', var, vmi), ('vena', var, -vma)]:
            data[name] = np.isclose(left, right, equal_nan=False).astype(np.int8)
        data['psm'] = ((np.sign(np.nan_to_num(pf)) != np.sign(np.nan_to_num(w))) & (np.nan_to_num(pf) != 0) & (np.nan_to_num(w) != 0)).astype(np.int8)
        tolw = np.maximum(50.0, 0.02 * np.nan_to_num(wmaxrtg, nan=0.0)).astype(np.float32)
        tolva = np.maximum(50.0, 0.02 * np.nan_to_num(vamax, nan=0.0)).astype(np.float32)
        tolvi = np.maximum(20.0, 0.02 * np.nan_to_num(vmi, nan=0.0)).astype(np.float32)
        tolva2 = np.maximum(20.0, 0.02 * np.nan_to_num(vma, nan=0.0)).astype(np.float32)
        for name, value, upper_bound in [('gw', w, wmax + tolw), ('gwr', w, wmaxrtg + tolw), ('gva', va, vamax + tolva), ('gvi', var, vmi + tolvi)]:
            data[name] = (value > upper_bound).astype(np.int8)
        data['vla'] = (var < -vma - tolva2).astype(np.int8)
        pq = np.sqrt(np.square(w.astype(np.float32)) + np.square(var.astype(np.float32)))
        data['vmp'] = (va - pq).astype(np.float32)
        data['vop'] = _d(va, pq)
        pfv = _d(w, va)
        data['pfv'] = pfv
        data['pf_error'] = (pf - pfv).astype(np.float32)
        for name, total, suffixes in [('wpe', w, ['WL1', 'WL2', 'WL3']), ('ape', va, ['VAL1', 'VAL2', 'VAL3']), ('vpe', var, ['VarL1', 'VarL2', 'VarL3'])]:
            ps0 = sum((df[f'DERMeasureAC[0].{suffix}'].to_numpy(float) for suffix in suffixes))
            data[name] = (total - ps0).astype(np.float32)
        for name, suffixes in [('phase_ll_spread', ['VL1L2', 'VL2L3', 'VL3L1']), ('phase_ln_spread', ['VL1', 'VL2', 'VL3']), ('pws', ['WL1', 'WL2', 'WL3']), ('pvs', ['VarL1', 'VarL2', 'VarL3'])]:
            pv = df[[f'DERMeasureAC[0].{suffix}' for suffix in suffixes]].to_numpy(float)
            data[name] = (_x(pv) - _n(pv)).astype(np.float32)
        for name, numerator, dnr in [('wmax_over_wmaxrtg', wmax, wmaxrtg), ('vamax_over_vamaxrtg', vamax, vamaxrtg), ('vmax_over_vnom', vmax, vnom), ('vmin_over_vnom', vmin, vnom)]:
            data[name] = _d(numerator, dnr)
        wsetena = np.nan_to_num(df['DERCtlAC[0].WSetEna'].to_numpy(float), nan=0.0)
        wset = df['DERCtlAC[0].WSet'].to_numpy(float)
        wsetpct = df['DERCtlAC[0].WSetPct'].to_numpy(float)
        wle = np.nan_to_num(df['DERCtlAC[0].WMaxLimPctEna'].to_numpy(float), nan=0.0)
        wmaxlimpct = df['DERCtlAC[0].WMaxLimPct'].to_numpy(float)
        varsetena = np.nan_to_num(df['DERCtlAC[0].VarSetEna'].to_numpy(float), nan=0.0)
        varset = df['DERCtlAC[0].VarSet'].to_numpy(float)
        varsetpct = df['DERCtlAC[0].VarSetPct'].to_numpy(float)
        wsae = np.where(wsetena > 0, np.abs(w - wset), np.nan)
        wspt = wmaxrtg * (wsetpct / 100.0)
        wspe = np.where(wsetena > 0, np.abs(w - wspt), np.nan)
        wltg = wmaxrtg * (wmaxlimpct / 100.0)
        wlex = np.where(wle > 0, w - wltg, np.nan)
        vsae = np.where(varsetena > 0, np.abs(var - varset), np.nan)
        vspt = vmi * (varsetpct / 100.0)
        vspe = np.where(varsetena > 0, np.abs(var - vspt), np.nan)
        data['wsae'] = wsae.astype(np.float32)
        data['wspt'] = wspt.astype(np.float32)
        data['wspe'] = wspe.astype(np.float32)
        data['wltg'] = wltg.astype(np.float32)
        data['wlex'] = wlex.astype(np.float32)
        data['vsae'] = vsae.astype(np.float32)
        data['vspt'] = vspt.astype(np.float32)
        data['vspe'] = vspe.astype(np.float32)
        data['wf'] = ((wsetena > 0) & (wsae > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data['wef'] = ((wsetena > 0) & (wspe > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data['wmf'] = ((wle > 0) & (wlex > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))).astype(np.int8)
        data['vef'] = ((varsetena > 0) & (vspe > np.maximum(20.0, 0.05 * np.nan_to_num(vmi, nan=0.0)))).astype(np.int8)
        self._ace(data, wmaxrtg=wmaxrtg, wmax=wmax, vamaxrtg=vamaxrtg, vamax=vamax, viri=viri, vmi=vmi, vart=vart, vma=vma, vnomrtg=df['DERCapacity[0].VNomRtg'].to_numpy(float), vnom=vnom, vmaxrtg=df['DERCapacity[0].VMaxRtg'].to_numpy(float), vmax=vmax, vminrtg=df['DERCapacity[0].VMinRtg'].to_numpy(float), vmin=vmin, amaxrtg=df['DERCapacity[0].AMaxRtg'].to_numpy(float), amax=amax, wcha_rtg=df['DERCapacity[0].WChaRteMaxRtg'].to_numpy(float), wdis_rtg=df['DERCapacity[0].WDisChaRteMaxRtg'].to_numpy(float), vcr=df['DERCapacity[0].VAChaRteMaxRtg'].to_numpy(float), vdr=df['DERCapacity[0].VADisChaRteMaxRtg'].to_numpy(float), wcha=df['DERCapacity[0].WChaRteMax'].to_numpy(float), wdis=df['DERCapacity[0].WDisChaRteMax'].to_numpy(float), vacha=df['DERCapacity[0].VAChaRteMax'].to_numpy(float), vadis=df['DERCapacity[0].VADisChaRteMax'].to_numpy(float), por=df['DERCapacity[0].PFOvrExtRtg'].to_numpy(float), pfover=df['DERCapacity[0].PFOvrExt'].to_numpy(float), pur=df['DERCapacity[0].PFUndExtRtg'].to_numpy(float), pfunder=df['DERCapacity[0].PFUndExt'].to_numpy(float))
        vp = 100.0 * _d(llv, vnom)
        lnp = 100.0 * _d(lnv * SQRT3, vnom)
        w_pct = 100.0 * _d(w, wmaxrtg)
        var_pct = _var_pct(var, vmi, vma)
        data['vp'] = vp.astype(np.float32)
        data['lnp'] = lnp.astype(np.float32)
        data['wpr'] = w_pct.astype(np.float32)
        data['vpl'] = var_pct.astype(np.float32)
        esa2, eip, eic = self._aes(data, df, vp=vp, hz=hz, abs_w=abs_w, va=va, a=a, tolw=tolw, tolva=tolva, amax=amax)
        pae, pre = self._apc(data, df, pf=pf, var=var, vmi=vmi, vma=vma)
        tof = []
        tpf = []
        for sn, (prefix, an, mode) in TS.items():
            mv = vp if an == 'V' else hz
            outside, pwo = self._atb(data, df, sn=sn, prefix=prefix, an=an, mode=mode, mv=mv, abs_w=abs_w, tolw=tolw)
            tof.append(outside)
            tpf.append(pwo)
        if tof:
            tao = np.column_stack(tof).any(axis=1).astype(np.int8)
            tpo = np.column_stack(tpf).any(axis=1).astype(np.int8)
        else:
            tao = np.zeros(len(df), dtype=np.int8)
            tpo = np.zeros(len(df), dtype=np.int8)
        data['tom'] = tao
        data['tpo'] = tpo
        csc0 = lambda prefix, field: [df[f'{prefix}.Crv[{curve}].{field}'].to_numpy(float) for curve in range(3)]
        cpt0 = lambda prefix, field, pc: [np.column_stack([df[f'{prefix}.Crv[{curve}].Pt[{point}].{field}'].to_numpy(float) for point in range(pc)]) for curve in range(3)]
        curve_specs = [('voltvar', 'DERVoltVar[0]', 4, 'V', 'Var', {'deptref': 'DeptRef', 'pri': 'Pri', 'vref': 'VRef', 'vref_auto': 'VRefAuto', 'vref_auto_ena': 'VRefAutoEna', 'vref_auto_tms': 'VRefAutoTms', 'rsp': 'RspTms', 'rdo': 'ReadOnly'}, vp - 100.0 + df['DERVoltVar[0].Crv[0].VRef'].fillna(100.0).to_numpy(float), var_pct), ('voltwatt', 'DERVoltWatt[0]', 2, 'V', 'W', {'deptref': 'DeptRef', 'rsp': 'RspTms', 'rdo': 'ReadOnly'}, vp, w_pct), ('wattvar', 'DERWattVar[0]', 6, 'W', 'Var', {'deptref': 'DeptRef', 'pri': 'Pri', 'rdo': 'ReadOnly'}, w_pct, var_pct)]
        for name, prefix, pc, x_field, y_field, meta_fields, mv, ov in curve_specs:
            self._acb(data, name=name, raw_idx=df[f'{prefix}.AdptCrvRslt'].to_numpy(float), curve_x=cpt0(prefix, x_field, pc), curve_y=cpt0(prefix, y_field, pc), capt=csc0(prefix, 'ActPt'), cmx={mn: csc0(prefix, field) for mn, field in meta_fields.items()}, mv=mv, ov=ov)
        self._afd(data, df, hz=hz, w_pct=w_pct)
        dpr = self._adc(data, df, w=w, abs_w=abs_w)
        ac_type = df['DERMeasureAC[0].ACType'].to_numpy(float)
        atr = np.isfinite(ac_type) & (ac_type == 3.0)
        data['atr'] = atr.astype(np.int8)
        fmp = {'nc': data['nc'] == 1, 'cmi': data['cma'] == 1, 'gww': data['gw'] == 1, 'gwrt': data['gwr'] == 1, 'vgv': data['gva'] == 1, 'vgi': data['gvi'] == 1, 'vlm': data['vla'] == 1, 'wsf': data['wf'] == 1, 'wpf': data['wef'] == 1, 'wlf': data['wmf'] == 1, 'vpf': data['vef'] == 1, 'ms0': data['msa'] == 1, 'atr2': atr == 1, 'dtr2': dpr == 1, 'es0': esa2 == 1, 'eip': eip == 1, 'eic': eic == 1, 'pa0': pae == 1, 'pr0': pre == 1, 'tp0': tpo == 1}
        hrf = np.column_stack([fmp[name] for name in HRN])
        hof = np.column_stack([fmp[name] for name in self.ovr])
        ff = {name: flag.astype(np.float32) for name, flag in fmp.items()}
        data['hrc'] = hrf.sum(axis=1).astype(np.int8)
        data['rs'] = 3.0 * ff['nc'] + 2.5 * ff['cmi'] + 2.0 * (ff['gww'] + ff['gwrt'] + ff['vgv'] + ff['vgi'] + ff['vlm'] + ff['ms0'] + ff['es0'] + ff['tp0']) + 1.5 * (ff['wsf'] + ff['wpf'] + ff['atr2'] + ff['dtr2'] + ff['pa0'] + ff['pr0']) + 1.0 * ff['vpf'] + 0.75 * ff['wlf'] + 0.35 * (ff['eip'] + ff['eic'])
        ra = hrf.any(axis=1).astype(np.int8)
        data['ra'] = ra
        data['oa'] = hof.any(axis=1).astype(np.int8)
        return pd.DataFrame(data)

    def irc(self, member, usecols, lmt=0):
        yielded = 0
        for chunk in pd.read_csv(SCRIPT_DIR / member, usecols=list(usecols), chunksize=self.chunksize, low_memory=False):
            if lmt > 0:
                rm = lmt - yielded
                if rm <= 0:
                    break
                if len(chunk) > rm:
                    chunk = chunk.iloc[:rm].copy()
            yielded += len(chunk)
            yield chunk
            if lmt > 0 and yielded >= lmt:
                break

    def _fts(self, x_train, y_train):
        out = x_train.copy()
        y_arr = y_train.to_numpy(np.float32)
        fs = out['df'].astype(str)
        self.fbr = pd.DataFrame({'family': fs, 'y': y_arr}).groupby('family')['y'].mean().to_dict()
        keys = _bsk(out)
        ok = _bok(out)
        fi = (out['Id'].to_numpy(np.int64) % self.kf).astype(np.int8)
        sr = np.zeros(len(out), dtype=np.float32)
        sc = np.zeros(len(out), dtype=np.int32)
        so = np.zeros(len(out), dtype=np.float32)
        oc = np.zeros(len(out), dtype=np.int32)
        gr = float(np.mean(y_arr))
        for fold in range(self.kf):
            tm = fi != fold
            vm = fi == fold
            if not vm.any():
                continue
            stats = pd.DataFrame({'key': keys[tm], 'y': y_arr[tm]}).groupby('key')['y'].agg(['sum', 'count'])
            ost = pd.DataFrame({'key': ok[tm], 'y': y_arr[tm]}).groupby('key')['y'].agg(['sum', 'count'])
            vk = pd.Series(keys[vm])
            valid_sum = vk.map(stats['sum']).fillna(0.0).to_numpy(np.float32)
            vc = vk.map(stats['count']).fillna(0).to_numpy(np.int32)
            vok = pd.Series(ok[vm])
            vos = vok.map(ost['sum']).fillna(0.0).to_numpy(np.float32)
            voc = vok.map(ost['count']).fillna(0).to_numpy(np.int32)
            vf = fs.loc[vm].tolist()
            prior = np.array([self.fbr.get(name, gr) for name in vf], dtype=np.float32)
            sr[vm] = (valid_sum + SM * prior) / (vc + SM)
            sc[vm] = vc
            so[vm] = (vos + SM * prior) / (voc + SM)
            oc[vm] = voc
        fst = pd.DataFrame({'key': keys, 'y': y_arr}).groupby('key')['y'].agg(['sum', 'count'])
        fos = pd.DataFrame({'key': ok, 'y': y_arr}).groupby('key')['y'].agg(['sum', 'count'])
        self.ssm = {int(idx): float(val) for idx, val in fst['sum'].items()}
        self.scm = {int(idx): int(val) for idx, val in fst['count'].items()}
        self.sosm = {int(idx): float(val) for idx, val in fos['sum'].items()}
        self.socm = {int(idx): int(val) for idx, val in fos['count'].items()}
        fp = fs.map(self.fbr).fillna(gr).to_numpy(np.float32)
        return _asf(out, fp=fp, sr=sr, sc=sc, so=so, oc=oc)

    def _apf(self, x_df):
        if not self.scm:
            return x_df
        out = x_df.copy()
        keys = _bsk(out)
        ok = _bok(out)
        sv, cv = _lss(keys, self.ssm, self.scm)
        osv, ocv = _lss(ok, self.sosm, self.socm)
        gr = float(np.mean(list(self.fbr.values()))) if self.fbr else 0.5
        fp = out['df'].astype(str).map(self.fbr).fillna(gr).to_numpy(np.float32)
        sr = (sv + SM * fp) / (cv + SM)
        so = (osv + SM * fp) / (ocv + SM)
        return _asf(out, fp=fp, sr=sr, sc=cv, so=so, oc=ocv)

    def _xsp(self, *, em, vb):
        return {'subsample': self.su, 'colsample_bytree': self.cb, 'eval_metric': em, 'tree_method': 'hist', 'n_jobs': self.nj, 'random_state': self.seed, 'seed': self.seed, 'verbosity': vb}

    def _nsm(self):
        return G(n_estimators=max(80, self.ne // 2), max_depth=max(4, self.md - 2), learning_rate=min(0.08, self.lr * 1.2), objective='reg:squarederror', **self._xsp(em='rmse', vb=0))

    def _ncf(self):
        return X(n_estimators=self.ne, max_depth=self.md, learning_rate=self.lr, objective='binary:logistic', **self._xsp(em='logloss', vb=1))

    def _fsm(self, x_train, y_train, vm):
        self.sc0 = _gsf(x_train.columns)
        fp = _spm(x_train['Id'], fp=True)
        normal_mask = (y_train == 0) & (x_train['oa'] == 0) & (x_train['df'] != 'o') & ~vm.to_numpy() & fp
        sdf2 = x_train.loc[normal_mask].copy()
        if sdf2.empty:
            raise RuntimeError('No rows avail to train surrogate models.')
        self.sgm = {}
        for family in FAM:
            fd = sdf2.loc[sdf2['df'] == family].copy()
            if fd.empty:
                continue
            xs = _edf(fd[self.sc0])
            for tg, (tc, _) in STG.items():
                model = self._nsm()
                y_target = fd[tc].to_numpy(np.float32)
                print(f'[surrogate] training {family}/{tg} on {len(fd):,} normal rows')
                model.fit(xs, y_target)
                self.sgm[family, tg] = model

    def _aws(self, x_df):
        if self.sc0 is None or not self.sgm:
            return x_df
        out = x_df.copy()
        for tg in STG:
            out[f'pred_{tg}'] = np.nan
            out[f'resid_{tg}'] = np.nan
            out[f'ar_{tg}'] = np.nan
            out[f'nr_{tg}'] = np.nan
            out[f'anr_{tg}'] = np.nan
            out[f'tr_{tg}'] = 0
            out[f'er_{tg}'] = 0
            out[f'ur_{tg}'] = 0
            out[f'q9r_{tg}'] = np.nan
        xs = _edf(out[self.sc0])
        for family in FAM:
            fm = out['df'] == family
            if not fm.any():
                continue
            x_family = xs.loc[fm]
            for tg, (tc, scale_col) in STG.items():
                model = self.sgm.get((family, tg))
                if model is None:
                    continue
                pred = model.predict(x_family).astype(np.float32)
                actual = out.loc[fm, tc].to_numpy(np.float32)
                resid = actual - pred
                out.loc[fm, f'pred_{tg}'] = pred
                out.loc[fm, f'resid_{tg}'] = resid
                out.loc[fm, f'ar_{tg}'] = np.abs(resid).astype(np.float32)
                if scale_col is not None:
                    scale = out.loc[fm, scale_col].to_numpy(np.float32)
                    norm_resid = _d(resid, scale)
                else:
                    scale = np.maximum(0.05, np.abs(actual))
                    norm_resid = (resid / scale).astype(np.float32)
                out.loc[fm, f'nr_{tg}'] = norm_resid.astype(np.float32)
                out.loc[fm, f'anr_{tg}'] = np.abs(norm_resid).astype(np.float32)
        out['ret'] = out[['ar_w', 'ar_va', 'ar_var', 'ar_pf', 'ar_a']].sum(axis=1).astype(np.float32)
        out['rvp'] = (out['pred_va'] - np.sqrt(np.square(out['pred_w']) + np.square(out['pred_var']))).astype(np.float32)
        out['rwr'] = _d(out['ar_w'].to_numpy(float), out['ar_var'].to_numpy(float) + 0.001)
        return out

    def _crq(self, x_train, y_train, vm):
        cpn = _spm(x_train['Id'], fp=False)
        base_mask = (y_train == 0) & (x_train['oa'] == 0) & (x_train['df'] != 'o') & ~vm.to_numpy()
        self.res_q = {}
        for family in FAM:
            fm = base_mask & (x_train['df'] == family)
            fc = fm & cpn
            if not fc.any():
                fc = fm
            fq = {}
            for tg in STG:
                series = x_train.loc[fc, f'anr_{tg}']
                values = pd.to_numeric(series, errors='coerce').to_numpy(np.float32)
                values = values[np.isfinite(values)]
                quantiles = RTF.copy()
                if values.size > 0:
                    for level_name, q in RTL.items():
                        quantiles[level_name] = float(np.quantile(values, q))
                fq[tg] = {key: max(1e-06, value) for key, value in quantiles.items()}
            self.res_q[family] = fq

    def _arf(self, x_df):
        if not self.res_q:
            return x_df
        out = x_df.copy()
        for tg in STG:
            out[f'tr_{tg}'] = 0
            out[f'er_{tg}'] = 0
            out[f'ur_{tg}'] = 0
            out[f'q9r_{tg}'] = np.nan
        for family in FAM:
            fm = out['df'] == family
            if not fm.any():
                continue
            fq = self.res_q.get(family, {})
            for tg in STG:
                anm = out.loc[fm, f'anr_{tg}'].to_numpy(np.float32)
                q = fq.get(tg, RTF)
                tail = anm >= q['tail']
                extreme = anm >= q['extreme']
                ultra = anm >= q['ultra']
                q99_ratio = _d(anm, np.full_like(anm, q['extreme'], dtype=np.float32))
                out.loc[fm, f'tr_{tg}'] = tail.astype(np.int8)
                out.loc[fm, f'er_{tg}'] = extreme.astype(np.int8)
                out.loc[fm, f'ur_{tg}'] = ultra.astype(np.int8)
                out.loc[fm, f'q9r_{tg}'] = q99_ratio.astype(np.float32)
        anw = np.nan_to_num(out['anr_w'].to_numpy(np.float32), nan=0.0)
        anv = np.nan_to_num(out['anr_var'].to_numpy(np.float32), nan=0.0)
        anp = np.nan_to_num(out['anr_pf'].to_numpy(np.float32), nan=0.0)
        ana = np.nan_to_num(out['anr_a'].to_numpy(np.float32), nan=0.0)
        pf_mode = np.nan_to_num(out['pcae'].to_numpy(np.float32), nan=0.0) > 0
        vvm = (np.nan_to_num(out['DERVoltVar_0_Ena'].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(out['voltvar_ce'].to_numpy(np.float32))
        vwm = (np.nan_to_num(out['DERVoltWatt_0_Ena'].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(out['voltwatt_ce'].to_numpy(np.float32))
        wvm = (np.nan_to_num(out['DERWattVar_0_Ena'].to_numpy(np.float32), nan=0.0) > 0) & np.isfinite(out['wattvar_ce'].to_numpy(np.float32))
        drm = np.nan_to_num(out['fod'].to_numpy(np.float32), nan=0.0) > 0
        eim = np.nan_to_num(out['esi'].to_numpy(np.float32), nan=0.0) > 0
        out['mrpp'] = (anp * pf_mode).astype(np.float32)
        out['mrvp'] = (anv * pf_mode).astype(np.float32)
        out['mrvv'] = (anv * vvm).astype(np.float32)
        out['mrwv'] = (anw * vwm).astype(np.float32)
        out['mrvw'] = (anv * wvm).astype(np.float32)
        out['mrwd'] = (anw * drm).astype(np.float32)
        out['mrwe'] = (anw * eim).astype(np.float32)
        out['mrae'] = (ana * eim).astype(np.float32)
        out['mcvr'] = (anv * (vvm | wvm | pf_mode)).astype(np.float32)
        out['mdwr'] = (anw * (vwm | drm | eim)).astype(np.float32)
        out['mevc'] = (np.nan_to_num(out['er_var'].to_numpy(np.float32), nan=0.0) * (vvm | wvm | pf_mode)).astype(np.int8)
        out['mewd'] = (np.nan_to_num(out['er_w'].to_numpy(np.float32), nan=0.0) * (vwm | drm | eim)).astype(np.int8)
        out['mtc'] = out[['mevc', 'mewd']].sum(axis=1).astype(np.int8)
        out['rtc'] = out[['tr_w', 'tr_va', 'tr_var', 'tr_pf', 'tr_a']].sum(axis=1).astype(np.int8)
        out['rec'] = out[['er_w', 'er_va', 'er_var', 'er_pf', 'er_a']].sum(axis=1).astype(np.int8)
        out['ruc'] = out[['ur_w', 'ur_va', 'ur_var', 'ur_pf', 'ur_a']].sum(axis=1).astype(np.int8)
        out['rqs'] = out[['q9r_w', 'q9r_va', 'q9r_var', 'q9r_pf', 'q9r_a']].sum(axis=1).astype(np.float32)
        return out

    def _eca(self):
        if C is None:
            raise RuntimeError('CatBoost is required for the full-data hybrid pipeline. Install dependencies from pyproject.toml before training.')

    def _ncm(self):
        self._eca()
        return C(iterations=self.cti, depth=self.cd, learning_rate=self.clr, loss_function='Logloss', eval_metric='Logloss', random_seed=self.seed, thread_count=self.nj, allow_writing_files=False, verbose=False)

    def _bta(self):
        shutil.rmtree(self.ad, ignore_errors=True)
        tr0 = self.ad / 'train'
        built = 0
        for ci, chunk in enumerate(self.irc('train.csv', UTR)):
            labels = chunk['Label'].astype(np.int8).to_numpy()
            feats = self.bf(chunk.drop(columns=['Label']))
            feats['Label'] = labels
            feats['fold_id'] = (feats['Id'].to_numpy(np.int64) % self.kf).astype(np.int8)
            feats['af'] = (_bsk(feats) % self.kf).astype(np.int8)
            for family in ('c0', 'c1', 'o'):
                fm = feats['df'] == family
                if fm.any():
                    fdir = tr0 / family
                    fdir.mkdir(parents=True, exist_ok=True)
                    fd = feats.loc[fm].copy()
                    fd.to_parquet(fdir / f'{ci:05d}.parquet', index=False)
                    built += len(fd)
            if ci % 10 == 0:
                print(f'[artifacts] materialized {built:,} training rows')
            del feats
            gc.collect()

    def _lfa(self, family, columns=None):
        fdir = self.ad / 'train' / family
        paths = sorted(fdir.glob('*.parquet'))
        if not paths:
            return pd.DataFrame(columns=list(columns or []))
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        return pd.concat(frames, ignore_index=True)

    def _roc(self, df):
        out = df.copy()
        hrf = np.column_stack([pd.to_numeric(out[RCM[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1 for name in HRN])
        if self.ovr:
            hof = np.column_stack([pd.to_numeric(out[RCM[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1 for name in self.ovr])
            out['oa'] = hof.any(axis=1).astype(np.int8)
        else:
            out['oa'] = np.zeros(len(out), dtype=np.int8)
        out['ra'] = hrf.any(axis=1).astype(np.int8)
        return out

    def _aho(self):
        cols = sorted({RCM[name] for name in OVR})
        counts = {name: [0, 0] for name in OVR}
        for family in ['c0', 'c1', 'o']:
            frame = self._lfa(family, columns=['Label', *cols])
            if frame.empty:
                continue
            labels = frame['Label'].to_numpy(np.int8)
            for name in OVR:
                mask = pd.to_numeric(frame[RCM[name]], errors='coerce').fillna(0).to_numpy(np.int8) == 1
                if mask.any():
                    counts[name][0] += int(mask.sum())
                    counts[name][1] += int(labels[mask].sum())
        self.ovr = [name for name, (count, positives) in counts.items() if count == 0 or positives / count >= MOP]

    def _csc(self):
        return (self.sc0, self.sgm, self.res_q, self.fbr, self.ssm, self.scm, self.sosm, self.socm)

    def _asc(self, cx):
        self.sc0, self.sgm, self.res_q, self.fbr, self.ssm, self.scm, self.sosm, self.socm = cx

    def _psf(self, bdf, y):
        work = self._roc(bdf)
        no_valid = pd.Series(np.zeros(len(work), dtype=bool), index=work.index)
        self._fsm(work, y, no_valid)
        work = self._aws(work)
        self._crq(work, y, no_valid)
        work = self._arf(work)
        work = self._fts(work, y)
        work = _afi(work)
        return (work, self._csc())

    def _pcf(self, bdf):
        out = self._roc(bdf)
        for col in [*SS.values(), 'dg']:
            if col in out.columns:
                out[col] = out[col].fillna('<NA>').astype(str)
        return out

    def _tso(self, sdf, y, fc, *, fc0, ff):
        probs = np.ones(len(sdf), dtype=np.float32)
        mm = sdf['oa'].to_numpy(np.int8) == 0
        fi = sdf[fc0].to_numpy(np.int8)
        fm = None
        for fold in range(self.kf):
            tm = mm & (fi != fold)
            vm = mm & (fi == fold)
            if not vm.any():
                continue
            model = self._ncf()
            x_train = sdf.loc[tm, fc]
            y_train = y[tm]
            weights = _bsw(sdf.loc[tm], y_train)
            model.fit(x_train, y_train, sample_weight=weights)
            probs[vm] = model.predict_proba(sdf.loc[vm, fc])[:, 1].astype(np.float32)
        if ff and mm.any():
            fm = self._ncf()
            weights = _bsw(sdf.loc[mm], y[mm])
            fm.fit(sdf.loc[mm, fc], y[mm], sample_weight=weights)
        return (probs, fm)

    def _tco(self, cat_df, y, fc, cc, *, fc0, ff):
        probs = np.ones(len(cat_df), dtype=np.float32)
        mm = cat_df['oa'].to_numpy(np.int8) == 0
        fi = cat_df[fc0].to_numpy(np.int8)
        fm = None
        for fold in range(self.kf):
            tm = mm & (fi != fold)
            vm = mm & (fi == fold)
            if not vm.any():
                continue
            model = self._ncm()
            weights = _bsw(cat_df.loc[tm], y[tm])
            model.fit(cat_df.loc[tm, fc], y[tm], cat_features=list(cc), sample_weight=weights, verbose=False)
            probs[vm] = model.predict_proba(cat_df.loc[vm, fc])[:, 1].astype(np.float32)
        if ff and mm.any():
            fm = self._ncm()
            weights = _bsw(cat_df.loc[mm], y[mm])
            fm.fit(cat_df.loc[mm, fc], y[mm], cat_features=list(cc), sample_weight=weights, verbose=False)
        return (probs, fm)

    def _sfb(self, y, ho, sp, sa, cp, ca):
        bpp = sp.copy()
        bpp[ho] = 1.0
        bth, _ = _tt(y, bpp)
        bpr = (bpp >= bth).astype(np.int8)
        bap = sa.copy()
        bap[ho] = 1.0
        bar = (bap >= bth).astype(np.int8)
        bps0 = float(fbeta_score(y, bpr, beta=2))
        bas0 = float(fbeta_score(y, bar, beta=2))
        bw = 1.0
        bt = bth
        bppd = bpr
        bapd = bar
        bps1 = bps0
        bafb = bas0
        wg = [round(step / 20.0, 2) for step in range(21)] if cp is not None else [1.0]
        for weight in wg:
            blp = _bp(sp, cp, weight)
            blp[ho] = 1.0
            thr, _ = _tt(y, blp)
            pp = (blp >= thr).astype(np.int8)
            bla = _bp(sa, ca, weight)
            bla[ho] = 1.0
            pa = (bla >= thr).astype(np.int8)
            ps = float(fbeta_score(y, pp, beta=2))
            as_ = float(fbeta_score(y, pa, beta=2))
            if as_ < bas0 - AT:
                continue
            if ps > bps1 + 1e-12 or (abs(ps - bps1) <= 1e-12 and as_ > bafb):
                bw = weight
                bt = thr
                bppd = pp
                bapd = pa
                bps1 = ps
                bafb = as_
        return (bw, bt, bppd.astype(np.int8), bapd.astype(np.int8))

    def fit(self):
        self._bta()
        self._aho()
        trained = 0
        for family in FAM:
            bdf = self._lfa(family)
            if bdf.empty:
                continue
            y_series = bdf['Label'].astype(np.int8)
            sdf, cx = self._psf(bdf.copy(), y_series)
            self.ctx[family] = cx
            semf = _snc(sdf, _sfc(sdf))
            self.sm0[family] = semf
            cat_df = self._pcf(bdf.copy())
            catf = _snc(cat_df, _cfc(cat_df))
            self.ccs[family] = catf
            catc = [col for col in [*SS.values(), 'dg'] if col in catf]
            y = y_series.to_numpy(np.int8)
            ho = sdf['oa'].to_numpy(np.int8) == 1
            sp1, sm = self._tso(sdf, y, semf, fc0='fold_id', ff=True)
            sa1, _ = self._tso(sdf, y, semf, fc0='af', ff=False)
            self.smd[family] = sm
            cp1 = ca1 = cm = None
            if catf:
                cp1, cm = self._tco(cat_df, y, catf, catc, fc0='fold_id', ff=True)
                ca1, _ = self._tco(cat_df, y, catf, catc, fc0='af', ff=False)
            self.cmd[family] = cm
            weight, thr, _, _ = self._sfb(y, ho, sp1, sa1, cp1, ca1)
            self.blend_w[family] = weight
            self.thr[family] = thr
            trained += 1
            print(f'[fit] {family} thr={thr:.3f}, blend_weight={weight:.2f}')
            del bdf, sdf, cat_df
            gc.collect()
        if not trained:
            raise RuntimeError('No training artifacts were avail for model fitting.')

    def _pfc(self, family, bdf):
        if family not in self.ctx or family not in self.smd:
            raise RuntimeError(f'Missing fitted semantic model bundle for family {family}.')
        cx = self.ctx[family]
        self._asc(cx)
        work = self._roc(bdf.copy())
        ho = work['oa'].to_numpy(np.int8) == 1
        sdf = self._aws(work.copy())
        sdf = self._arf(sdf)
        sdf = self._apf(sdf)
        sdf = _afi(sdf)
        sp = np.ones(len(sdf), dtype=np.float32)
        if (~ho).any():
            sm = self.smd[family]
            sp[~ho] = sm.predict_proba(sdf.loc[~ho, self.sm0[family]])[:, 1].astype(np.float32)
        cpb = None
        cm = self.cmd.get(family)
        if cm is not None and self.ccs.get(family):
            cat_df = self._pcf(bdf.copy())
            cpb = np.ones(len(cat_df), dtype=np.float32)
            if (~ho).any():
                cpb[~ho] = cm.predict_proba(cat_df.loc[~ho, self.ccs[family]])[:, 1].astype(np.float32)
        bp = _bp(sp, cpb, self.blend_w.get(family, 1.0))
        bp[ho] = 1.0
        pred = (bp >= self.thr.get(family, 0.5)).astype(np.int8)
        pred[ho] = 1
        return pred

    def pt(self, out_csv):
        if not self.smd:
            raise RuntimeError('Model is not fitted.')
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tr = 0
        pr = 0
        with out_csv.open('w', encoding='utf-8') as fh:
            fh.write('Id,Label\n')
            for ci, chunk in enumerate(self.irc('test.csv', UTE)):
                feats = self.bf(chunk)
                pred = feats['oa'].astype(np.int8).to_numpy()
                for family in FAM:
                    fm = feats['df'] == family
                    if fm.any():
                        pred[np.flatnonzero(fm.to_numpy())] = self._pfc(family, feats.loc[fm].copy())
                out = pd.DataFrame({'Id': feats['Id'].astype(np.int64), 'Label': pred.astype(np.int8)})
                out.to_csv(fh, index=False, header=False)
                tr += len(out)
                pr += int(out['Label'].sum())
                if ci % 10 == 0:
                    print(f'[test] wrote {tr:,} predictions')
        print(f'[test] done; tr={tr:,}, pr={pr:,}, positive_rate={pr / max(tr, 1):.6f}')

if __name__ == '__main__':
    s(DEFAULT_SEED);bl=R();bl.fit();out=DOD/'submission_full_data.csv';bl.pt(out);print(f'[solution] path={out}')
