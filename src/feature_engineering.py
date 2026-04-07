"""Feature construction for the DER anomaly baseline."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.contracts import (
    CANON1,
    CANON2,
    SQRT3,
)
from src.rules import (
    DEFAULT_HARD_OVERRIDE_NAMES,
    compute_hard_rule_outputs,
)
from src.schema import (
    BLOCK_SOURCE_COLUMNS,
    COMMON_STR,
    CURVE_FEATURE_SPECS,
    ENTER_SERVICE_COLUMNS,
    EXPECTED_MODEL_META,
    NUMERIC_SOURCE_COLUMNS,
    RAW_NUMERIC,
    RAW_STRING_COLUMNS,
    SAFE_RAW,
    SAFE_STR,
    TRIP_SPECS,
)

# These are the engineered columns that CatBoost consumes alongside raw strings.
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


def _nanextreme_rows(arr: np.ndarray, *, fill_value: float, use_max: bool) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    if arr.shape[1] == 0:
        return out
    reduced = np.where(mask, arr, fill_value)
    reduced = reduced.max(axis=1) if use_max else reduced.min(axis=1)
    valid_rows = mask.any(axis=1)
    out[valid_rows] = reduced[valid_rows]
    return out


def nanmin_rows(arr: np.ndarray) -> np.ndarray:
    return _nanextreme_rows(arr, fill_value=np.inf, use_max=False)


def nanmax_rows(arr: np.ndarray) -> np.ndarray:
    return _nanextreme_rows(arr, fill_value=-np.inf, use_max=True)


def nanmean_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    counts = mask.sum(axis=1)
    valid_rows = counts > 0
    if valid_rows.any():
        totals = np.where(mask, arr, 0.0).sum(axis=1)
        out[valid_rows] = totals[valid_rows] / counts[valid_rows]
    return out


def curve_index(raw_idx: np.ndarray, num_options: int) -> np.ndarray:
    idx = np.nan_to_num(np.asarray(raw_idx, dtype=np.float32), nan=1.0)
    idx = idx.astype(np.int16) - 1
    idx[(idx < 0) | (idx >= num_options)] = 0
    return idx.astype(np.int8)


def select_curve_scalar(curves: Sequence[np.ndarray], idx: np.ndarray) -> np.ndarray:
    stacked = np.stack(curves, axis=1)
    return np.take_along_axis(stacked, idx[:, None], axis=1)[:, 0]


def select_curve_points(curves: Sequence[np.ndarray], idx: np.ndarray) -> np.ndarray:
    stacked = np.stack(curves, axis=1)
    return np.take_along_axis(stacked, idx[:, None, None], axis=1)[:, 0, :]


def pair_point_count(x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
    return (np.isfinite(np.asarray(x_points, dtype=np.float32)) & np.isfinite(np.asarray(y_points, dtype=np.float32))).sum(axis=1).astype(np.int16)


def curve_reverse_steps(x_points: np.ndarray) -> np.ndarray:
    x_points = np.asarray(x_points, dtype=np.float32)
    finite_pair = np.isfinite(x_points[:, :-1]) & np.isfinite(x_points[:, 1:])
    return (((np.diff(x_points, axis=1) < -1e-6) & finite_pair).sum(axis=1)).astype(np.int8)


def curve_slope_stats(x_points: np.ndarray, y_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_points = np.asarray(x_points, dtype=np.float32)
    y_points = np.asarray(y_points, dtype=np.float32)
    dx = np.diff(x_points, axis=1)
    dy = np.diff(y_points, axis=1)
    valid = np.isfinite(x_points[:, :-1]) & np.isfinite(x_points[:, 1:]) & np.isfinite(y_points[:, :-1]) & np.isfinite(y_points[:, 1:]) & (np.abs(dx) > 1e-6)
    slopes = np.full(dx.shape, np.nan, dtype=np.float32)
    slopes[valid] = dy[valid] / dx[valid]
    return nanmean_rows(slopes), nanmax_rows(np.abs(slopes))


def piecewise_interp(x: np.ndarray, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
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


def var_pct(var: np.ndarray, varmaxinj: np.ndarray, varmaxabs: np.ndarray) -> np.ndarray:
    var = np.asarray(var, dtype=np.float32)
    denom = np.where(
        var >= 0,
        np.asarray(varmaxinj, dtype=np.float32),
        np.asarray(varmaxabs, dtype=np.float32),
    )
    return 100.0 * np.divide(
        var,
        denom,
        out=np.full_like(var, np.nan, dtype=np.float32),
        where=np.isfinite(var) & np.isfinite(denom) & (np.abs(denom) > 1e-6),
    )


def coerce_numeric(df: pd.DataFrame) -> None:
    for col in NUMERIC_SOURCE_COLUMNS:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _initialize_feature_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Capture the raw identity fields before the heavier numeric feature work."""

    fingerprint = df[COMMON_STR].fillna("<NA>").agg("|".join, axis=1)
    device_family = np.where(
        fingerprint == CANON1,
        "canon10",
        np.where(fingerprint == CANON2, "canon100", "other"),
    )
    return {
        "Id": df["Id"].to_numpy(),
        "device_fingerprint": fingerprint.to_numpy(dtype=object),
        "device_family": device_family,
        "common_missing_any": df[COMMON_STR].isna().any(axis=1).astype(np.int8).to_numpy(),
        "common_missing_count": df[COMMON_STR].isna().sum(axis=1).astype(np.int16).to_numpy(),
        "common_sn_has_decimal_suffix": df["common[0].SN"].fillna("").astype(str).str.endswith(".0").astype(np.int8).to_numpy(),
        "noncanonical": (device_family == "other").astype(np.int8),
    }


def _copy_raw_source_columns(data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
    """Mirror the selected raw columns using the sanitized downstream names."""

    for col in RAW_NUMERIC:
        arr = df[col].to_numpy()
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        data[SAFE_RAW[col]] = arr
    for col in RAW_STRING_COLUMNS:
        data[SAFE_STR[col]] = df[col].fillna("<NA>").astype(str).to_numpy(dtype=object)


def add_block_missingness(data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
    block_missing_total = np.zeros(len(df), dtype=np.int16)
    block_missing_any = np.zeros(len(df), dtype=np.int16)
    for block_name, cols in BLOCK_SOURCE_COLUMNS.items():
        missing = df[cols].isna()
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


def add_model_integrity_features(data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
    anomaly_sum = np.zeros(len(df), dtype=np.int16)
    missing_sum = np.zeros(len(df), dtype=np.int16)
    for block_name, (id_col, len_col, expected_id, expected_len) in EXPECTED_MODEL_META.items():
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


def add_capacity_extension_features(
    data: Dict[str, np.ndarray],
    *,
    wmaxrtg: np.ndarray,
    wmax: np.ndarray,
    vamaxrtg: np.ndarray,
    vamax: np.ndarray,
    varmaxinjrtg: np.ndarray,
    varmaxinj: np.ndarray,
    varmaxabsrtg: np.ndarray,
    varmaxabs: np.ndarray,
    vnomrtg: np.ndarray,
    vnom: np.ndarray,
    vmaxrtg: np.ndarray,
    vmax: np.ndarray,
    vminrtg: np.ndarray,
    vmin: np.ndarray,
    amaxrtg: np.ndarray,
    amax: np.ndarray,
    wcha_rtg: np.ndarray,
    wdis_rtg: np.ndarray,
    vacha_rtg: np.ndarray,
    vadis_rtg: np.ndarray,
    wcha: np.ndarray,
    wdis: np.ndarray,
    vacha: np.ndarray,
    vadis: np.ndarray,
    pfover_rtg: np.ndarray,
    pfover: np.ndarray,
    pfunder_rtg: np.ndarray,
    pfunder: np.ndarray,
) -> None:
    data["vnom_setting_delta"] = (vnom - vnomrtg).astype(np.float32)
    data["vmax_setting_delta"] = (vmax - vmaxrtg).astype(np.float32)
    data["vmin_setting_delta"] = (vmin - vminrtg).astype(np.float32)
    data["amax_setting_delta"] = (amax - amaxrtg).astype(np.float32)
    data["pfover_setting_delta"] = (pfover - pfover_rtg).astype(np.float32)
    data["pfunder_setting_delta"] = (pfunder - pfunder_rtg).astype(np.float32)
    for name, numerator, denominator in [
        ("charge_rate_share_rtg", wcha_rtg, wmaxrtg),
        ("discharge_rate_share_rtg", wdis_rtg, wmaxrtg),
        ("charge_va_share_rtg", vacha_rtg, vamaxrtg),
        ("discharge_va_share_rtg", vadis_rtg, vamaxrtg),
        ("charge_rate_share_setting", wcha, wmax),
        ("discharge_rate_share_setting", wdis, wmax),
        ("charge_va_share_setting", vacha, vamax),
        ("discharge_va_share_setting", vadis, vamax),
    ]:
        numerator = np.asarray(numerator, dtype=np.float32)
        denominator = np.asarray(denominator, dtype=np.float32)
        data[name] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float32),
            where=np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6),
        )
    rating_pairs = [
        (wmaxrtg, wmax),
        (vamaxrtg, vamax),
        (varmaxinjrtg, varmaxinj),
        (varmaxabsrtg, varmaxabs),
        (vnomrtg, vnom),
        (vmaxrtg, vmax),
        (vminrtg, vmin),
        (amaxrtg, amax),
    ]
    gap_count = np.zeros(len(wmaxrtg), dtype=np.int16)
    for rating, setting in rating_pairs:
        tol = np.maximum(1.0, 0.01 * np.nan_to_num(np.abs(rating), nan=0.0)).astype(np.float32)
        gap = np.isfinite(rating) & np.isfinite(setting) & (np.abs(setting - rating) > tol)
        gap_count += gap.astype(np.int16)
    data["rating_setting_gap_count"] = gap_count.astype(np.int8)


def add_temperature_features(data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
    temp_cols = [
        "DERMeasureAC[0].TmpAmb",
        "DERMeasureAC[0].TmpCab",
        "DERMeasureAC[0].TmpSnk",
        "DERMeasureAC[0].TmpTrns",
        "DERMeasureAC[0].TmpSw",
        "DERMeasureAC[0].TmpOt",
    ]
    temps = df[temp_cols].to_numpy(float)
    temp_min = nanmin_rows(temps)
    temp_max = nanmax_rows(temps)
    temp_mean = nanmean_rows(temps)
    amb = df["DERMeasureAC[0].TmpAmb"].to_numpy(float)
    data["temp_min"] = temp_min
    data["temp_max"] = temp_max
    data["temp_mean"] = temp_mean
    data["temp_spread"] = (temp_max - temp_min).astype(np.float32)
    data["temp_max_over_ambient"] = (temp_max - amb).astype(np.float32)


def add_enter_service_features(
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    *,
    voltage_pct: np.ndarray,
    hz: np.ndarray,
    abs_w: np.ndarray,
    va: np.ndarray,
    a: np.ndarray,
    tolw: np.ndarray,
    tolva: np.ndarray,
    amax: np.ndarray,
) -> None:
    es = df["DEREnterService[0].ES"].to_numpy(float)
    es_v_hi = df["DEREnterService[0].ESVHi"].to_numpy(float)
    es_v_lo = df["DEREnterService[0].ESVLo"].to_numpy(float)
    es_hz_hi = df["DEREnterService[0].ESHzHi"].to_numpy(float)
    es_hz_lo = df["DEREnterService[0].ESHzLo"].to_numpy(float)
    es_delay = df["DEREnterService[0].ESDlyTms"].to_numpy(float)
    es_random = df["DEREnterService[0].ESRndTms"].to_numpy(float)
    es_ramp = df["DEREnterService[0].ESRmpTms"].to_numpy(float)
    es_delay_rem = df["DEREnterService[0].ESDlyRemTms"].to_numpy(float)

    inside_v = np.isfinite(voltage_pct) & np.isfinite(es_v_hi) & np.isfinite(es_v_lo) & (voltage_pct >= es_v_lo) & (voltage_pct <= es_v_hi)
    inside_hz = np.isfinite(hz) & np.isfinite(es_hz_hi) & np.isfinite(es_hz_lo) & (hz >= es_hz_lo) & (hz <= es_hz_hi)
    inside_window = inside_v & inside_hz
    enabled = np.isfinite(es) & (es == 1.0)
    state_anomaly = np.isfinite(es) & (es >= 1.5)
    should_idle = (~enabled) | (~inside_window)
    current_tol = np.maximum(1.0, 0.02 * np.nan_to_num(amax, nan=0.0))

    data["enter_service_enabled"] = enabled.astype(np.int8)
    data["enter_service_state_anomaly"] = state_anomaly.astype(np.int8)
    data["enter_service_inside_window"] = inside_window.astype(np.int8)
    data["enter_service_outside_window"] = (~inside_window).astype(np.int8)
    data["enter_service_should_idle"] = should_idle.astype(np.int8)
    data["enter_service_v_window_width"] = (es_v_hi - es_v_lo).astype(np.float32)
    data["enter_service_hz_window_width"] = (es_hz_hi - es_hz_lo).astype(np.float32)
    data["enter_service_v_margin_low"] = (voltage_pct - es_v_lo).astype(np.float32)
    data["enter_service_v_margin_high"] = (es_v_hi - voltage_pct).astype(np.float32)
    data["enter_service_hz_margin_low"] = (hz - es_hz_lo).astype(np.float32)
    data["enter_service_hz_margin_high"] = (es_hz_hi - hz).astype(np.float32)
    data["enter_service_total_delay"] = (es_delay + es_random).astype(np.float32)
    data["enter_service_delay_remaining"] = es_delay_rem.astype(np.float32)
    data["enter_service_ramp_time"] = es_ramp.astype(np.float32)
    data["enter_service_delay_active"] = (np.nan_to_num(es_delay_rem, nan=0.0) > 0).astype(np.int8)

    blocked_power = should_idle & (abs_w > tolw)
    blocked_va = should_idle & (va > tolva)
    blocked_current = should_idle & (a > current_tol)
    data["enter_service_blocked_power"] = blocked_power.astype(np.int8)
    data["enter_service_blocked_va"] = blocked_va.astype(np.int8)
    data["enter_service_blocked_current"] = blocked_current.astype(np.int8)


def add_pf_control_features(
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    *,
    pf: np.ndarray,
    var: np.ndarray,
    varmaxinj: np.ndarray,
    varmaxabs: np.ndarray,
) -> None:
    pfinj_ena = np.nan_to_num(df["DERCtlAC[0].PFWInjEna"].to_numpy(float), nan=0.0)
    pfinj_ena_rvrt = np.nan_to_num(df["DERCtlAC[0].PFWInjEnaRvrt"].to_numpy(float), nan=0.0)
    pfabs_ena = np.nan_to_num(df["DERCtlAC[0].PFWAbsEna"].to_numpy(float), nan=0.0)
    pfabs_ena_rvrt = np.nan_to_num(df["DERCtlAC[0].PFWAbsEnaRvrt"].to_numpy(float), nan=0.0)
    pfinj_target = df["DERCtlAC[0].PFWInj.PF"].to_numpy(float)
    pfinj_rvrt_target = df["DERCtlAC[0].PFWInjRvrt.PF"].to_numpy(float)
    pfinj_ext = df["DERCtlAC[0].PFWInj.Ext"].to_numpy(float)
    pfinj_rvrt_ext = df["DERCtlAC[0].PFWInjRvrt.Ext"].to_numpy(float)
    pfabs_ext = df["DERCtlAC[0].PFWAbs.Ext"].to_numpy(float)
    pfabs_rvrt_ext = df["DERCtlAC[0].PFWAbsRvrt.Ext"].to_numpy(float)

    observed_var_pct = var_pct(var, varmaxinj, varmaxabs)
    inj_target_error = np.where(
        (pfinj_ena > 0) & np.isfinite(pfinj_target),
        np.abs(np.abs(pf) - pfinj_target),
        np.nan,
    )
    inj_rvrt_error = np.where(
        (pfinj_ena_rvrt > 0) & np.isfinite(pfinj_rvrt_target),
        np.abs(np.abs(pf) - pfinj_rvrt_target),
        np.nan,
    )
    data["pf_control_any_enabled"] = ((pfinj_ena > 0) | (pfabs_ena > 0)).astype(np.int8)
    data["pf_control_any_reversion"] = ((pfinj_ena_rvrt > 0) | (pfabs_ena_rvrt > 0)).astype(np.int8)
    data["pf_inj_target_error"] = inj_target_error.astype(np.float32)
    data["pf_inj_reversion_error"] = inj_rvrt_error.astype(np.float32)
    data["pf_inj_ext_present"] = np.isfinite(pfinj_ext).astype(np.int8)
    data["pf_inj_rvrt_ext_present"] = np.isfinite(pfinj_rvrt_ext).astype(np.int8)
    data["pf_abs_ext_present"] = np.isfinite(pfabs_ext).astype(np.int8)
    data["pf_abs_rvrt_ext_present"] = np.isfinite(pfabs_rvrt_ext).astype(np.int8)
    data["pf_inj_enabled_missing_target"] = ((pfinj_ena > 0) & ~np.isfinite(pfinj_target)).astype(np.int8)
    data["pf_reactive_near_limit"] = (np.abs(observed_var_pct) >= 95.0).astype(np.int8)


def add_trip_block_features(
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    *,
    short_name: str,
    prefix: str,
    axis_name: str,
    mode: str,
    measure_value: np.ndarray,
    abs_w: np.ndarray,
    tolw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    adpt_idx = curve_index(df[f"{prefix}.AdptCrvRslt"].to_numpy(float), 2)
    must_actpt = select_curve_scalar([df[f"{prefix}.Crv[{curve}].MustTrip.ActPt"].to_numpy(float) for curve in range(2)], adpt_idx)
    mom_actpt = select_curve_scalar([df[f"{prefix}.Crv[{curve}].MomCess.ActPt"].to_numpy(float) for curve in range(2)], adpt_idx)
    must_x = select_curve_points(
        [np.column_stack([df[f"{prefix}.Crv[{curve}].MustTrip.Pt[{point}].{axis_name}"].to_numpy(float) for point in range(5)]) for curve in range(2)],
        adpt_idx,
    )
    must_t = select_curve_points(
        [np.column_stack([df[f"{prefix}.Crv[{curve}].MustTrip.Pt[{point}].Tms"].to_numpy(float) for point in range(5)]) for curve in range(2)],
        adpt_idx,
    )
    mom_x = select_curve_points(
        [np.column_stack([df[f"{prefix}.Crv[{curve}].MomCess.Pt[{point}].{axis_name}"].to_numpy(float) for point in range(5)]) for curve in range(2)],
        adpt_idx,
    )
    mom_t = select_curve_points(
        [np.column_stack([df[f"{prefix}.Crv[{curve}].MomCess.Pt[{point}].Tms"].to_numpy(float) for point in range(5)]) for curve in range(2)],
        adpt_idx,
    )
    may_present = np.column_stack([df[f"{prefix}.Crv[{curve}].MayTrip.Pt[{point}].{axis_name}"].to_numpy(float) for curve in range(2) for point in range(5)])

    enabled = np.nan_to_num(df[f"{prefix}.Ena"].to_numpy(float), nan=0.0) > 0
    must_count = pair_point_count(must_x, must_t)
    mom_count = pair_point_count(mom_x, mom_t)
    must_x_min = nanmin_rows(must_x)
    must_x_max = nanmax_rows(must_x)
    must_t_min = nanmin_rows(must_t)
    must_t_max = nanmax_rows(must_t)
    mom_x_min = nanmin_rows(mom_x)
    mom_x_max = nanmax_rows(mom_x)
    mom_t_min = nanmin_rows(mom_t)
    mom_t_max = nanmax_rows(mom_t)

    if mode == "low":
        margin = measure_value - must_x_max
    else:
        margin = must_x_min - measure_value
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
    data[f"trip_{short_name}_musttrip_reverse_steps"] = curve_reverse_steps(must_x)
    data[f"trip_{short_name}_momcess_count"] = mom_count
    data[f"trip_{short_name}_momcess_actpt_gap"] = (mom_actpt - mom_count).astype(np.float32)
    data[f"trip_{short_name}_momcess_axis_span"] = (mom_x_max - mom_x_min).astype(np.float32)
    data[f"trip_{short_name}_momcess_tms_span"] = (mom_t_max - mom_t_min).astype(np.float32)
    data[f"trip_{short_name}_momcess_reverse_steps"] = curve_reverse_steps(mom_x)
    data[f"trip_{short_name}_maytrip_present_any"] = np.isfinite(may_present).any(axis=1).astype(np.int8)
    data[f"trip_{short_name}_musttrip_margin"] = margin.astype(np.float32)
    data[f"trip_{short_name}_outside_musttrip"] = outside.astype(np.int8)
    data[f"trip_{short_name}_power_when_outside"] = power_when_outside.astype(np.int8)
    data[f"trip_{short_name}_momcess_musttrip_gap"] = envelope_gap.astype(np.float32)
    return outside.astype(np.int8), power_when_outside.astype(np.int8)


def add_curve_block_features(
    data: Dict[str, np.ndarray],
    *,
    name: str,
    raw_idx: np.ndarray,
    curve_x: Sequence[np.ndarray],
    curve_y: Sequence[np.ndarray],
    curve_actpt: Sequence[np.ndarray],
    curve_meta: Dict[str, Sequence[np.ndarray]],
    measure_value: np.ndarray,
    observed_value: Optional[np.ndarray] = None,
) -> None:
    adpt_idx = curve_index(raw_idx, len(curve_x))
    selected_x = select_curve_points(curve_x, adpt_idx)
    selected_y = select_curve_points(curve_y, adpt_idx)
    selected_actpt = select_curve_scalar(curve_actpt, adpt_idx)
    data[f"{name}_curve_idx"] = adpt_idx.astype(np.int8)
    point_count = pair_point_count(selected_x, selected_y)
    data[f"{name}_curve_point_count"] = point_count
    data[f"{name}_curve_actpt_gap"] = (selected_actpt - point_count).astype(np.float32)
    x_min = nanmin_rows(selected_x)
    x_max = nanmax_rows(selected_x)
    y_min = nanmin_rows(selected_y)
    y_max = nanmax_rows(selected_y)
    mean_slope, max_abs_slope = curve_slope_stats(selected_x, selected_y)
    data[f"{name}_curve_x_span"] = (x_max - x_min).astype(np.float32)
    data[f"{name}_curve_y_span"] = (y_max - y_min).astype(np.float32)
    data[f"{name}_curve_reverse_steps"] = curve_reverse_steps(selected_x)
    data[f"{name}_curve_mean_slope"] = mean_slope
    data[f"{name}_curve_max_abs_slope"] = max_abs_slope
    data[f"{name}_curve_measure_margin_low"] = (measure_value - x_min).astype(np.float32)
    data[f"{name}_curve_measure_margin_high"] = (x_max - measure_value).astype(np.float32)
    if observed_value is not None:
        expected_value = piecewise_interp(measure_value, selected_x, selected_y)
        data[f"{name}_curve_expected"] = expected_value.astype(np.float32)
        data[f"{name}_curve_error"] = (observed_value - expected_value).astype(np.float32)
    for meta_name, curves in curve_meta.items():
        data[f"{name}_curve_{meta_name}"] = select_curve_scalar(curves, adpt_idx).astype(np.float32)


def add_freq_droop_features(
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    *,
    hz: np.ndarray,
    w_pct: np.ndarray,
) -> None:
    raw_idx = df["DERFreqDroop[0].AdptCtlRslt"].to_numpy(float)
    ctl_idx = curve_index(raw_idx, 3)
    dbof_curves = [df[f"DERFreqDroop[0].Ctl[{i}].DbOf"].to_numpy(float) for i in range(3)]
    dbuf_curves = [df[f"DERFreqDroop[0].Ctl[{i}].DbUf"].to_numpy(float) for i in range(3)]
    kof_curves = [df[f"DERFreqDroop[0].Ctl[{i}].KOf"].to_numpy(float) for i in range(3)]
    kuf_curves = [df[f"DERFreqDroop[0].Ctl[{i}].KUf"].to_numpy(float) for i in range(3)]
    rsp_curves = [df[f"DERFreqDroop[0].Ctl[{i}].RspTms"].to_numpy(float) for i in range(3)]
    pmin_curves = [df[f"DERFreqDroop[0].Ctl[{i}].PMin"].to_numpy(float) for i in range(3)]
    ro_curves = [df[f"DERFreqDroop[0].Ctl[{i}].ReadOnly"].to_numpy(float) for i in range(3)]
    dbof = select_curve_scalar(dbof_curves, ctl_idx)
    dbuf = select_curve_scalar(dbuf_curves, ctl_idx)
    kof = select_curve_scalar(kof_curves, ctl_idx)
    kuf = select_curve_scalar(kuf_curves, ctl_idx)
    rsp = select_curve_scalar(rsp_curves, ctl_idx)
    pmin = select_curve_scalar(pmin_curves, ctl_idx)
    readonly = select_curve_scalar(ro_curves, ctl_idx)

    over_activation = np.maximum(hz - (60.0 + dbof), 0.0)
    under_activation = np.maximum((60.0 - dbuf) - hz, 0.0)
    over_activation = np.asarray(over_activation, dtype=np.float32)
    under_activation = np.asarray(under_activation, dtype=np.float32)
    kof = np.asarray(kof, dtype=np.float32)
    kuf = np.asarray(kuf, dtype=np.float32)
    expected_delta_pct = 100.0 * np.divide(
        over_activation,
        kof,
        out=np.full_like(over_activation, np.nan, dtype=np.float32),
        where=np.isfinite(over_activation) & np.isfinite(kof) & (np.abs(kof) > 1e-6),
    ) - 100.0 * np.divide(
        under_activation,
        kuf,
        out=np.full_like(under_activation, np.nan, dtype=np.float32),
        where=np.isfinite(under_activation) & np.isfinite(kuf) & (np.abs(kuf) > 1e-6),
    )
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
    data["freqdroop_db_span"] = (nanmax_rows(np.column_stack([dbof_stack, dbuf_stack])) - nanmin_rows(np.column_stack([dbof_stack, dbuf_stack]))).astype(np.float32)
    data["freqdroop_k_span"] = (nanmax_rows(k_stack) - nanmin_rows(k_stack)).astype(np.float32)
    data["freqdroop_pmin_span"] = (nanmax_rows(pmin_stack) - nanmin_rows(pmin_stack)).astype(np.float32)


def add_dc_features(
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    *,
    w: np.ndarray,
    abs_w: np.ndarray,
) -> None:
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

    dcw32 = np.asarray(dcw, dtype=np.float32)
    w32 = np.asarray(w, dtype=np.float32)
    abs_w32 = np.asarray(abs_w, dtype=np.float32)
    data["dcw_over_w"] = np.divide(
        dcw32,
        w32,
        out=np.full_like(dcw32, np.nan, dtype=np.float32),
        where=np.isfinite(dcw32) & np.isfinite(w32) & (np.abs(w32) > 1e-6),
    )
    data["dcw_over_abs_w"] = np.divide(
        dcw32,
        abs_w32,
        out=np.full_like(dcw32, np.nan, dtype=np.float32),
        where=np.isfinite(dcw32) & np.isfinite(abs_w32) & (np.abs(abs_w32) > 1e-6),
    )
    data["dcw_minus_port_sum"] = (dcw - (prt0 + prt1)).astype(np.float32)
    data["dcv_spread"] = np.abs(prt0_v - prt1_v).astype(np.float32)
    data["dca_spread"] = np.abs(prt0_a - prt1_a).astype(np.float32)
    prt032 = np.asarray(prt0, dtype=np.float32)
    prt0_plus_prt1 = np.asarray(prt0 + prt1, dtype=np.float32)
    data["dc_port0_share"] = np.divide(
        prt032,
        prt0_plus_prt1,
        out=np.full_like(prt032, np.nan, dtype=np.float32),
        where=np.isfinite(prt032) & np.isfinite(prt0_plus_prt1) & (np.abs(prt0_plus_prt1) > 1e-6),
    )
    data["dc_port_type_mismatch"] = (np.isfinite(prt0_t) & np.isfinite(prt1_t) & (prt0_t != prt1_t)).astype(np.int8)
    rare_type = (prt0_t == 7) | (prt1_t == 7)
    data["dc_port_type_rare_any"] = rare_type.astype(np.int8)
    data["ac_zero_dc_positive"] = ((np.abs(w) <= 1e-6) & (dcw > 0)).astype(np.int8)
    data["ac_positive_dc_zero"] = ((w > 0) & (np.abs(dcw) <= 1e-6)).astype(np.int8)
    data["ac_dc_same_sign"] = (np.sign(np.nan_to_num(w, nan=0.0)) == np.sign(np.nan_to_num(dcw, nan=0.0))).astype(np.int8)
    dca32 = np.asarray(dca, dtype=np.float32)
    prt_total_a = np.asarray(prt0_a + prt1_a, dtype=np.float32)
    data["dca_over_total"] = np.divide(
        dca32,
        prt_total_a,
        out=np.full_like(dca32, np.nan, dtype=np.float32),
        where=np.isfinite(dca32) & np.isfinite(prt_total_a) & (np.abs(prt_total_a) > 1e-6),
    )


def _add_adaptive_curve_features(
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    *,
    voltage_pct: np.ndarray,
    reactive_pct: np.ndarray,
    w_pct: np.ndarray,
) -> None:
    """Bind live measurements to the schema-owned adaptive curve layouts."""

    runtime_measurements = {
        "voltvar": (
            voltage_pct - 100.0 + df["DERVoltVar[0].Crv[0].VRef"].fillna(100.0).to_numpy(float),
            reactive_pct,
        ),
        "voltwatt": (voltage_pct, w_pct),
        "wattvar": (w_pct, reactive_pct),
    }
    for name, spec in CURVE_FEATURE_SPECS.items():
        measure_value, observed_value = runtime_measurements[name]
        add_curve_block_features(
            data,
            name=name,
            raw_idx=df[f"{spec.prefix}.AdptCrvRslt"].to_numpy(float),
            curve_x=[np.column_stack([df[f"{spec.prefix}.Crv[{curve}].Pt[{point}].{spec.x_field}"].to_numpy(float) for point in range(spec.point_count)]) for curve in range(3)],
            curve_y=[np.column_stack([df[f"{spec.prefix}.Crv[{curve}].Pt[{point}].{spec.y_field}"].to_numpy(float) for point in range(spec.point_count)]) for curve in range(3)],
            curve_actpt=[df[f"{spec.prefix}.Crv[{curve}].ActPt"].to_numpy(float) for curve in range(3)],
            curve_meta={meta_name: [df[f"{spec.prefix}.Crv[{curve}].{field}"].to_numpy(float) for curve in range(3)] for meta_name, field in spec.meta_fields.items()},
            measure_value=measure_value,
            observed_value=observed_value,
        )


def _apply_hard_rule_features(
    data: Dict[str, np.ndarray],
    *,
    hard_override_names: Sequence[str],
) -> None:
    """Derive aggregate hard-rule signals from the centralized rule metadata."""

    outputs = compute_hard_rule_outputs(data, hard_override_names)
    data["hard_override_anomaly"] = outputs.hard_override_anomaly
    data["hard_rule_count"] = outputs.hard_rule_count
    data["hard_rule_score"] = outputs.hard_rule_score
    data["hard_rule_anomaly"] = outputs.hard_rule_anomaly


def build_features(df: pd.DataFrame, hard_override_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if hard_override_names is None:
        hard_override_names = DEFAULT_HARD_OVERRIDE_NAMES

    def block_values(prefix: str, fields: Sequence[str]) -> Dict[str, np.ndarray]:
        return {field: df[f"{prefix}.{field}"].to_numpy(float) for field in fields}

    # The feature frame is built in ordered phases so downstream model code sees
    # the same columns and dtypes as the original monolithic implementation.
    coerce_numeric(df)

    data = _initialize_feature_data(df)
    _copy_raw_source_columns(data, df)

    # Missingness and structural integrity are cheap signals, so populate them
    # before the heavier physical-control feature set below.
    add_block_missingness(data, df)
    add_model_integrity_features(data, df)
    add_temperature_features(data, df)

    measure_ac = block_values(
        "DERMeasureAC[0]",
        ["W", "VA", "Var", "PF", "A", "LLV", "LNV", "Hz", "ACType"],
    )
    capacity = block_values(
        "DERCapacity[0]",
        [
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
        ],
    )
    ctl_ac = block_values(
        "DERCtlAC[0]",
        ["WSetEna", "WSet", "WSetPct", "WMaxLimPctEna", "WMaxLimPct", "VarSetEna", "VarSet", "VarSetPct"],
    )
    phase_ac = block_values(
        "DERMeasureAC[0]",
        ["WL1", "WL2", "WL3", "VAL1", "VAL2", "VAL3", "VarL1", "VarL2", "VarL3", "VL1L2", "VL2L3", "VL3L1", "VL1", "VL2", "VL3"],
    )
    capacity_ext = block_values(
        "DERCapacity[0]",
        [
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
        ],
    )

    w = measure_ac["W"]
    abs_w = np.abs(w)
    va = measure_ac["VA"]
    var = measure_ac["Var"]
    pf = measure_ac["PF"]
    a = measure_ac["A"]
    llv = measure_ac["LLV"]
    lnv = measure_ac["LNV"]
    lnv_scaled = np.asarray(lnv * SQRT3, dtype=np.float32)
    hz = measure_ac["Hz"]

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

    for name, numerator, denominator in [
        ("w_over_wmaxrtg", w, wmaxrtg),
        ("w_over_wmax", w, wmax),
        ("va_over_vamax", va, vamax),
        ("va_over_vamaxrtg", va, vamaxrtg),
        ("var_over_injmax", var, varmaxinj),
        ("var_over_absmax", var, varmaxabs),
        ("a_over_amax", a, amax),
        ("llv_over_vnom", llv, vnom),
        ("lnv_over_vnom", lnv * SQRT3, vnom),
    ]:
        numerator = np.asarray(numerator, dtype=np.float32)
        denominator = np.asarray(denominator, dtype=np.float32)
        data[name] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float32),
            where=np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6),
        )

    for name, value in [
        ("w_minus_wmax", w - wmax),
        ("w_minus_wmaxrtg", w - wmaxrtg),
        ("va_minus_vamax", va - vamax),
        ("var_minus_injmax", var - varmaxinj),
        ("var_plus_absmax", var + varmaxabs),
        ("llv_minus_lnv_sqrt3", llv - lnv * SQRT3),
        ("hz_delta_60", hz - 60.0),
    ]:
        data[name] = value.astype(np.float32)

    for name, left, right in [
        ("w_eq_wmaxrtg", w, wmaxrtg),
        ("w_eq_wmax", w, wmax),
        ("var_eq_varmaxinj", var, varmaxinj),
        ("var_eq_neg_varmaxabs", var, -varmaxabs),
    ]:
        data[name] = np.isclose(left, right, equal_nan=False).astype(np.int8)
    data["pf_sign_mismatch"] = ((np.sign(np.nan_to_num(pf)) != np.sign(np.nan_to_num(w))) & (np.nan_to_num(pf) != 0) & (np.nan_to_num(w) != 0)).astype(np.int8)

    tolw = np.maximum(50.0, 0.02 * np.nan_to_num(wmaxrtg, nan=0.0)).astype(np.float32)
    tolva = np.maximum(50.0, 0.02 * np.nan_to_num(vamax, nan=0.0)).astype(np.float32)
    tolvi = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxinj, nan=0.0)).astype(np.float32)
    tolva2 = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxabs, nan=0.0)).astype(np.float32)
    for name, value, upper_bound in [
        ("w_gt_wmax_tol", w, wmax + tolw),
        ("w_gt_wmaxrtg_tol", w, wmaxrtg + tolw),
        ("va_gt_vamax_tol", va, vamax + tolva),
        ("var_gt_injmax_tol", var, varmaxinj + tolvi),
    ]:
        data[name] = (value > upper_bound).astype(np.int8)
    data["var_lt_absmax_tol"] = (var < (-varmaxabs - tolva2)).astype(np.int8)

    pq = np.sqrt(np.square(w.astype(np.float32)) + np.square(var.astype(np.float32)))
    data["va_minus_pqmag"] = (va - pq).astype(np.float32)
    va32 = np.asarray(va, dtype=np.float32)
    pq = np.asarray(pq, dtype=np.float32)
    data["va_over_pqmag"] = np.divide(
        va32,
        pq,
        out=np.full_like(va32, np.nan, dtype=np.float32),
        where=np.isfinite(va32) & np.isfinite(pq) & (np.abs(pq) > 1e-6),
    )
    w32 = np.asarray(w, dtype=np.float32)
    pf_from_w_va = np.divide(
        w32,
        va32,
        out=np.full_like(w32, np.nan, dtype=np.float32),
        where=np.isfinite(w32) & np.isfinite(va32) & (np.abs(va32) > 1e-6),
    )
    data["pf_from_w_va"] = pf_from_w_va
    data["pf_error"] = (pf - pf_from_w_va).astype(np.float32)

    for name, total, suffixes in [
        ("w_phase_sum_error", w, ["WL1", "WL2", "WL3"]),
        ("va_phase_sum_error", va, ["VAL1", "VAL2", "VAL3"]),
        ("var_phase_sum_error", var, ["VarL1", "VarL2", "VarL3"]),
    ]:
        phase_sum = sum(phase_ac[suffix] for suffix in suffixes)
        data[name] = (total - phase_sum).astype(np.float32)
    for name, suffixes in [
        ("phase_ll_spread", ["VL1L2", "VL2L3", "VL3L1"]),
        ("phase_ln_spread", ["VL1", "VL2", "VL3"]),
        ("phase_w_spread", ["WL1", "WL2", "WL3"]),
        ("phase_var_spread", ["VarL1", "VarL2", "VarL3"]),
    ]:
        phase_values = np.column_stack([phase_ac[suffix] for suffix in suffixes])
        data[name] = (nanmax_rows(phase_values) - nanmin_rows(phase_values)).astype(np.float32)

    for name, numerator, denominator in [
        ("wmax_over_wmaxrtg", wmax, wmaxrtg),
        ("vamax_over_vamaxrtg", vamax, vamaxrtg),
        ("vmax_over_vnom", vmax, vnom),
        ("vmin_over_vnom", vmin, vnom),
    ]:
        numerator = np.asarray(numerator, dtype=np.float32)
        denominator = np.asarray(denominator, dtype=np.float32)
        data[name] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float32),
            where=np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6),
        )

    wsetena = np.nan_to_num(ctl_ac["WSetEna"], nan=0.0)
    wset = ctl_ac["WSet"]
    wsetpct = ctl_ac["WSetPct"]
    wmaxlimena = np.nan_to_num(ctl_ac["WMaxLimPctEna"], nan=0.0)
    wmaxlimpct = ctl_ac["WMaxLimPct"]
    varsetena = np.nan_to_num(ctl_ac["VarSetEna"], nan=0.0)
    varset = ctl_ac["VarSet"]
    varsetpct = ctl_ac["VarSetPct"]
    wset_abs_error = np.where(wsetena > 0, np.abs(w - wset), np.nan)
    wsetpct_target = wmaxrtg * (wsetpct / 100.0)
    wsetpct_abs_error = np.where(wsetena > 0, np.abs(w - wsetpct_target), np.nan)
    wmaxlim_target = wmaxrtg * (wmaxlimpct / 100.0)
    wmaxlim_excess = np.where(wmaxlimena > 0, w - wmaxlim_target, np.nan)
    varset_abs_error = np.where(varsetena > 0, np.abs(var - varset), np.nan)
    varsetpct_target = varmaxinj * (varsetpct / 100.0)
    varsetpct_abs_error = np.where(varsetena > 0, np.abs(var - varsetpct_target), np.nan)
    for name, value in [
        ("wset_abs_error", wset_abs_error),
        ("wsetpct_target", wsetpct_target),
        ("wsetpct_abs_error", wsetpct_abs_error),
        ("wmaxlim_target", wmaxlim_target),
        ("wmaxlim_excess", wmaxlim_excess),
        ("varset_abs_error", varset_abs_error),
        ("varsetpct_target", varsetpct_target),
        ("varsetpct_abs_error", varsetpct_abs_error),
    ]:
        data[name] = value.astype(np.float32)
    for name, value in [
        ("wset_enabled_far", (wsetena > 0) & (wset_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))),
        ("wsetpct_enabled_far", (wsetena > 0) & (wsetpct_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))),
        ("wmaxlim_enabled_far", (wmaxlimena > 0) & (wmaxlim_excess > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))),
        ("varsetpct_enabled_far", (varsetena > 0) & (varsetpct_abs_error > np.maximum(20.0, 0.05 * np.nan_to_num(varmaxinj, nan=0.0)))),
    ]:
        data[name] = value.astype(np.int8)

    add_capacity_extension_features(
        data,
        wmaxrtg=wmaxrtg,
        wmax=wmax,
        vamaxrtg=vamaxrtg,
        vamax=vamax,
        varmaxinjrtg=varmaxinjrtg,
        varmaxinj=varmaxinj,
        varmaxabsrtg=varmaxabsrtg,
        varmaxabs=varmaxabs,
        vnomrtg=capacity_ext["VNomRtg"],
        vnom=vnom,
        vmaxrtg=capacity_ext["VMaxRtg"],
        vmax=vmax,
        vminrtg=capacity_ext["VMinRtg"],
        vmin=vmin,
        amaxrtg=capacity_ext["AMaxRtg"],
        amax=amax,
        wcha_rtg=capacity_ext["WChaRteMaxRtg"],
        wdis_rtg=capacity_ext["WDisChaRteMaxRtg"],
        vacha_rtg=capacity_ext["VAChaRteMaxRtg"],
        vadis_rtg=capacity_ext["VADisChaRteMaxRtg"],
        wcha=capacity_ext["WChaRteMax"],
        wdis=capacity_ext["WDisChaRteMax"],
        vacha=capacity_ext["VAChaRteMax"],
        vadis=capacity_ext["VADisChaRteMax"],
        pfover_rtg=capacity_ext["PFOvrExtRtg"],
        pfover=capacity_ext["PFOvrExt"],
        pfunder_rtg=capacity_ext["PFUndExtRtg"],
        pfunder=capacity_ext["PFUndExt"],
    )

    llv = np.asarray(llv, dtype=np.float32)
    lnv = np.asarray(lnv, dtype=np.float32)
    vnom = np.asarray(vnom, dtype=np.float32)
    wmaxrtg = np.asarray(wmaxrtg, dtype=np.float32)
    voltage_pct = 100.0 * np.divide(
        llv,
        vnom,
        out=np.full_like(llv, np.nan, dtype=np.float32),
        where=np.isfinite(llv) & np.isfinite(vnom) & (np.abs(vnom) > 1e-6),
    )
    line_neutral_voltage_pct = 100.0 * np.divide(
        lnv_scaled,
        vnom,
        out=np.full_like(lnv_scaled, np.nan, dtype=np.float32),
        where=np.isfinite(lnv_scaled) & np.isfinite(vnom) & (np.abs(vnom) > 1e-6),
    )
    w_pct = 100.0 * np.divide(
        w32,
        wmaxrtg,
        out=np.full_like(w32, np.nan, dtype=np.float32),
        where=np.isfinite(w32) & np.isfinite(wmaxrtg) & (np.abs(wmaxrtg) > 1e-6),
    )
    reactive_pct = var_pct(var, varmaxinj, varmaxabs)

    for name, value in [
        ("voltage_pct", voltage_pct),
        ("line_neutral_voltage_pct", line_neutral_voltage_pct),
        ("w_pct_of_rtg", w_pct),
        ("var_pct_of_limit", reactive_pct),
    ]:
        data[name] = value.astype(np.float32)

    add_enter_service_features(
        data,
        df,
        voltage_pct=voltage_pct,
        hz=hz,
        abs_w=abs_w,
        va=va,
        a=a,
        tolw=tolw,
        tolva=tolva,
        amax=amax,
    )
    add_pf_control_features(
        data,
        df,
        pf=pf,
        var=var,
        varmaxinj=varmaxinj,
        varmaxabs=varmaxabs,
    )

    trip_outside_flags = []
    trip_power_flags = []
    for short_name, (prefix, axis_name, mode) in TRIP_SPECS.items():
        measure_value = voltage_pct if axis_name == "V" else hz
        outside, power_when_outside = add_trip_block_features(
            data,
            df,
            short_name=short_name,
            prefix=prefix,
            axis_name=axis_name,
            mode=mode,
            measure_value=measure_value,
            abs_w=abs_w,
            tolw=tolw,
        )
        trip_outside_flags.append(outside)
        trip_power_flags.append(power_when_outside)
    if trip_outside_flags:
        trip_any_outside = np.column_stack(trip_outside_flags).any(axis=1).astype(np.int8)
        trip_any_power_when_outside = np.column_stack(trip_power_flags).any(axis=1).astype(np.int8)
    else:
        trip_any_outside = np.zeros(len(df), dtype=np.int8)
        trip_any_power_when_outside = np.zeros(len(df), dtype=np.int8)
    data["trip_any_outside_musttrip"] = trip_any_outside
    data["trip_any_power_when_outside"] = trip_any_power_when_outside

    _add_adaptive_curve_features(
        data,
        df,
        voltage_pct=voltage_pct,
        reactive_pct=reactive_pct,
        w_pct=w_pct,
    )

    add_freq_droop_features(data, df, hz=hz, w_pct=w_pct)
    add_dc_features(data, df, w=w, abs_w=abs_w)

    ac_type = measure_ac["ACType"]
    data["ac_type_is_rare"] = (np.isfinite(ac_type) & (ac_type == 3.0)).astype(np.int8)

    _apply_hard_rule_features(data, hard_override_names=hard_override_names)
    return pd.DataFrame(data)
