from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .constants import *  # noqa: F403
from .pipeline import ResearchBaseline


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
    def _build_identity_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        common_str = df[COMMON_STR]
        fingerprint = common_str.fillna("<NA>").agg("|".join, axis=1)
        device_family = np.where(
            fingerprint == CANON1,
            "canon10",
            np.where(fingerprint == CANON2, "canon100", "other"),
        )
        return {
            "Id": df["Id"].to_numpy(),
            "device_fingerprint": fingerprint.to_numpy(dtype=object),
            "device_family": device_family,
            "common_missing_any": common_str.isna().any(axis=1).astype(np.int8).to_numpy(),
            "common_missing_count": common_str.isna().sum(axis=1).astype(np.int16).to_numpy(),
            "common_sn_has_decimal_suffix": df["common[0].SN"].fillna("").astype(str).str.endswith(".0").astype(np.int8).to_numpy(),
            "noncanonical": (device_family == "other").astype(np.int8),
        }

    @staticmethod
    def _copy_raw_source_features(data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
        for col in RAW_NUMERIC:
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            data[SAFE_RAW[col]] = arr
        for col in RAW_STRING_COLUMNS:
            data[SAFE_STR[col]] = df[col].fillna("<NA>").astype(str).to_numpy(dtype=object)

    @staticmethod
    def _build_rule_flag_map(
        data: Dict[str, np.ndarray],
        *,
        ac_type_is_rare: np.ndarray,
        dc_port_type_rare: np.ndarray,
        gate_flags: Dict[str, np.ndarray],
        trip_any_power_when_outside: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        return {
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

    @staticmethod
    def _hard_rule_score(float_flags: Dict[str, np.ndarray]) -> np.ndarray:
        return (
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

    def _assign_hard_rule_features(
        self,
        data: Dict[str, np.ndarray],
        flag_map: Dict[str, np.ndarray],
    ) -> None:
        hard_rule_flags = np.column_stack([flag_map[name] for name in HARD_RULE_NAMES])
        hard_override_flags = np.column_stack([flag_map[name] for name in self.hard_override_names])
        float_flags = {name: flag.astype(np.float32) for name, flag in flag_map.items()}
        data["hard_rule_count"] = hard_rule_flags.sum(axis=1).astype(np.int8)
        data["hard_rule_score"] = self._hard_rule_score(float_flags)
        data["hard_rule_anomaly"] = hard_rule_flags.any(axis=1).astype(np.int8)
        data["hard_override_anomaly"] = hard_override_flags.any(axis=1).astype(np.int8)

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

        data = self._build_identity_features(df)
        self._copy_raw_source_features(data, df)

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

        flag_map = self._build_rule_flag_map(
            data,
            ac_type_is_rare=ac_type_is_rare,
            dc_port_type_rare=dc_port_type_rare,
            gate_flags=gate_flags,
            trip_any_power_when_outside=trip_any_power_when_outside,
        )
        self._assign_hard_rule_features(data, flag_map)
        return pd.DataFrame(data)


__all__ = ["FeatureBuilder"]
