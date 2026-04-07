"""Model training and inference for the DER anomaly baseline."""

import gc
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

from src.contracts import (
    AUDIT_TOLERANCE,
    BaselineConfig,
    CANON100_INTERACTION_FEATURES,
    CANON100_NEGATIVE_WEIGHT,
    DEVICE_FAMILY_MAP,
    FAMILY_THRESHOLD_FLOOR,
    FamilyModelBundle,
    FamilySemanticContext,
    HARD_OVERRIDE_TRAIN_WEIGHT,
    MAX_THRESHOLD,
    MetricSummary,
    MIN_OVERRIDE_PRECISION,
    RESIDUAL_TAIL_FALLBACKS,
    RESIDUAL_TAIL_LEVELS,
    SCENARIO_SMOOTHING,
    ScenarioStats,
    SURROGATE_LEAKY_FEATURES,
    SURROGATE_TARGETS,
)
from src.feature_engineering import CAT_ENGINEERED_COLUMNS, build_features
from src.rules import (
    DEFAULT_HARD_OVERRIDE_NAMES,
    build_named_rule_flags,
    compute_active_override_anomaly,
)
from src.schema import (
    RAW_NUMERIC,
    RAW_STRING_COLUMNS,
    SAFE_RAW,
    SAFE_STR,
    USECOLS_TEST,
    USECOLS_TRAIN,
    dedupe,
)

LOGGER = logging.getLogger(__name__)


class ResearchBaseline:
    """Own the learned per-family model bundles and inference stack."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        # Keep fixed model knobs grouped together so fit-time state is easier to
        # distinguish from learned model state.
        self.config = config or BaselineConfig()
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        """Reset learned state while keeping immutable config untouched."""

        self.active_hard_override_names = list(DEFAULT_HARD_OVERRIDE_NAMES)
        self.family_models: Dict[str, FamilyModelBundle] = {}
        self.semantic_context = FamilySemanticContext()

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
            chunksize=self.config.chunksize,
            low_memory=False,
        ):
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

    def _encode_device_family(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["device_family"] = out["device_family"].map(DEVICE_FAMILY_MAP).fillna(-1).astype(np.int8)
        return out

    def _get_surrogate_feature_cols(self, columns: Sequence[str]) -> List[str]:
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

    def _build_sample_weights(self, x_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
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
        out = pd.to_numeric(values, errors="coerce")
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

    def _build_scenario_keys(self, x_df: pd.DataFrame, *, include_output_bins: bool = False) -> np.ndarray:
        return pd.util.hash_pandas_object(self._build_scenario_frame(x_df, include_output_bins=include_output_bins), index=False).to_numpy(np.uint64)

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

    @staticmethod
    def _group_key_stats(keys: np.ndarray, y_values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"key": keys, "y": y_values}).groupby("key")["y"].agg(["sum", "count"])

    def _family_prior(self, family_series: pd.Series, global_rate: float) -> np.ndarray:
        return family_series.map(self.semantic_context.family_base_rates).fillna(global_rate).to_numpy(np.float32)

    @staticmethod
    def _smoothed_rate(
        sum_values: np.ndarray,
        count_values: np.ndarray,
        prior: np.ndarray,
    ) -> np.ndarray:
        return (sum_values + SCENARIO_SMOOTHING * prior) / (count_values + SCENARIO_SMOOTHING)

    def _fit_transform_scenario_features(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        out = x_train.copy()
        y_arr = y_train.to_numpy(np.float32)
        family_series = out["device_family"].astype(str)
        self.semantic_context.family_base_rates = pd.DataFrame({"family": family_series, "y": y_arr}).groupby("family")["y"].mean().to_dict()
        keys = self._build_scenario_keys(out)
        output_keys = self._build_scenario_keys(out, include_output_bins=True)
        fold_ids = (out["Id"].to_numpy(np.int64) % self.config.cv_folds).astype(np.int8)
        scenario_rate = np.zeros(len(out), dtype=np.float32)
        scenario_count = np.zeros(len(out), dtype=np.int32)
        scenario_output_rate = np.zeros(len(out), dtype=np.float32)
        scenario_output_count = np.zeros(len(out), dtype=np.int32)
        global_rate = float(np.mean(y_arr))
        family_base_rates = self.semantic_context.family_base_rates

        for fold in range(self.config.cv_folds):
            train_mask = fold_ids != fold
            valid_mask = fold_ids == fold
            if not valid_mask.any():
                continue
            stats = self._group_key_stats(keys[train_mask], y_arr[train_mask])
            output_stats = self._group_key_stats(output_keys[train_mask], y_arr[train_mask])
            valid_keys = pd.Series(keys[valid_mask])
            valid_sum = valid_keys.map(stats["sum"]).fillna(0.0).to_numpy(np.float32)
            valid_count = valid_keys.map(stats["count"]).fillna(0).to_numpy(np.int32)
            valid_output_keys = pd.Series(output_keys[valid_mask])
            valid_output_sum = valid_output_keys.map(output_stats["sum"]).fillna(0.0).to_numpy(np.float32)
            valid_output_count = valid_output_keys.map(output_stats["count"]).fillna(0).to_numpy(np.int32)
            valid_family = family_series.loc[valid_mask].tolist()
            prior = np.array(
                [family_base_rates.get(name, global_rate) for name in valid_family],
                dtype=np.float32,
            )
            scenario_rate[valid_mask] = self._smoothed_rate(valid_sum, valid_count, prior)
            scenario_count[valid_mask] = valid_count
            scenario_output_rate[valid_mask] = self._smoothed_rate(
                valid_output_sum,
                valid_output_count,
                prior,
            )
            scenario_output_count[valid_mask] = valid_output_count

        full_stats = self._group_key_stats(keys, y_arr)
        full_output_stats = self._group_key_stats(output_keys, y_arr)
        self.semantic_context.scenario_stats = ScenarioStats(
            sum_map={int(idx): float(val) for idx, val in full_stats["sum"].items()},
            count_map={int(idx): int(val) for idx, val in full_stats["count"].items()},
            output_sum_map={int(idx): float(val) for idx, val in full_output_stats["sum"].items()},
            output_count_map={int(idx): int(val) for idx, val in full_output_stats["count"].items()},
        )

        family_prior = self._family_prior(family_series, global_rate)
        return self._assign_scenario_features(
            out,
            family_prior=family_prior,
            scenario_rate=scenario_rate,
            scenario_count=scenario_count,
            scenario_output_rate=scenario_output_rate,
            scenario_output_count=scenario_output_count,
        )

    def _apply_scenario_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        scenario_stats = self.semantic_context.scenario_stats
        if not scenario_stats.count_map:
            return x_df

        out = x_df.copy()
        keys = self._build_scenario_keys(out)
        output_keys = self._build_scenario_keys(out, include_output_bins=True)
        sum_values, count_values = self._lookup_scenario_stats(
            keys,
            scenario_stats.sum_map,
            scenario_stats.count_map,
        )
        output_sum_values, output_count_values = self._lookup_scenario_stats(
            output_keys,
            scenario_stats.output_sum_map,
            scenario_stats.output_count_map,
        )
        family_base_rates = self.semantic_context.family_base_rates
        global_rate = float(np.mean(list(family_base_rates.values()))) if family_base_rates else 0.5
        family_prior = self._family_prior(out["device_family"].astype(str), global_rate)
        scenario_rate = self._smoothed_rate(sum_values, count_values, family_prior)
        scenario_output_rate = self._smoothed_rate(
            output_sum_values,
            output_count_values,
            family_prior,
        )
        return self._assign_scenario_features(
            out,
            family_prior=family_prior,
            scenario_rate=scenario_rate,
            scenario_count=count_values,
            scenario_output_rate=scenario_output_rate,
            scenario_output_count=output_count_values,
        )

    def _add_family_interaction_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        out = x_df.copy()
        canon100_mask = out["device_family"].astype(str) == "canon100"
        for feature_name in CANON100_INTERACTION_FEATURES:
            if feature_name not in out.columns:
                continue
            values = pd.to_numeric(out[feature_name], errors="coerce").to_numpy(np.float32)
            out[f"canon100_{feature_name}"] = np.where(canon100_mask.to_numpy(), values, 0.0).astype(np.float32)
        return out

    def _surrogate_partition_mask(self, ids: Sequence[int], *, fit_partition: bool) -> np.ndarray:
        ids_arr = np.asarray(ids, dtype=np.int64)
        fit_mask = (ids_arr % 2) == 0
        return fit_mask if fit_partition else ~fit_mask

    @staticmethod
    def _normal_training_mask(
        x_df: pd.DataFrame,
        y_train: pd.Series,
        valid_mask: pd.Series,
    ) -> np.ndarray:
        return (y_train.to_numpy(np.int8) == 0) & (x_df["hard_override_anomaly"].to_numpy(np.int8) == 0) & (x_df["device_family"].to_numpy(dtype=object) != "other") & (~valid_mask.to_numpy())

    @staticmethod
    def _initialize_residual_columns(out: pd.DataFrame) -> None:
        for target_name in SURROGATE_TARGETS:
            out[f"pred_{target_name}"] = np.nan
            out[f"resid_{target_name}"] = np.nan
            out[f"abs_resid_{target_name}"] = np.nan
            out[f"norm_resid_{target_name}"] = np.nan
            out[f"abs_norm_resid_{target_name}"] = np.nan
            out[f"tail_resid_{target_name}"] = 0
            out[f"extreme_resid_{target_name}"] = 0
            out[f"ultra_resid_{target_name}"] = 0
            out[f"q99_ratio_resid_{target_name}"] = np.nan

    @staticmethod
    def _curve_mode(
        out: pd.DataFrame,
        *,
        enabled_col: str,
        expected_col: str,
    ) -> np.ndarray:
        enabled = np.nan_to_num(out[enabled_col].to_numpy(np.float32), nan=0.0) > 0
        expected = out[expected_col].to_numpy(np.float32)
        return enabled & np.isfinite(expected)

    @staticmethod
    def _cat_categorical_cols(feature_cols: Sequence[str]) -> List[str]:
        categorical_candidates = [*SAFE_STR.values(), "device_fingerprint"]
        return [col for col in categorical_candidates if col in feature_cols]

    def _fit_surrogate_models(self, x_train: pd.DataFrame, y_train: pd.Series, valid_mask: pd.Series) -> None:
        self.semantic_context.surrogate_feature_cols = self._get_surrogate_feature_cols(x_train.columns)
        fit_partition = self._surrogate_partition_mask(x_train["Id"], fit_partition=True)
        normal_mask = self._normal_training_mask(x_train, y_train, valid_mask) & fit_partition
        surrogate_df = x_train.loc[normal_mask].copy()
        if surrogate_df.empty:
            raise RuntimeError("No rows available to train surrogate models.")

        self.semantic_context.surrogate_models = {}
        model_params = {
            "n_estimators": max(80, self.config.n_estimators // 2),
            "max_depth": max(4, self.config.max_depth - 2),
            "learning_rate": min(0.08, self.config.learning_rate * 1.2),
            "objective": "reg:squarederror",
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_jobs": self.config.n_jobs,
            "random_state": self.config.seed,
            "seed": self.config.seed,
            "verbosity": 0,
        }
        for family in DEVICE_FAMILY_MAP:
            family_df = surrogate_df.loc[surrogate_df["device_family"] == family].copy()
            if family_df.empty:
                continue
            x_surrogate = self._encode_device_family(family_df[self.semantic_context.surrogate_feature_cols])
            for target_name, (target_col, _) in SURROGATE_TARGETS.items():
                model = XGBRegressor(**model_params)
                y_target = family_df[target_col].to_numpy(np.float32)
                LOGGER.info(f"[surrogate] training {family}/{target_name} on {len(family_df):,} normal rows")
                model.fit(x_surrogate, y_target)
                self.semantic_context.surrogate_models[(family, target_name)] = model

    def _augment_with_surrogates(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.semantic_context.surrogate_feature_cols or not self.semantic_context.surrogate_models:
            return x_df

        out = x_df.copy()
        self._initialize_residual_columns(out)
        x_surrogate = self._encode_device_family(out[self.semantic_context.surrogate_feature_cols])
        for family in DEVICE_FAMILY_MAP:
            family_mask = out["device_family"] == family
            if not family_mask.any():
                continue
            x_family = x_surrogate.loc[family_mask]
            for target_name, (target_col, scale_col) in SURROGATE_TARGETS.items():
                model = self.semantic_context.surrogate_models.get((family, target_name))
                if model is None:
                    continue
                pred = model.predict(x_family).astype(np.float32)
                actual = out.loc[family_mask, target_col].to_numpy(np.float32)
                resid = actual - pred
                out.loc[family_mask, f"pred_{target_name}"] = pred
                out.loc[family_mask, f"resid_{target_name}"] = resid
                out.loc[family_mask, f"abs_resid_{target_name}"] = np.abs(resid).astype(np.float32)
                if scale_col is not None:
                    scale = out.loc[family_mask, scale_col].to_numpy(np.float32)
                    norm_resid = np.divide(
                        resid,
                        scale,
                        out=np.full_like(resid, np.nan, dtype=np.float32),
                        where=np.isfinite(resid) & np.isfinite(scale) & (np.abs(scale) > 1e-6),
                    )
                else:
                    scale = np.maximum(0.05, np.abs(actual))
                    norm_resid = (resid / scale).astype(np.float32)
                out.loc[family_mask, f"norm_resid_{target_name}"] = norm_resid.astype(np.float32)
                out.loc[family_mask, f"abs_norm_resid_{target_name}"] = np.abs(norm_resid).astype(np.float32)

        abs_resid_cols = [
            "abs_resid_w",
            "abs_resid_va",
            "abs_resid_var",
            "abs_resid_pf",
            "abs_resid_a",
        ]
        out["resid_energy_total"] = out[abs_resid_cols].sum(axis=1).astype(np.float32)
        out["resid_va_minus_pq"] = (out["pred_va"] - np.sqrt(np.square(out["pred_w"]) + np.square(out["pred_var"]))).astype(np.float32)
        abs_resid_w = out["abs_resid_w"].to_numpy(np.float32)
        abs_resid_var = out["abs_resid_var"].to_numpy(np.float32) + 1e-3
        out["resid_w_var_ratio"] = np.divide(
            abs_resid_w,
            abs_resid_var,
            out=np.full_like(abs_resid_w, np.nan, dtype=np.float32),
            where=np.isfinite(abs_resid_w) & np.isfinite(abs_resid_var) & (np.abs(abs_resid_var) > 1e-6),
        )
        return out

    def _compute_residual_quantiles(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        valid_mask: pd.Series,
    ) -> None:
        calibration_partition = self._surrogate_partition_mask(x_train["Id"], fit_partition=False)
        base_mask = self._normal_training_mask(x_train, y_train, valid_mask)
        self.semantic_context.residual_quantiles = {}
        for family in DEVICE_FAMILY_MAP:
            family_mask = base_mask & (x_train["device_family"] == family)
            family_calibration = family_mask & calibration_partition
            if not family_calibration.any():
                family_calibration = family_mask
            family_quantiles: Dict[str, Dict[str, float]] = {}
            for target_name in SURROGATE_TARGETS:
                series = x_train.loc[family_calibration, f"abs_norm_resid_{target_name}"]
                values = pd.to_numeric(series, errors="coerce").to_numpy(np.float32)
                values = values[np.isfinite(values)]
                quantiles = RESIDUAL_TAIL_FALLBACKS.copy()
                if values.size > 0:
                    for level_name, q in RESIDUAL_TAIL_LEVELS.items():
                        quantiles[level_name] = float(np.quantile(values, q))
                family_quantiles[target_name] = {key: max(1e-6, value) for key, value in quantiles.items()}
            self.semantic_context.residual_quantiles[family] = family_quantiles

    def _apply_residual_calibration_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        residual_quantiles = self.semantic_context.residual_quantiles
        if not residual_quantiles:
            return x_df

        out = x_df.copy()
        calibration_levels = (
            ("tail", 0),
            ("extreme", 0),
            ("ultra", 0),
            ("q99_ratio", np.nan),
        )
        for target_name in SURROGATE_TARGETS:
            for prefix, default_value in calibration_levels:
                out[f"{prefix}_resid_{target_name}"] = default_value

        for family in DEVICE_FAMILY_MAP:
            family_mask = out["device_family"] == family
            if not family_mask.any():
                continue
            family_quantiles = residual_quantiles.get(family, {})
            for target_name in SURROGATE_TARGETS:
                abs_norm = out.loc[family_mask, f"abs_norm_resid_{target_name}"].to_numpy(np.float32)
                q = family_quantiles.get(target_name, RESIDUAL_TAIL_FALLBACKS)
                calibrated_values = (
                    (f"tail_resid_{target_name}", abs_norm >= q["tail"], np.int8),
                    (f"extreme_resid_{target_name}", abs_norm >= q["extreme"], np.int8),
                    (f"ultra_resid_{target_name}", abs_norm >= q["ultra"], np.int8),
                    (
                        f"q99_ratio_resid_{target_name}",
                        np.divide(
                            abs_norm,
                            np.full_like(abs_norm, q["extreme"], dtype=np.float32),
                            out=np.full_like(abs_norm, np.nan, dtype=np.float32),
                            where=np.isfinite(abs_norm) & np.isfinite(q["extreme"]) & (abs(q["extreme"]) > 1e-6),
                        ),
                        np.float32,
                    ),
                )
                for column_name, values, dtype in calibrated_values:
                    out.loc[family_mask, column_name] = values.astype(dtype)

        abs_norm_values = {
            "w": np.nan_to_num(out["abs_norm_resid_w"].to_numpy(np.float32), nan=0.0),
            "var": np.nan_to_num(out["abs_norm_resid_var"].to_numpy(np.float32), nan=0.0),
            "pf": np.nan_to_num(out["abs_norm_resid_pf"].to_numpy(np.float32), nan=0.0),
            "a": np.nan_to_num(out["abs_norm_resid_a"].to_numpy(np.float32), nan=0.0),
        }
        mode_flags = {
            "pf": np.nan_to_num(out["pf_control_any_enabled"].to_numpy(np.float32), nan=0.0) > 0,
            "voltvar": self._curve_mode(
                out,
                enabled_col="DERVoltVar_0_Ena",
                expected_col="voltvar_curve_expected",
            ),
            "voltwatt": self._curve_mode(
                out,
                enabled_col="DERVoltWatt_0_Ena",
                expected_col="voltwatt_curve_expected",
            ),
            "wattvar": self._curve_mode(
                out,
                enabled_col="DERWattVar_0_Ena",
                expected_col="wattvar_curve_expected",
            ),
            "droop": np.nan_to_num(out["freqdroop_outside_deadband"].to_numpy(np.float32), nan=0.0) > 0,
            "enter_idle": np.nan_to_num(out["enter_service_should_idle"].to_numpy(np.float32), nan=0.0) > 0,
        }
        mode_flags["curve_var"] = mode_flags["voltvar"] | mode_flags["wattvar"] | mode_flags["pf"]
        mode_flags["dispatch"] = mode_flags["voltwatt"] | mode_flags["droop"] | mode_flags["enter_idle"]
        extreme_residuals = {
            "var": np.nan_to_num(out["extreme_resid_var"].to_numpy(np.float32), nan=0.0),
            "w": np.nan_to_num(out["extreme_resid_w"].to_numpy(np.float32), nan=0.0),
        }

        mode_residual_specs = (
            ("mode_resid_pf_pf", "pf", "pf"),
            ("mode_resid_var_pf", "var", "pf"),
            ("mode_resid_var_voltvar", "var", "voltvar"),
            ("mode_resid_w_voltwatt", "w", "voltwatt"),
            ("mode_resid_var_wattvar", "var", "wattvar"),
            ("mode_resid_w_droop", "w", "droop"),
            ("mode_resid_w_enter_idle", "w", "enter_idle"),
            ("mode_resid_a_enter_idle", "a", "enter_idle"),
            ("mode_curve_var_resid", "var", "curve_var"),
            ("mode_dispatch_w_resid", "w", "dispatch"),
        )
        for column_name, residual_name, mode_name in mode_residual_specs:
            out[column_name] = (abs_norm_values[residual_name] * mode_flags[mode_name]).astype(np.float32)

        extreme_mode_specs = (
            ("mode_extreme_var_curve", "var", "curve_var"),
            ("mode_extreme_w_dispatch", "w", "dispatch"),
        )
        for column_name, residual_name, mode_name in extreme_mode_specs:
            out[column_name] = (extreme_residuals[residual_name] * mode_flags[mode_name]).astype(np.int8)

        out["mode_tail_count"] = out[["mode_extreme_var_curve", "mode_extreme_w_dispatch"]].sum(axis=1).astype(np.int8)
        count_specs = (
            ("resid_tail_count", "tail"),
            ("resid_extreme_count", "extreme"),
            ("resid_ultra_count", "ultra"),
        )
        for column_name, prefix in count_specs:
            out[column_name] = out[[f"{prefix}_resid_{name}" for name in SURROGATE_TARGETS]].sum(axis=1).astype(np.int8)
        out["resid_quantile_score"] = out[[f"q99_ratio_resid_{name}" for name in SURROGATE_TARGETS]].sum(axis=1).astype(np.float32)
        return out

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
        best_thr, best_f2 = 0.5, -1.0
        thresholds = np.arange(low, high + 1e-9, step, dtype=np.float32)
        for thr in thresholds:
            pred = (prob >= thr).astype(np.int8)
            score = fbeta_score(y_true, pred, beta=2)
            if score > best_f2:
                best_thr, best_f2 = float(thr), float(score)
        return best_thr, best_f2

    @staticmethod
    def _metric_summary_from_pred(y_true: np.ndarray, pred: np.ndarray) -> MetricSummary:
        if len(y_true) == 0:
            return MetricSummary(f2=0.0, precision=0.0, recall=0.0, positive_rate=0.0, rows=0)
        return MetricSummary(
            f2=float(fbeta_score(y_true, pred, beta=2)),
            precision=float(precision_score(y_true, pred, zero_division=0)),
            recall=float(recall_score(y_true, pred, zero_division=0)),
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

    def _build_train_chunk_features(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        labels = chunk["Label"].astype(np.int8).to_numpy()
        features = build_features(
            chunk.drop(columns=["Label"]),
            hard_override_names=DEFAULT_HARD_OVERRIDE_NAMES,
        )
        return features, labels

    def _audit_hard_override_rules(self, train_path: Path) -> Tuple[Dict[str, int], List[str], pd.DataFrame]:
        row_counts = {"canon10": 0, "canon100": 0, "other": 0}
        total_positive = 0
        total_rows = 0
        per_rule_counts = {name: {"count": 0, "positives": 0} for name in DEFAULT_HARD_OVERRIDE_NAMES}
        other_frames: List[pd.DataFrame] = []

        LOGGER.info("[fit] auditing hard override rules from %s", train_path)
        with tqdm(
            self.iter_raw_chunks(train_path, USECOLS_TRAIN, 0),
            desc="audit chunks",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            for chunk in progress:
                feats, labels = self._build_train_chunk_features(chunk)
                rule_flags = build_named_rule_flags(feats)
                total_rows += len(feats)
                total_positive += int(labels.sum())
                family_values = feats["device_family"]
                for family in row_counts:
                    family_mask = family_values == family
                    family_count = int(family_mask.sum())
                    if family_count == 0:
                        continue
                    row_counts[family] += family_count
                    if family == "other":
                        other_idx = family_mask.to_numpy()
                        other_frames.append(
                            pd.DataFrame(
                                {
                                    "Id": feats.loc[family_mask, "Id"].to_numpy(np.int64),
                                    "Label": labels[other_idx],
                                }
                            )
                        )
                for rule_name in DEFAULT_HARD_OVERRIDE_NAMES:
                    mask = rule_flags[rule_name]
                    if not mask.any():
                        continue
                    per_rule_counts[rule_name]["count"] += int(mask.sum())
                    per_rule_counts[rule_name]["positives"] += int(labels[mask].sum())
                progress.set_postfix(rows=f"{total_rows:,}", positives=f"{total_positive:,}")
                del feats
                gc.collect()

        demoted: List[str] = []
        for rule_name, counts in per_rule_counts.items():
            count = counts["count"]
            positives = counts["positives"]
            precision = float(positives / count) if count else 1.0
            recall = float(positives / total_positive) if total_positive else 0.0
            LOGGER.info(
                "[fit] hard override rule=%s count=%s positives=%s precision=%.4f recall=%.4f",
                rule_name,
                f"{count:,}",
                f"{positives:,}",
                precision,
                recall,
            )
            if count > 0 and precision < MIN_OVERRIDE_PRECISION:
                demoted.append(rule_name)
        self.active_hard_override_names = [name for name in DEFAULT_HARD_OVERRIDE_NAMES if name not in demoted]
        LOGGER.info("[fit] hard override audit active=%s demoted=%s", self.active_hard_override_names, demoted)
        other_df = pd.concat(other_frames, ignore_index=True) if other_frames else pd.DataFrame(columns=["Id", "Label"])
        return row_counts, demoted, other_df

    def _load_family_training_frame(self, train_path: Path, family: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        total_rows = 0
        LOGGER.info("[fit] materializing %s rows from %s", family, train_path)
        with tqdm(
            self.iter_raw_chunks(train_path, USECOLS_TRAIN, 0),
            desc=f"{family} chunks",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            for chunk in progress:
                feats, labels = self._build_train_chunk_features(chunk)
                family_mask = feats["device_family"] == family
                if not family_mask.any():
                    continue
                family_idx = family_mask.to_numpy()
                family_df = feats.loc[family_mask].copy()
                family_df["Label"] = labels[family_idx]
                family_df["fold_id"] = (family_df["Id"].to_numpy(np.int64) % self.config.cv_folds).astype(np.int8)
                family_df["audit_fold_id"] = (self._build_scenario_keys(family_df) % self.config.cv_folds).astype(np.int8)
                frames.append(family_df)
                total_rows += len(family_df)
                progress.set_postfix(rows=f"{total_rows:,}")
                del feats
                gc.collect()
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _refresh_override_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rule_flags = build_named_rule_flags(out)
        row_count = len(out)
        out["hard_override_anomaly"] = compute_active_override_anomaly(
            out,
            self.active_hard_override_names,
            rule_flags=rule_flags,
            row_count=row_count,
        )
        out["hard_rule_anomaly"] = compute_active_override_anomaly(
            out,
            DEFAULT_HARD_OVERRIDE_NAMES,
            rule_flags=rule_flags,
            row_count=row_count,
        )
        return out

    def _capture_semantic_context(self) -> FamilySemanticContext:
        return FamilySemanticContext(
            surrogate_feature_cols=list(self.semantic_context.surrogate_feature_cols),
            surrogate_models=dict(self.semantic_context.surrogate_models),
            residual_quantiles=deepcopy(self.semantic_context.residual_quantiles),
            family_base_rates=dict(self.semantic_context.family_base_rates),
            scenario_stats=ScenarioStats(
                sum_map=dict(self.semantic_context.scenario_stats.sum_map),
                count_map=dict(self.semantic_context.scenario_stats.count_map),
                output_sum_map=dict(self.semantic_context.scenario_stats.output_sum_map),
                output_count_map=dict(self.semantic_context.scenario_stats.output_count_map),
            ),
        )

    def _activate_semantic_context(self, context: FamilySemanticContext) -> None:
        self.semantic_context = deepcopy(context)

    def _apply_semantic_transforms(
        self,
        augmented_df: pd.DataFrame,
        *,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        out = self._apply_residual_calibration_features(augmented_df)
        if y is None:
            out = self._apply_scenario_features(out)
        else:
            out = self._fit_transform_scenario_features(out, y)
        return self._add_family_interaction_features(out)

    def _prepare_semantic_frame(self, base_df: pd.DataFrame) -> pd.DataFrame:
        work = self._refresh_override_columns(base_df)
        work = self._augment_with_surrogates(work)
        return self._apply_semantic_transforms(work)

    def _prepare_family_semantic_frame(
        self,
        base_df: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, FamilySemanticContext]:
        work = self._refresh_override_columns(base_df)
        no_valid = pd.Series(np.zeros(len(work), dtype=bool), index=work.index)
        self._fit_surrogate_models(work, y, no_valid)
        work = self._augment_with_surrogates(work)
        self._compute_residual_quantiles(work, y, no_valid)
        work = self._apply_semantic_transforms(work, y=y)
        return work, self._capture_semantic_context()

    def _semantic_feature_candidates(self, semantic_df: pd.DataFrame) -> List[str]:
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

    def _cat_feature_candidates(self, cat_df: pd.DataFrame) -> List[str]:
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
        final_model: Optional[XGBClassifier] = None
        model_params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "objective": "binary:logistic",
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": self.config.n_jobs,
            "random_state": self.config.seed,
            "seed": self.config.seed,
            "verbosity": 1,
        }
        for fold in range(self.config.cv_folds):
            train_mask = model_mask & (fold_ids != fold)
            valid_mask = model_mask & (fold_ids == fold)
            if not valid_mask.any():
                continue
            model = XGBClassifier(**model_params)
            x_train = semantic_df.loc[train_mask, feature_cols]
            y_train = y[train_mask]
            weights = self._build_sample_weights(semantic_df.loc[train_mask], y_train)
            model.fit(x_train, y_train, sample_weight=weights)
            probs[valid_mask] = model.predict_proba(semantic_df.loc[valid_mask, feature_cols])[:, 1].astype(np.float32)
        if fit_final and model_mask.any():
            final_model = XGBClassifier(**model_params)
            weights = self._build_sample_weights(semantic_df.loc[model_mask], y[model_mask])
            final_model.fit(
                semantic_df.loc[model_mask, feature_cols],
                y[model_mask],
                sample_weight=weights,
            )
        return probs, final_model

    def _train_cat_oof(
        self,
        cat_df: pd.DataFrame,
        y: np.ndarray,
        feature_cols: Sequence[str],
        categorical_cols: Sequence[str],
        *,
        fold_col: str,
        fit_final: bool,
    ) -> Tuple[np.ndarray, Optional[CatBoostClassifier]]:
        probs = np.ones(len(cat_df), dtype=np.float32)
        model_mask = cat_df["hard_override_anomaly"].to_numpy(np.int8) == 0
        fold_ids = cat_df[fold_col].to_numpy(np.int8)
        final_model: Optional[CatBoostClassifier] = None
        cat_features = list(categorical_cols)
        model_params = {
            "iterations": self.config.cat_iterations,
            "depth": self.config.cat_depth,
            "learning_rate": self.config.cat_learning_rate,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": self.config.seed,
            "thread_count": self.config.n_jobs,
            "allow_writing_files": False,
            "verbose": False,
        }
        for fold in range(self.config.cv_folds):
            train_mask = model_mask & (fold_ids != fold)
            valid_mask = model_mask & (fold_ids == fold)
            if not valid_mask.any():
                continue
            model = CatBoostClassifier(**model_params)
            weights = self._build_sample_weights(cat_df.loc[train_mask], y[train_mask])
            model.fit(
                cat_df.loc[train_mask, feature_cols],
                y[train_mask],
                cat_features=cat_features,
                sample_weight=weights,
                verbose=False,
            )
            probs[valid_mask] = model.predict_proba(cat_df.loc[valid_mask, feature_cols])[:, 1].astype(np.float32)
        if fit_final and model_mask.any():
            final_model = CatBoostClassifier(**model_params)
            weights = self._build_sample_weights(cat_df.loc[model_mask], y[model_mask])
            final_model.fit(
                cat_df.loc[model_mask, feature_cols],
                y[model_mask],
                cat_features=cat_features,
                sample_weight=weights,
                verbose=False,
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
            improved_primary = primary_score > best_primary_score + 1e-12
            tied_primary = abs(primary_score - best_primary_score) <= 1e-12
            improved_audit = audit_score > best_audit_for_best
            if improved_primary or (tied_primary and improved_audit):
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

    def fit(self, train_path: Path) -> "ResearchBaseline":
        self._reset_fit_state()
        row_counts, demoted_rules, other_df = self._audit_hard_override_rules(train_path)
        LOGGER.info("[fit] training row counts=%s", row_counts)
        primary_metrics: Dict[str, MetricSummary] = {}
        audit_metrics: Dict[str, MetricSummary] = {}
        prediction_rows: List[pd.DataFrame] = []

        LOGGER.info("[fit] starting family training")
        with tqdm(
            list(DEVICE_FAMILY_MAP),
            desc="fit families",
            unit="family",
            dynamic_ncols=True,
        ) as family_progress:
            for family in family_progress:
                family_progress.set_postfix(family=family)
                if row_counts[family] == 0:
                    LOGGER.info("[fit] skipping %s; no training rows", family)
                    continue
                base_df = self._load_family_training_frame(train_path, family)
                if base_df.empty:
                    LOGGER.info("[fit] skipping %s; no training rows", family)
                    continue
                LOGGER.info("[fit] training %s on %s rows", family, f"{len(base_df):,}")
                y_series = base_df["Label"].astype(np.int8)
                semantic_df, context = self._prepare_family_semantic_frame(base_df.copy(), y_series)
                semantic_feature_cols = self._select_nonconstant_columns(
                    semantic_df,
                    self._semantic_feature_candidates(semantic_df),
                )

                cat_df = self._prepare_cat_frame(base_df.copy())
                cat_feature_cols = self._select_nonconstant_columns(cat_df, self._cat_feature_candidates(cat_df))
                cat_categorical_cols = self._cat_categorical_cols(cat_feature_cols)

                y = y_series.to_numpy(np.int8)
                hard_override = semantic_df["hard_override_anomaly"].to_numpy(np.int8) == 1

                semantic_primary_prob, semantic_model = self._train_semantic_oof(
                    semantic_df,
                    y,
                    semantic_feature_cols,
                    fold_col="fold_id",
                    fit_final=True,
                )
                semantic_audit_prob, _ = self._train_semantic_oof(
                    semantic_df,
                    y,
                    semantic_feature_cols,
                    fold_col="audit_fold_id",
                    fit_final=False,
                )

                cat_primary_prob: Optional[np.ndarray] = None
                cat_audit_prob: Optional[np.ndarray] = None
                cat_model: Optional[CatBoostClassifier] = None
                if cat_feature_cols:
                    cat_primary_prob, cat_model = self._train_cat_oof(
                        cat_df,
                        y,
                        cat_feature_cols,
                        cat_categorical_cols,
                        fold_col="fold_id",
                        fit_final=True,
                    )
                    cat_audit_prob, _ = self._train_cat_oof(
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
                self.family_models[family] = FamilyModelBundle(
                    semantic_model=semantic_model,
                    cat_model=cat_model,
                    semantic_context=context,
                    semantic_feature_cols=semantic_feature_cols,
                    cat_feature_cols=cat_feature_cols,
                    threshold=threshold,
                    blend_weight=weight,
                )

                family_rows = pd.DataFrame(
                    {
                        "Id": semantic_df["Id"].to_numpy(np.int64),
                        "Label": y,
                        "family": family,
                        "pred_primary": primary_pred,
                        "pred_audit": audit_pred,
                    }
                )
                prediction_rows.append(family_rows)
                primary_metrics[family] = self._metric_summary_from_pred(family_rows["Label"].to_numpy(np.int8), family_rows["pred_primary"].to_numpy(np.int8))
                audit_metrics[family] = self._metric_summary_from_pred(family_rows["Label"].to_numpy(np.int8), family_rows["pred_audit"].to_numpy(np.int8))
                LOGGER.info(
                    "[fit] %s primary F2=%.6f, audit F2=%.6f, threshold=%.3f, blend_weight=%.2f",
                    family,
                    primary_metrics[family].f2,
                    audit_metrics[family].f2,
                    threshold,
                    weight,
                )
                del base_df, semantic_df, cat_df, family_rows
                gc.collect()

        if not other_df.empty:
            other_rows = other_df.copy()
            other_rows["family"] = "other"
            other_rows["pred_primary"] = 1
            other_rows["pred_audit"] = 1
            prediction_rows.append(other_rows[["Id", "Label", "family", "pred_primary", "pred_audit"]])
            primary_metrics["other"] = self._metric_summary_from_pred(other_rows["Label"].to_numpy(np.int8), other_rows["pred_primary"].to_numpy(np.int8))
            audit_metrics["other"] = self._metric_summary_from_pred(other_rows["Label"].to_numpy(np.int8), other_rows["pred_audit"].to_numpy(np.int8))

        if not prediction_rows:
            raise RuntimeError("No training rows were available for model fitting.")
        all_predictions = pd.concat(prediction_rows, ignore_index=True)
        primary_metrics["overall"] = self._metric_summary_from_pred(all_predictions["Label"].to_numpy(np.int8), all_predictions["pred_primary"].to_numpy(np.int8))
        audit_metrics["overall"] = self._metric_summary_from_pred(all_predictions["Label"].to_numpy(np.int8), all_predictions["pred_audit"].to_numpy(np.int8))
        LOGGER.info(
            "[fit] overall primary F2=%.6f, precision=%.4f, recall=%.4f",
            primary_metrics["overall"].f2,
            primary_metrics["overall"].precision,
            primary_metrics["overall"].recall,
        )
        LOGGER.info(
            "[fit] overall audit F2=%.6f, precision=%.4f, recall=%.4f",
            audit_metrics["overall"].f2,
            audit_metrics["overall"].precision,
            audit_metrics["overall"].recall,
        )
        if demoted_rules:
            LOGGER.info("[fit] demoted hard override rules=%s", demoted_rules)
        return self

    def _predict_family_chunk(self, family: str, base_df: pd.DataFrame) -> np.ndarray:
        bundle = self.family_models.get(family)
        if bundle is None or bundle.semantic_model is None:
            raise RuntimeError(f"Missing fitted semantic model bundle for family {family}.")
        self._activate_semantic_context(bundle.semantic_context)
        semantic_df = self._prepare_semantic_frame(base_df.copy())
        hard_override = semantic_df["hard_override_anomaly"].to_numpy(np.int8) == 1
        semantic_prob = np.ones(len(semantic_df), dtype=np.float32)
        if (~hard_override).any():
            semantic_features = semantic_df.loc[~hard_override, bundle.semantic_feature_cols]
            semantic_prob[~hard_override] = bundle.semantic_model.predict_proba(semantic_features)[:, 1].astype(np.float32)

        cat_prob: Optional[np.ndarray] = None
        if bundle.cat_model is not None and bundle.cat_feature_cols:
            cat_df = self._prepare_cat_frame(base_df.copy())
            cat_prob = np.ones(len(cat_df), dtype=np.float32)
            if (~hard_override).any():
                cat_features = cat_df.loc[~hard_override, bundle.cat_feature_cols]
                cat_prob[~hard_override] = bundle.cat_model.predict_proba(cat_features)[:, 1].astype(np.float32)
        blend_prob = self._blend_probs(semantic_prob, cat_prob, bundle.blend_weight)
        blend_prob[hard_override] = 1.0
        pred = (blend_prob >= bundle.threshold).astype(np.int8)
        pred[hard_override] = 1
        return pred

    def predict_test(self, test_path: Path, out_csv: Path) -> None:
        if not self.family_models:
            raise RuntimeError("Model is not fitted.")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        positive_rows = 0
        LOGGER.info("[test] generating predictions from %s", test_path)
        with out_csv.open("w", encoding="utf-8") as fh:
            fh.write("Id,Label\n")
            with tqdm(
                self.iter_raw_chunks(test_path, USECOLS_TEST, 0),
                desc="test chunks",
                unit="chunk",
                dynamic_ncols=True,
            ) as progress:
                for chunk in progress:
                    feats = build_features(chunk, hard_override_names=self.active_hard_override_names)
                    pred = feats["hard_override_anomaly"].astype(np.int8).to_numpy()
                    for family in DEVICE_FAMILY_MAP:
                        family_mask = feats["device_family"] == family
                        if not family_mask.any():
                            continue
                        family_df = feats.loc[family_mask].copy()
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
        LOGGER.info(
            "[test] done; total_rows=%s, positive_rows=%s, positive_rate=%.6f",
            f"{total_rows:,}",
            f"{positive_rows:,}",
            positive_rows / max(total_rows, 1),
        )
