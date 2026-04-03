import gc
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

from .constants import *  # noqa: F403

LOGGER = logging.getLogger(__name__)


@dataclass
class FamilySemanticContext:
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
    semantic_model: XGBClassifier
    semantic_context: FamilySemanticContext
    semantic_feature_cols: List[str]
    cat_model: Optional[CatBoostClassifier]
    cat_feature_cols: List[str]
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
        from .features import FeatureBuilder

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
    def _f2_from_pred(y_true: np.ndarray, pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        y_true_arr = np.asarray(y_true, dtype=np.int8) == 1
        pred_arr = np.asarray(pred, dtype=np.int8) == 1
        tp = int(np.count_nonzero(y_true_arr & pred_arr))
        fp = int(np.count_nonzero(~y_true_arr & pred_arr))
        fn = int(np.count_nonzero(y_true_arr & ~pred_arr))
        denom = 5 * tp + 4 * fn + fp
        if denom == 0:
            return 0.0
        return float((5.0 * tp) / denom)

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

    def _capture_semantic_context(self) -> FamilySemanticContext:
        return FamilySemanticContext(
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
    ) -> Tuple[pd.DataFrame, FamilySemanticContext]:
        work = self._refresh_override_columns(base_df)
        no_valid = pd.Series(np.zeros(len(work), dtype=bool), index=work.index)
        self._fit_surrogate_models(work, y, no_valid)
        work = self._augment_with_surrogates(work)
        self._compute_residual_quantiles(work, y, no_valid)
        work = self._apply_residual_calibration_features(work)
        work = self._fit_transform_scenario_features(work, y)
        work = self._add_family_interaction_features(work)
        return work, self._capture_semantic_context()

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
                semantic_df, context = self._prepare_family_semantic_frame(base_df, y_series)
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
                    semantic_model=semantic_model,
                    semantic_context=context,
                    semantic_feature_cols=list(semantic_feature_cols),
                    cat_model=cat_model,
                    cat_feature_cols=list(cat_feature_cols),
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
                family_primary_f2 = self._f2_from_pred(y, primary_pred)
                family_audit_f2 = self._f2_from_pred(y, audit_pred)
                LOGGER.info(f"[fit] {family} primary F2={family_primary_f2:.6f}, audit F2={family_audit_f2:.6f}, threshold={threshold:.3f}, blend_weight={weight:.2f}")
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
        y_true = all_predictions["Label"].to_numpy(np.int8)
        LOGGER.info(f"[fit] overall primary F2={self._f2_from_pred(y_true, all_predictions['pred_primary'].to_numpy(np.int8)):.6f}")
        LOGGER.info(f"[fit] overall audit F2={self._f2_from_pred(y_true, all_predictions['pred_audit'].to_numpy(np.int8)):.6f}")

    def _predict_family_chunk(self, bundle: FamilyModelBundle, base_df: pd.DataFrame) -> np.ndarray:
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
                    for family, bundle in self.family_bundles.items():
                        family_mask = feats["device_family"] == family
                        if not family_mask.any():
                            continue
                        family_df = feats.loc[family_mask].reset_index(drop=True)
                        family_pred = self._predict_family_chunk(bundle, family_df)
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


__all__ = [
    "DEFAULT_RUN_CONFIG",
    "FamilyModelBundle",
    "FamilySemanticContext",
    "LOGGER",
    "ResearchBaseline",
    "RunConfig",
    "configure_logging",
    "main",
    "run_pipeline",
    "seed_everything",
]
