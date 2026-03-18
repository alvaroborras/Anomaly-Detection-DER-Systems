#!/usr/bin/env python3
"""Research-guided semantic v2 baseline for the DER anomaly-detection task.

Pipeline:
1. Stream train/test directly from the provided zip archive.
2. Build semantically motivated features from SunSpec / DERSec blocks.
3. Apply high-precision hard rules derived from simulator docs + EDA.
4. Train an XGBoost residual model only on canonical rows not already
   flagged by the hard rules.
5. Tune the probability threshold for F2 on a deterministic validation split.
6. Generate a submission for test.csv.

This v2 keeps the original training recipe but expands semantic coverage for:
- common model integrity fields,
- enter-service logic,
- PF and reversion controls,
- curve-driven Volt-Var / Volt-Watt / Watt-Var / Frequency-Droop blocks,
- ride-through / trip blocks,
- structural missingness and DC regime features.
"""
import json
import hashlib
import math
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier, XGBRegressor


def dedupe(columns: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(columns))


def build_volt_var_columns(prefix: str) -> List[str]:
    cols = [
        f"{prefix}.ID",
        f"{prefix}.L",
        f"{prefix}.Ena",
        f"{prefix}.AdptCrvReq",
        f"{prefix}.AdptCrvRslt",
        f"{prefix}.NPt",
        f"{prefix}.NCrv",
        f"{prefix}.RvrtTms",
        f"{prefix}.RvrtRem",
        f"{prefix}.RvrtCrv",
    ]
    for curve in range(3):
        curve_prefix = f"{prefix}.Crv[{curve}]"
        cols.extend(
            [
                f"{curve_prefix}.ActPt",
                f"{curve_prefix}.DeptRef",
                f"{curve_prefix}.Pri",
                f"{curve_prefix}.VRef",
                f"{curve_prefix}.VRefAuto",
                f"{curve_prefix}.VRefAutoEna",
                f"{curve_prefix}.VRefAutoTms",
                f"{curve_prefix}.RspTms",
                f"{curve_prefix}.ReadOnly",
            ]
        )
        for point in range(4):
            cols.extend(
                [
                    f"{curve_prefix}.Pt[{point}].V",
                    f"{curve_prefix}.Pt[{point}].Var",
                ]
            )
    return cols


def build_volt_watt_columns(prefix: str) -> List[str]:
    cols = [
        f"{prefix}.ID",
        f"{prefix}.L",
        f"{prefix}.Ena",
        f"{prefix}.AdptCrvReq",
        f"{prefix}.AdptCrvRslt",
        f"{prefix}.NPt",
        f"{prefix}.NCrv",
        f"{prefix}.RvrtTms",
        f"{prefix}.RvrtRem",
        f"{prefix}.RvrtCrv",
    ]
    for curve in range(3):
        curve_prefix = f"{prefix}.Crv[{curve}]"
        cols.extend(
            [
                f"{curve_prefix}.ActPt",
                f"{curve_prefix}.DeptRef",
                f"{curve_prefix}.RspTms",
                f"{curve_prefix}.ReadOnly",
            ]
        )
        for point in range(2):
            cols.extend(
                [
                    f"{curve_prefix}.Pt[{point}].V",
                    f"{curve_prefix}.Pt[{point}].W",
                ]
            )
    return cols


def build_watt_var_columns(prefix: str) -> List[str]:
    cols = [
        f"{prefix}.ID",
        f"{prefix}.L",
        f"{prefix}.Ena",
        f"{prefix}.AdptCrvReq",
        f"{prefix}.AdptCrvRslt",
        f"{prefix}.NPt",
        f"{prefix}.NCrv",
        f"{prefix}.RvrtTms",
        f"{prefix}.RvrtRem",
        f"{prefix}.RvrtCrv",
    ]
    for curve in range(3):
        curve_prefix = f"{prefix}.Crv[{curve}]"
        cols.extend(
            [
                f"{curve_prefix}.ActPt",
                f"{curve_prefix}.DeptRef",
                f"{curve_prefix}.Pri",
                f"{curve_prefix}.ReadOnly",
            ]
        )
        for point in range(6):
            cols.extend(
                [
                    f"{curve_prefix}.Pt[{point}].W",
                    f"{curve_prefix}.Pt[{point}].Var",
                ]
            )
    return cols


def build_freq_droop_columns(prefix: str) -> List[str]:
    cols = [
        f"{prefix}.ID",
        f"{prefix}.L",
        f"{prefix}.Ena",
        f"{prefix}.AdptCtlReq",
        f"{prefix}.AdptCtlRslt",
        f"{prefix}.NCtl",
        f"{prefix}.RvrtTms",
        f"{prefix}.RvrtRem",
        f"{prefix}.RvrtCtl",
    ]
    for ctl in range(3):
        ctl_prefix = f"{prefix}.Ctl[{ctl}]"
        cols.extend(
            [
                f"{ctl_prefix}.DbOf",
                f"{ctl_prefix}.DbUf",
                f"{ctl_prefix}.KOf",
                f"{ctl_prefix}.KUf",
                f"{ctl_prefix}.RspTms",
                f"{ctl_prefix}.PMin",
                f"{ctl_prefix}.ReadOnly",
            ]
        )
    return cols


def build_trip_columns(prefix: str, axis_name: str) -> List[str]:
    cols = [
        f"{prefix}.ID",
        f"{prefix}.L",
        f"{prefix}.Ena",
        f"{prefix}.AdptCrvReq",
        f"{prefix}.AdptCrvRslt",
        f"{prefix}.NPt",
        f"{prefix}.NCrvSet",
    ]
    for curve in range(2):
        curve_prefix = f"{prefix}.Crv[{curve}]"
        cols.append(f"{curve_prefix}.ReadOnly")
        for group in ["MustTrip", "MayTrip", "MomCess"]:
            group_prefix = f"{curve_prefix}.{group}"
            cols.append(f"{group_prefix}.ActPt")
            for point in range(5):
                cols.extend(
                    [
                        f"{group_prefix}.Pt[{point}].{axis_name}",
                        f"{group_prefix}.Pt[{point}].Tms",
                    ]
                )
    return cols


COMMON_STR = [
    "common[0].Mn",
    "common[0].Md",
    "common[0].Opt",
    "common[0].Vr",
    "common[0].SN",
]

COMMON_COLUMNS = [
    "common[0].ID",
    "common[0].L",
    *COMMON_STR,
    "common[0].DA",
]

MEASURE_AC_COLUMNS = [
    "DERMeasureAC[0].ID",
    "DERMeasureAC[0].L",
    "DERMeasureAC[0].ACType",
    "DERMeasureAC[0].W",
    "DERMeasureAC[0].VA",
    "DERMeasureAC[0].Var",
    "DERMeasureAC[0].PF",
    "DERMeasureAC[0].A",
    "DERMeasureAC[0].LLV",
    "DERMeasureAC[0].LNV",
    "DERMeasureAC[0].Hz",
    "DERMeasureAC[0].TmpAmb",
    "DERMeasureAC[0].TmpCab",
    "DERMeasureAC[0].TmpSnk",
    "DERMeasureAC[0].TmpTrns",
    "DERMeasureAC[0].TmpSw",
    "DERMeasureAC[0].TmpOt",
    "DERMeasureAC[0].ThrotPct",
    "DERMeasureAC[0].ThrotSrc",
    "DERMeasureAC[0].WL1",
    "DERMeasureAC[0].WL2",
    "DERMeasureAC[0].WL3",
    "DERMeasureAC[0].VAL1",
    "DERMeasureAC[0].VAL2",
    "DERMeasureAC[0].VAL3",
    "DERMeasureAC[0].VarL1",
    "DERMeasureAC[0].VarL2",
    "DERMeasureAC[0].VarL3",
    "DERMeasureAC[0].PFL1",
    "DERMeasureAC[0].PFL2",
    "DERMeasureAC[0].PFL3",
    "DERMeasureAC[0].AL1",
    "DERMeasureAC[0].AL2",
    "DERMeasureAC[0].AL3",
    "DERMeasureAC[0].VL1L2",
    "DERMeasureAC[0].VL2L3",
    "DERMeasureAC[0].VL3L1",
    "DERMeasureAC[0].VL1",
    "DERMeasureAC[0].VL2",
    "DERMeasureAC[0].VL3",
]

CAPACITY_COLUMNS = [
    "DERCapacity[0].ID",
    "DERCapacity[0].L",
    "DERCapacity[0].WMaxRtg",
    "DERCapacity[0].VAMaxRtg",
    "DERCapacity[0].VarMaxInjRtg",
    "DERCapacity[0].VarMaxAbsRtg",
    "DERCapacity[0].WChaRteMaxRtg",
    "DERCapacity[0].WDisChaRteMaxRtg",
    "DERCapacity[0].VAChaRteMaxRtg",
    "DERCapacity[0].VADisChaRteMaxRtg",
    "DERCapacity[0].VNomRtg",
    "DERCapacity[0].VMaxRtg",
    "DERCapacity[0].VMinRtg",
    "DERCapacity[0].AMaxRtg",
    "DERCapacity[0].PFOvrExtRtg",
    "DERCapacity[0].PFUndExtRtg",
    "DERCapacity[0].NorOpCatRtg",
    "DERCapacity[0].AbnOpCatRtg",
    "DERCapacity[0].IntIslandCatRtg",
    "DERCapacity[0].WMax",
    "DERCapacity[0].WMaxOvrExt",
    "DERCapacity[0].WOvrExtPF",
    "DERCapacity[0].WMaxUndExt",
    "DERCapacity[0].WUndExtPF",
    "DERCapacity[0].VAMax",
    "DERCapacity[0].VarMaxInj",
    "DERCapacity[0].VarMaxAbs",
    "DERCapacity[0].WChaRteMax",
    "DERCapacity[0].WDisChaRteMax",
    "DERCapacity[0].VAChaRteMax",
    "DERCapacity[0].VADisChaRteMax",
    "DERCapacity[0].VNom",
    "DERCapacity[0].VMax",
    "DERCapacity[0].VMin",
    "DERCapacity[0].AMax",
    "DERCapacity[0].PFOvrExt",
    "DERCapacity[0].PFUndExt",
    "DERCapacity[0].CtrlModes",
    "DERCapacity[0].IntIslandCat",
]

ENTER_SERVICE_COLUMNS = [
    "DEREnterService[0].ID",
    "DEREnterService[0].L",
    "DEREnterService[0].ES",
    "DEREnterService[0].ESVHi",
    "DEREnterService[0].ESVLo",
    "DEREnterService[0].ESHzHi",
    "DEREnterService[0].ESHzLo",
    "DEREnterService[0].ESDlyTms",
    "DEREnterService[0].ESRndTms",
    "DEREnterService[0].ESRmpTms",
    "DEREnterService[0].ESDlyRemTms",
]

CTL_AC_COLUMNS = [
    "DERCtlAC[0].ID",
    "DERCtlAC[0].L",
    "DERCtlAC[0].PFWInjEna",
    "DERCtlAC[0].PFWInjEnaRvrt",
    "DERCtlAC[0].PFWInjRvrtTms",
    "DERCtlAC[0].PFWInjRvrtRem",
    "DERCtlAC[0].PFWAbsEna",
    "DERCtlAC[0].PFWAbsEnaRvrt",
    "DERCtlAC[0].PFWAbsRvrtTms",
    "DERCtlAC[0].PFWAbsRvrtRem",
    "DERCtlAC[0].WMaxLimPctEna",
    "DERCtlAC[0].WMaxLimPct",
    "DERCtlAC[0].WMaxLimPctRvrt",
    "DERCtlAC[0].WMaxLimPctEnaRvrt",
    "DERCtlAC[0].WMaxLimPctRvrtTms",
    "DERCtlAC[0].WMaxLimPctRvrtRem",
    "DERCtlAC[0].WSetEna",
    "DERCtlAC[0].WSetMod",
    "DERCtlAC[0].WSet",
    "DERCtlAC[0].WSetRvrt",
    "DERCtlAC[0].WSetPct",
    "DERCtlAC[0].WSetPctRvrt",
    "DERCtlAC[0].WSetEnaRvrt",
    "DERCtlAC[0].WSetRvrtTms",
    "DERCtlAC[0].WSetRvrtRem",
    "DERCtlAC[0].VarSetEna",
    "DERCtlAC[0].VarSetMod",
    "DERCtlAC[0].VarSetPri",
    "DERCtlAC[0].VarSet",
    "DERCtlAC[0].VarSetRvrt",
    "DERCtlAC[0].VarSetPct",
    "DERCtlAC[0].VarSetPctRvrt",
    "DERCtlAC[0].VarSetEnaRvrt",
    "DERCtlAC[0].VarSetRvrtTms",
    "DERCtlAC[0].VarSetRvrtRem",
    "DERCtlAC[0].WRmp",
    "DERCtlAC[0].WRmpRef",
    "DERCtlAC[0].VarRmp",
    "DERCtlAC[0].AntiIslEna",
    "DERCtlAC[0].PFWInj.PF",
    "DERCtlAC[0].PFWInj.Ext",
    "DERCtlAC[0].PFWInjRvrt.PF",
    "DERCtlAC[0].PFWInjRvrt.Ext",
    "DERCtlAC[0].PFWAbs.Ext",
    "DERCtlAC[0].PFWAbsRvrt.Ext",
]

VOLT_VAR_COLUMNS = build_volt_var_columns("DERVoltVar[0]")
VOLT_WATT_COLUMNS = build_volt_watt_columns("DERVoltWatt[0]")
FREQ_DROOP_COLUMNS = build_freq_droop_columns("DERFreqDroop[0]")
WATT_VAR_COLUMNS = build_watt_var_columns("DERWattVar[0]")

TRIP_SPECS: Dict[str, Tuple[str, str, str]] = {
    "lv": ("DERTripLV[0]", "V", "low"),
    "hv": ("DERTripHV[0]", "V", "high"),
    "lf": ("DERTripLF[0]", "Hz", "low"),
    "hf": ("DERTripHF[0]", "Hz", "high"),
}
TRIP_COLUMNS = {
    short_name: build_trip_columns(prefix, axis_name)
    for short_name, (prefix, axis_name, _) in TRIP_SPECS.items()
}

MEASURE_DC_COLUMNS = [
    "DERMeasureDC[0].ID",
    "DERMeasureDC[0].L",
    "DERMeasureDC[0].NPrt",
    "DERMeasureDC[0].DCA",
    "DERMeasureDC[0].DCW",
    "DERMeasureDC[0].Prt[0].PrtTyp",
    "DERMeasureDC[0].Prt[0].ID",
    "DERMeasureDC[0].Prt[0].DCA",
    "DERMeasureDC[0].Prt[0].DCV",
    "DERMeasureDC[0].Prt[0].DCW",
    "DERMeasureDC[0].Prt[0].Tmp",
    "DERMeasureDC[0].Prt[1].PrtTyp",
    "DERMeasureDC[0].Prt[1].ID",
    "DERMeasureDC[0].Prt[1].DCA",
    "DERMeasureDC[0].Prt[1].DCV",
    "DERMeasureDC[0].Prt[1].DCW",
    "DERMeasureDC[0].Prt[1].Tmp",
]

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

RAW_NUMERIC = dedupe(
    [
        "common[0].DA",
        "DERMeasureAC[0].ACType",
        "DERMeasureAC[0].W",
        "DERMeasureAC[0].VA",
        "DERMeasureAC[0].Var",
        "DERMeasureAC[0].PF",
        "DERMeasureAC[0].A",
        "DERMeasureAC[0].LLV",
        "DERMeasureAC[0].LNV",
        "DERMeasureAC[0].Hz",
        "DERMeasureAC[0].TmpAmb",
        "DERMeasureAC[0].TmpCab",
        "DERMeasureAC[0].TmpSnk",
        "DERMeasureAC[0].TmpTrns",
        "DERMeasureAC[0].TmpSw",
        "DERMeasureAC[0].TmpOt",
        "DERMeasureAC[0].ThrotPct",
        "DERMeasureAC[0].ThrotSrc",
        "DERMeasureAC[0].WL1",
        "DERMeasureAC[0].WL2",
        "DERMeasureAC[0].WL3",
        "DERMeasureAC[0].VAL1",
        "DERMeasureAC[0].VAL2",
        "DERMeasureAC[0].VAL3",
        "DERMeasureAC[0].VarL1",
        "DERMeasureAC[0].VarL2",
        "DERMeasureAC[0].VarL3",
        "DERMeasureAC[0].PFL1",
        "DERMeasureAC[0].PFL2",
        "DERMeasureAC[0].PFL3",
        "DERMeasureAC[0].AL1",
        "DERMeasureAC[0].AL2",
        "DERMeasureAC[0].AL3",
        "DERMeasureAC[0].VL1L2",
        "DERMeasureAC[0].VL2L3",
        "DERMeasureAC[0].VL3L1",
        "DERMeasureAC[0].VL1",
        "DERMeasureAC[0].VL2",
        "DERMeasureAC[0].VL3",
        "DERCapacity[0].WMaxRtg",
        "DERCapacity[0].VAMaxRtg",
        "DERCapacity[0].VarMaxInjRtg",
        "DERCapacity[0].VarMaxAbsRtg",
        "DERCapacity[0].WChaRteMaxRtg",
        "DERCapacity[0].WDisChaRteMaxRtg",
        "DERCapacity[0].VAChaRteMaxRtg",
        "DERCapacity[0].VADisChaRteMaxRtg",
        "DERCapacity[0].VNomRtg",
        "DERCapacity[0].VMaxRtg",
        "DERCapacity[0].VMinRtg",
        "DERCapacity[0].AMaxRtg",
        "DERCapacity[0].PFOvrExtRtg",
        "DERCapacity[0].PFUndExtRtg",
        "DERCapacity[0].NorOpCatRtg",
        "DERCapacity[0].AbnOpCatRtg",
        "DERCapacity[0].IntIslandCatRtg",
        "DERCapacity[0].WMax",
        "DERCapacity[0].WMaxOvrExt",
        "DERCapacity[0].WOvrExtPF",
        "DERCapacity[0].WMaxUndExt",
        "DERCapacity[0].WUndExtPF",
        "DERCapacity[0].VAMax",
        "DERCapacity[0].VarMaxInj",
        "DERCapacity[0].VarMaxAbs",
        "DERCapacity[0].WChaRteMax",
        "DERCapacity[0].WDisChaRteMax",
        "DERCapacity[0].VAChaRteMax",
        "DERCapacity[0].VADisChaRteMax",
        "DERCapacity[0].VNom",
        "DERCapacity[0].VMax",
        "DERCapacity[0].VMin",
        "DERCapacity[0].AMax",
        "DERCapacity[0].PFOvrExt",
        "DERCapacity[0].PFUndExt",
        "DERCapacity[0].CtrlModes",
        "DERCapacity[0].IntIslandCat",
        "DEREnterService[0].ES",
        "DEREnterService[0].ESVHi",
        "DEREnterService[0].ESVLo",
        "DEREnterService[0].ESHzHi",
        "DEREnterService[0].ESHzLo",
        "DEREnterService[0].ESDlyTms",
        "DEREnterService[0].ESRndTms",
        "DEREnterService[0].ESRmpTms",
        "DEREnterService[0].ESDlyRemTms",
        "DERCtlAC[0].PFWInjEna",
        "DERCtlAC[0].PFWInjEnaRvrt",
        "DERCtlAC[0].PFWInjRvrtTms",
        "DERCtlAC[0].PFWInjRvrtRem",
        "DERCtlAC[0].PFWAbsEna",
        "DERCtlAC[0].PFWAbsEnaRvrt",
        "DERCtlAC[0].PFWAbsRvrtTms",
        "DERCtlAC[0].PFWAbsRvrtRem",
        "DERCtlAC[0].WMaxLimPctEna",
        "DERCtlAC[0].WMaxLimPct",
        "DERCtlAC[0].WMaxLimPctRvrt",
        "DERCtlAC[0].WMaxLimPctEnaRvrt",
        "DERCtlAC[0].WMaxLimPctRvrtTms",
        "DERCtlAC[0].WMaxLimPctRvrtRem",
        "DERCtlAC[0].WSetEna",
        "DERCtlAC[0].WSetMod",
        "DERCtlAC[0].WSet",
        "DERCtlAC[0].WSetRvrt",
        "DERCtlAC[0].WSetPct",
        "DERCtlAC[0].WSetPctRvrt",
        "DERCtlAC[0].WSetEnaRvrt",
        "DERCtlAC[0].WSetRvrtTms",
        "DERCtlAC[0].WSetRvrtRem",
        "DERCtlAC[0].VarSetEna",
        "DERCtlAC[0].VarSetMod",
        "DERCtlAC[0].VarSetPri",
        "DERCtlAC[0].VarSet",
        "DERCtlAC[0].VarSetRvrt",
        "DERCtlAC[0].VarSetPct",
        "DERCtlAC[0].VarSetPctRvrt",
        "DERCtlAC[0].VarSetEnaRvrt",
        "DERCtlAC[0].VarSetRvrtTms",
        "DERCtlAC[0].VarSetRvrtRem",
        "DERCtlAC[0].WRmp",
        "DERCtlAC[0].WRmpRef",
        "DERCtlAC[0].VarRmp",
        "DERCtlAC[0].AntiIslEna",
        "DERCtlAC[0].PFWInj.PF",
        "DERCtlAC[0].PFWInj.Ext",
        "DERCtlAC[0].PFWInjRvrt.PF",
        "DERCtlAC[0].PFWInjRvrt.Ext",
        "DERCtlAC[0].PFWAbs.Ext",
        "DERCtlAC[0].PFWAbsRvrt.Ext",
        "DERVoltVar[0].Ena",
        "DERVoltVar[0].AdptCrvReq",
        "DERVoltVar[0].AdptCrvRslt",
        "DERVoltVar[0].NPt",
        "DERVoltVar[0].NCrv",
        "DERVoltVar[0].RvrtTms",
        "DERVoltVar[0].RvrtRem",
        "DERVoltVar[0].RvrtCrv",
        "DERVoltWatt[0].Ena",
        "DERVoltWatt[0].AdptCrvReq",
        "DERVoltWatt[0].AdptCrvRslt",
        "DERVoltWatt[0].NPt",
        "DERVoltWatt[0].NCrv",
        "DERVoltWatt[0].RvrtTms",
        "DERVoltWatt[0].RvrtRem",
        "DERVoltWatt[0].RvrtCrv",
        "DERFreqDroop[0].Ena",
        "DERFreqDroop[0].AdptCtlReq",
        "DERFreqDroop[0].AdptCtlRslt",
        "DERFreqDroop[0].NCtl",
        "DERFreqDroop[0].RvrtTms",
        "DERFreqDroop[0].RvrtRem",
        "DERFreqDroop[0].RvrtCtl",
        "DERWattVar[0].Ena",
        "DERWattVar[0].AdptCrvReq",
        "DERWattVar[0].AdptCrvRslt",
        "DERWattVar[0].NPt",
        "DERWattVar[0].NCrv",
        "DERWattVar[0].RvrtTms",
        "DERWattVar[0].RvrtRem",
        "DERWattVar[0].RvrtCrv",
        "DERMeasureDC[0].NPrt",
        "DERMeasureDC[0].DCA",
        "DERMeasureDC[0].DCW",
        "DERMeasureDC[0].Prt[0].PrtTyp",
        "DERMeasureDC[0].Prt[0].ID",
        "DERMeasureDC[0].Prt[0].DCA",
        "DERMeasureDC[0].Prt[0].DCV",
        "DERMeasureDC[0].Prt[0].DCW",
        "DERMeasureDC[0].Prt[0].Tmp",
        "DERMeasureDC[0].Prt[1].PrtTyp",
        "DERMeasureDC[0].Prt[1].ID",
        "DERMeasureDC[0].Prt[1].DCA",
        "DERMeasureDC[0].Prt[1].DCV",
        "DERMeasureDC[0].Prt[1].DCW",
        "DERMeasureDC[0].Prt[1].Tmp",
    ]
)

TRIP_META_COLUMNS = []
for prefix, _, _ in TRIP_SPECS.values():
    TRIP_META_COLUMNS.extend(
        [
            f"{prefix}.Ena",
            f"{prefix}.AdptCrvReq",
            f"{prefix}.AdptCrvRslt",
            f"{prefix}.NPt",
            f"{prefix}.NCrvSet",
        ]
    )
RAW_NUMERIC = dedupe([*RAW_NUMERIC, *TRIP_META_COLUMNS])

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
        *TRIP_COLUMNS["lv"],
        *TRIP_COLUMNS["hv"],
        *TRIP_COLUMNS["lf"],
        *TRIP_COLUMNS["hf"],
    ]
)
NUMERIC_SOURCE_COLUMNS = [c for c in ALL_SOURCE_COLUMNS if c not in COMMON_STR]

USECOLS_TRAIN = dedupe(["Id", "Label", *ALL_SOURCE_COLUMNS])
USECOLS_TEST = dedupe(["Id", *ALL_SOURCE_COLUMNS])

CANON1 = "DERSec|DER Simulator|10 kW DER|1.2.3|SN-Three-Phase"
CANON2 = "DERSec|DER Simulator 100 kW|1.2.3.1|1.0.0|1100058974"
SAFE_RAW = {c: re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in RAW_NUMERIC}
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ARCHIVE_NAME = "cyber-physical-anomaly-detection-for-der-systems.zip"
DEFAULT_SEED = 42
MODEL_FILENAME = "semantic_v4_single_xgb.json"
REPORT_FILENAME = "semantic_v4_single_validation.json"


def default_zip_path() -> Path:
    local_zip_path = SCRIPT_DIR / DATASET_ARCHIVE_NAME
    if local_zip_path.exists():
        return local_zip_path
    return Path("/mnt/data") / DATASET_ARCHIVE_NAME


DEFAULT_ZIP_PATH = default_zip_path()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "v4_single"
DEVICE_FAMILY_MAP = {"canon10": 0, "canon100": 1}
RESIDUAL_TAIL_LEVELS = {"tail": 0.95, "extreme": 0.99, "ultra": 0.999}
RESIDUAL_TAIL_FALLBACKS = {"tail": 0.05, "extreme": 0.10, "ultra": 0.20}
FAMILY_THRESHOLD_FLOOR = 0.10
FAMILY_THRESHOLD_MAX_DELTA = 0.10
FAMILY_THRESHOLD_SHRINK = 0.50
CANON100_NEGATIVE_WEIGHT = 1.50
HARD_OVERRIDE_TRAIN_WEIGHT = 0.35
SCENARIO_SMOOTHING = 50.0
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
    "DERMeasureAC_0_W",
    "DERMeasureAC_0_VA",
    "DERMeasureAC_0_Var",
    "DERMeasureAC_0_PF",
    "DERMeasureAC_0_A",
    "DERMeasureAC_0_WL1",
    "DERMeasureAC_0_WL2",
    "DERMeasureAC_0_WL3",
    "DERMeasureAC_0_VAL1",
    "DERMeasureAC_0_VAL2",
    "DERMeasureAC_0_VAL3",
    "DERMeasureAC_0_VarL1",
    "DERMeasureAC_0_VarL2",
    "DERMeasureAC_0_VarL3",
    "DERMeasureAC_0_PFL1",
    "DERMeasureAC_0_PFL2",
    "DERMeasureAC_0_PFL3",
    "DERMeasureAC_0_AL1",
    "DERMeasureAC_0_AL2",
    "DERMeasureAC_0_AL3",
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
}

EXPECTED_MODEL_META = {
    "common": ("common[0].ID", "common[0].L", 1.0, 66.0),
    "measure_ac": ("DERMeasureAC[0].ID", "DERMeasureAC[0].L", 701.0, 153.0),
    "capacity": ("DERCapacity[0].ID", "DERCapacity[0].L", 702.0, 50.0),
    "enter_service": ("DEREnterService[0].ID", "DEREnterService[0].L", 703.0, 17.0),
    "measure_dc": ("DERMeasureDC[0].ID", "DERMeasureDC[0].L", 714.0, 68.0),
}


@dataclass
class ValidationReport:
    threshold: float
    f2: float
    precision: float
    recall: float
    canon10_threshold: float
    canon100_threshold: float
    hard_rule_valid_f2: float
    hard_rule_valid_precision: float
    hard_rule_valid_recall: float
    sample_rows: int
    residual_train_rows: int
    residual_valid_rows: int
    feature_count: int
    surrogate_feature_count: int

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "f2": self.f2,
            "precision": self.precision,
            "recall": self.recall,
            "canon10_threshold": self.canon10_threshold,
            "canon100_threshold": self.canon100_threshold,
            "hard_rule_valid_f2": self.hard_rule_valid_f2,
            "hard_rule_valid_precision": self.hard_rule_valid_precision,
            "hard_rule_valid_recall": self.hard_rule_valid_recall,
            "sample_rows": self.sample_rows,
            "residual_train_rows": self.residual_train_rows,
            "residual_valid_rows": self.residual_valid_rows,
            "feature_count": self.feature_count,
            "surrogate_feature_count": self.surrogate_feature_count,
        }


@dataclass(frozen=True)
class RunConfig:
    zip_path: Path = DEFAULT_ZIP_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    write_test_predictions: bool = True
    chunksize: int = 5000
    sample_rows: int = 100000
    n_estimators: int = 150
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int = 4
    seed: int = DEFAULT_SEED

    def create_baseline(self) -> "ResearchBaseline":
        return ResearchBaseline(
            chunksize=self.chunksize,
            sample_rows=self.sample_rows,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )


def seed_everything(seed: int) -> None:
    # Keep the pipeline reproducible without forcing single-threaded execution.
    random.seed(seed)
    np.random.seed(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchBaseline:
    def __init__(
        self,
        *,
        chunksize: int = 5000,
        sample_rows: int = 100000,
        n_estimators: int = 150,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = 4,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.chunksize = chunksize
        self.sample_rows = sample_rows
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.seed = seed
        self.model: Optional[XGBClassifier] = None
        self.threshold: float = 0.5
        self.family_thresholds: Dict[str, float] = {"canon10": 0.5, "canon100": 0.5}
        self.feature_cols: Optional[List[str]] = None
        self.surrogate_feature_cols: Optional[List[str]] = None
        self.surrogate_models: Dict[Tuple[str, str], XGBRegressor] = {}
        self.residual_quantiles: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.family_base_rates: Dict[str, float] = {}
        self.scenario_sum_map: Dict[int, float] = {}
        self.scenario_count_map: Dict[int, int] = {}
        self.scenario_output_sum_map: Dict[int, float] = {}
        self.scenario_output_count_map: Dict[int, int] = {}
        self.validation_report: Optional[ValidationReport] = None

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
        return (
            np.isfinite(np.asarray(x_points, dtype=np.float32))
            & np.isfinite(np.asarray(y_points, dtype=np.float32))
        ).sum(axis=1).astype(np.int16)

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
        valid = (
            np.isfinite(x_points[:, :-1])
            & np.isfinite(x_points[:, 1:])
            & np.isfinite(y_points[:, :-1])
            & np.isfinite(y_points[:, 1:])
            & (np.abs(dx) > 1e-6)
        )
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
            valid_seg = (
                np.isfinite(x0)
                & np.isfinite(x1)
                & np.isfinite(y0)
                & np.isfinite(y1)
                & (np.abs(x1 - x0) > 1e-6)
            )
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
        denom = np.where(var >= 0, np.asarray(varmaxinj, dtype=np.float32), np.asarray(varmaxabs, dtype=np.float32))
        return 100.0 * ResearchBaseline._safe_div(var, denom)

    def _coerce_numeric(self, df: pd.DataFrame) -> None:
        for col in NUMERIC_SOURCE_COLUMNS:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def _add_block_missingness(self, data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
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

    def _add_model_integrity_features(self, data: Dict[str, np.ndarray], df: pd.DataFrame) -> None:
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

    def _add_capacity_extension_features(
        self,
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
        data["charge_rate_share_rtg"] = self._safe_div(wcha_rtg, wmaxrtg)
        data["discharge_rate_share_rtg"] = self._safe_div(wdis_rtg, wmaxrtg)
        data["charge_va_share_rtg"] = self._safe_div(vacha_rtg, vamaxrtg)
        data["discharge_va_share_rtg"] = self._safe_div(vadis_rtg, vamaxrtg)
        data["charge_rate_share_setting"] = self._safe_div(wcha, wmax)
        data["discharge_rate_share_setting"] = self._safe_div(wdis, wmax)
        data["charge_va_share_setting"] = self._safe_div(vacha, vamax)
        data["discharge_va_share_setting"] = self._safe_div(vadis, vamax)
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

    def _add_enter_service_features(
        self,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return state_anomaly.astype(np.int8), blocked_power.astype(np.int8), blocked_current.astype(np.int8)

    def _add_pf_control_features(
        self,
        data: Dict[str, np.ndarray],
        df: pd.DataFrame,
        *,
        pf: np.ndarray,
        var: np.ndarray,
        varmaxinj: np.ndarray,
        varmaxabs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        observed_var_pct = self._var_pct(var, varmaxinj, varmaxabs)
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
        data["pf_reactive_near_limit"] = (
            np.abs(observed_var_pct) >= 95.0
        ).astype(np.int8)
        return np.isfinite(pfabs_ext).astype(np.int8), np.isfinite(pfabs_rvrt_ext).astype(np.int8)

    def _add_trip_block_features(
        self,
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
        adpt_idx = self._curve_index(df[f"{prefix}.AdptCrvRslt"].to_numpy(float), 2)
        must_actpt = self._select_curve_scalar(
            [
                df[f"{prefix}.Crv[0].MustTrip.ActPt"].to_numpy(float),
                df[f"{prefix}.Crv[1].MustTrip.ActPt"].to_numpy(float),
            ],
            adpt_idx,
        )
        mom_actpt = self._select_curve_scalar(
            [
                df[f"{prefix}.Crv[0].MomCess.ActPt"].to_numpy(float),
                df[f"{prefix}.Crv[1].MomCess.ActPt"].to_numpy(float),
            ],
            adpt_idx,
        )
        must_x = self._select_curve_points(
            [
                np.column_stack([df[f"{prefix}.Crv[0].MustTrip.Pt[{i}].{axis_name}"].to_numpy(float) for i in range(5)]),
                np.column_stack([df[f"{prefix}.Crv[1].MustTrip.Pt[{i}].{axis_name}"].to_numpy(float) for i in range(5)]),
            ],
            adpt_idx,
        )
        must_t = self._select_curve_points(
            [
                np.column_stack([df[f"{prefix}.Crv[0].MustTrip.Pt[{i}].Tms"].to_numpy(float) for i in range(5)]),
                np.column_stack([df[f"{prefix}.Crv[1].MustTrip.Pt[{i}].Tms"].to_numpy(float) for i in range(5)]),
            ],
            adpt_idx,
        )
        mom_x = self._select_curve_points(
            [
                np.column_stack([df[f"{prefix}.Crv[0].MomCess.Pt[{i}].{axis_name}"].to_numpy(float) for i in range(5)]),
                np.column_stack([df[f"{prefix}.Crv[1].MomCess.Pt[{i}].{axis_name}"].to_numpy(float) for i in range(5)]),
            ],
            adpt_idx,
        )
        mom_t = self._select_curve_points(
            [
                np.column_stack([df[f"{prefix}.Crv[0].MomCess.Pt[{i}].Tms"].to_numpy(float) for i in range(5)]),
                np.column_stack([df[f"{prefix}.Crv[1].MomCess.Pt[{i}].Tms"].to_numpy(float) for i in range(5)]),
            ],
            adpt_idx,
        )
        may_present = np.column_stack(
            [
                df[f"{prefix}.Crv[{curve}].MayTrip.Pt[{point}].{axis_name}"].to_numpy(float)
                for curve in range(2)
                for point in range(5)
            ]
        )

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
        data[f"trip_{short_name}_curve_req_gap"] = (
            df[f"{prefix}.AdptCrvReq"].to_numpy(float) - df[f"{prefix}.AdptCrvRslt"].to_numpy(float)
        ).astype(np.float32)
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
        return outside.astype(np.int8), power_when_outside.astype(np.int8)

    def _add_curve_block_features(
        self,
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
        adpt_idx = self._curve_index(raw_idx, len(curve_x))
        selected_x = self._select_curve_points(curve_x, adpt_idx)
        selected_y = self._select_curve_points(curve_y, adpt_idx)
        selected_actpt = self._select_curve_scalar(curve_actpt, adpt_idx)
        data[f"{name}_curve_idx"] = adpt_idx.astype(np.int8)
        point_count = self._pair_point_count(selected_x, selected_y)
        data[f"{name}_curve_point_count"] = point_count
        data[f"{name}_curve_actpt_gap"] = (selected_actpt - point_count).astype(np.float32)
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
        if observed_value is not None:
            expected_value = self._piecewise_interp(measure_value, selected_x, selected_y)
            data[f"{name}_curve_expected"] = expected_value.astype(np.float32)
            data[f"{name}_curve_error"] = (observed_value - expected_value).astype(np.float32)
        for meta_name, curves in curve_meta.items():
            data[f"{name}_curve_{meta_name}"] = self._select_curve_scalar(curves, adpt_idx).astype(np.float32)

    def _add_freq_droop_features(
        self,
        data: Dict[str, np.ndarray],
        df: pd.DataFrame,
        *,
        hz: np.ndarray,
        w_pct: np.ndarray,
    ) -> None:
        raw_idx = df["DERFreqDroop[0].AdptCtlRslt"].to_numpy(float)
        ctl_idx = self._curve_index(raw_idx, 3)
        dbof_curves = [df[f"DERFreqDroop[0].Ctl[{i}].DbOf"].to_numpy(float) for i in range(3)]
        dbuf_curves = [df[f"DERFreqDroop[0].Ctl[{i}].DbUf"].to_numpy(float) for i in range(3)]
        kof_curves = [df[f"DERFreqDroop[0].Ctl[{i}].KOf"].to_numpy(float) for i in range(3)]
        kuf_curves = [df[f"DERFreqDroop[0].Ctl[{i}].KUf"].to_numpy(float) for i in range(3)]
        rsp_curves = [df[f"DERFreqDroop[0].Ctl[{i}].RspTms"].to_numpy(float) for i in range(3)]
        pmin_curves = [df[f"DERFreqDroop[0].Ctl[{i}].PMin"].to_numpy(float) for i in range(3)]
        ro_curves = [df[f"DERFreqDroop[0].Ctl[{i}].ReadOnly"].to_numpy(float) for i in range(3)]
        dbof = self._select_curve_scalar(dbof_curves, ctl_idx)
        dbuf = self._select_curve_scalar(dbuf_curves, ctl_idx)
        kof = self._select_curve_scalar(kof_curves, ctl_idx)
        kuf = self._select_curve_scalar(kuf_curves, ctl_idx)
        rsp = self._select_curve_scalar(rsp_curves, ctl_idx)
        pmin = self._select_curve_scalar(pmin_curves, ctl_idx)
        readonly = self._select_curve_scalar(ro_curves, ctl_idx)

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
        data["ac_dc_same_sign"] = (
            np.sign(np.nan_to_num(w, nan=0.0)) == np.sign(np.nan_to_num(dcw, nan=0.0))
        ).astype(np.int8)
        data["dca_over_total"] = self._safe_div(dca, prt0_a + prt1_a)
        return rare_type.astype(np.int8)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._coerce_numeric(df)

        fingerprint = df[COMMON_STR].fillna("<NA>").agg("|".join, axis=1)
        data: Dict[str, np.ndarray] = {
            "Id": df["Id"].to_numpy(),
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

        self._add_block_missingness(data, df)
        self._add_model_integrity_features(data, df)
        self._add_temperature_features(data, df)

        w = df["DERMeasureAC[0].W"].to_numpy(float)
        abs_w = np.abs(w)
        va = df["DERMeasureAC[0].VA"].to_numpy(float)
        var = df["DERMeasureAC[0].Var"].to_numpy(float)
        pf = df["DERMeasureAC[0].PF"].to_numpy(float)
        a = df["DERMeasureAC[0].A"].to_numpy(float)
        llv = df["DERMeasureAC[0].LLV"].to_numpy(float)
        lnv = df["DERMeasureAC[0].LNV"].to_numpy(float)
        hz = df["DERMeasureAC[0].Hz"].to_numpy(float)

        wmaxrtg = df["DERCapacity[0].WMaxRtg"].to_numpy(float)
        vamaxrtg = df["DERCapacity[0].VAMaxRtg"].to_numpy(float)
        varmaxinjrtg = df["DERCapacity[0].VarMaxInjRtg"].to_numpy(float)
        varmaxabsrtg = df["DERCapacity[0].VarMaxAbsRtg"].to_numpy(float)
        wmax = df["DERCapacity[0].WMax"].to_numpy(float)
        vamax = df["DERCapacity[0].VAMax"].to_numpy(float)
        varmaxinj = df["DERCapacity[0].VarMaxInj"].to_numpy(float)
        varmaxabs = df["DERCapacity[0].VarMaxAbs"].to_numpy(float)
        amax = df["DERCapacity[0].AMax"].to_numpy(float)
        vnom = df["DERCapacity[0].VNom"].to_numpy(float)
        vmax = df["DERCapacity[0].VMax"].to_numpy(float)
        vmin = df["DERCapacity[0].VMin"].to_numpy(float)

        data["w_over_wmaxrtg"] = self._safe_div(w, wmaxrtg)
        data["w_over_wmax"] = self._safe_div(w, wmax)
        data["va_over_vamax"] = self._safe_div(va, vamax)
        data["va_over_vamaxrtg"] = self._safe_div(va, vamaxrtg)
        data["var_over_injmax"] = self._safe_div(var, varmaxinj)
        data["var_over_absmax"] = self._safe_div(var, varmaxabs)
        data["a_over_amax"] = self._safe_div(a, amax)
        data["llv_over_vnom"] = self._safe_div(llv, vnom)
        data["lnv_over_vnom"] = self._safe_div(lnv * math.sqrt(3.0), vnom)

        data["w_minus_wmax"] = (w - wmax).astype(np.float32)
        data["w_minus_wmaxrtg"] = (w - wmaxrtg).astype(np.float32)
        data["va_minus_vamax"] = (va - vamax).astype(np.float32)
        data["var_minus_injmax"] = (var - varmaxinj).astype(np.float32)
        data["var_plus_absmax"] = (var + varmaxabs).astype(np.float32)
        data["llv_minus_lnv_sqrt3"] = (llv - lnv * math.sqrt(3.0)).astype(np.float32)
        data["hz_delta_60"] = (hz - 60.0).astype(np.float32)

        data["w_eq_wmaxrtg"] = np.isclose(w, wmaxrtg, equal_nan=False).astype(np.int8)
        data["w_eq_wmax"] = np.isclose(w, wmax, equal_nan=False).astype(np.int8)
        data["var_eq_varmaxinj"] = np.isclose(var, varmaxinj, equal_nan=False).astype(np.int8)
        data["var_eq_neg_varmaxabs"] = np.isclose(var, -varmaxabs, equal_nan=False).astype(np.int8)
        data["pf_sign_mismatch"] = (
            (np.sign(np.nan_to_num(pf)) != np.sign(np.nan_to_num(w)))
            & (np.nan_to_num(pf) != 0)
            & (np.nan_to_num(w) != 0)
        ).astype(np.int8)

        tolw = np.maximum(50.0, 0.02 * np.nan_to_num(wmaxrtg, nan=0.0)).astype(np.float32)
        tolva = np.maximum(50.0, 0.02 * np.nan_to_num(vamax, nan=0.0)).astype(np.float32)
        tolvi = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxinj, nan=0.0)).astype(np.float32)
        tolva2 = np.maximum(20.0, 0.02 * np.nan_to_num(varmaxabs, nan=0.0)).astype(np.float32)
        data["w_gt_wmax_tol"] = (w > (wmax + tolw)).astype(np.int8)
        data["w_gt_wmaxrtg_tol"] = (w > (wmaxrtg + tolw)).astype(np.int8)
        data["va_gt_vamax_tol"] = (va > (vamax + tolva)).astype(np.int8)
        data["var_gt_injmax_tol"] = (var > (varmaxinj + tolvi)).astype(np.int8)
        data["var_lt_absmax_tol"] = (var < (-varmaxabs - tolva2)).astype(np.int8)

        pq = np.sqrt(np.square(w.astype(np.float32)) + np.square(var.astype(np.float32)))
        data["va_minus_pqmag"] = (va - pq).astype(np.float32)
        data["va_over_pqmag"] = self._safe_div(va, pq)
        data["pf_from_w_va"] = self._safe_div(w, va)
        data["pf_error"] = (pf - data["pf_from_w_va"]).astype(np.float32)

        data["w_phase_sum_error"] = (
            w
            - (
                df["DERMeasureAC[0].WL1"].to_numpy(float)
                + df["DERMeasureAC[0].WL2"].to_numpy(float)
                + df["DERMeasureAC[0].WL3"].to_numpy(float)
            )
        ).astype(np.float32)
        data["va_phase_sum_error"] = (
            va
            - (
                df["DERMeasureAC[0].VAL1"].to_numpy(float)
                + df["DERMeasureAC[0].VAL2"].to_numpy(float)
                + df["DERMeasureAC[0].VAL3"].to_numpy(float)
            )
        ).astype(np.float32)
        data["var_phase_sum_error"] = (
            var
            - (
                df["DERMeasureAC[0].VarL1"].to_numpy(float)
                + df["DERMeasureAC[0].VarL2"].to_numpy(float)
                + df["DERMeasureAC[0].VarL3"].to_numpy(float)
            )
        ).astype(np.float32)
        phase_ll = df[["DERMeasureAC[0].VL1L2", "DERMeasureAC[0].VL2L3", "DERMeasureAC[0].VL3L1"]].to_numpy(float)
        phase_ln = df[["DERMeasureAC[0].VL1", "DERMeasureAC[0].VL2", "DERMeasureAC[0].VL3"]].to_numpy(float)
        phase_w = df[["DERMeasureAC[0].WL1", "DERMeasureAC[0].WL2", "DERMeasureAC[0].WL3"]].to_numpy(float)
        phase_var = df[["DERMeasureAC[0].VarL1", "DERMeasureAC[0].VarL2", "DERMeasureAC[0].VarL3"]].to_numpy(float)
        data["phase_ll_spread"] = (self._nanmax_rows(phase_ll) - self._nanmin_rows(phase_ll)).astype(np.float32)
        data["phase_ln_spread"] = (self._nanmax_rows(phase_ln) - self._nanmin_rows(phase_ln)).astype(np.float32)
        data["phase_w_spread"] = (self._nanmax_rows(phase_w) - self._nanmin_rows(phase_w)).astype(np.float32)
        data["phase_var_spread"] = (self._nanmax_rows(phase_var) - self._nanmin_rows(phase_var)).astype(np.float32)

        data["wmax_over_wmaxrtg"] = self._safe_div(wmax, wmaxrtg)
        data["vamax_over_vamaxrtg"] = self._safe_div(vamax, vamaxrtg)
        data["vmax_over_vnom"] = self._safe_div(vmax, vnom)
        data["vmin_over_vnom"] = self._safe_div(vmin, vnom)

        wsetena = np.nan_to_num(df["DERCtlAC[0].WSetEna"].to_numpy(float), nan=0.0)
        wset = df["DERCtlAC[0].WSet"].to_numpy(float)
        wsetpct = df["DERCtlAC[0].WSetPct"].to_numpy(float)
        wmaxlimena = np.nan_to_num(df["DERCtlAC[0].WMaxLimPctEna"].to_numpy(float), nan=0.0)
        wmaxlimpct = df["DERCtlAC[0].WMaxLimPct"].to_numpy(float)
        varsetena = np.nan_to_num(df["DERCtlAC[0].VarSetEna"].to_numpy(float), nan=0.0)
        varset = df["DERCtlAC[0].VarSet"].to_numpy(float)
        varsetpct = df["DERCtlAC[0].VarSetPct"].to_numpy(float)
        wset_abs_error = np.where(wsetena > 0, np.abs(w - wset), np.nan)
        wsetpct_target = wmaxrtg * (wsetpct / 100.0)
        wsetpct_abs_error = np.where(wsetena > 0, np.abs(w - wsetpct_target), np.nan)
        wmaxlim_target = wmaxrtg * (wmaxlimpct / 100.0)
        wmaxlim_excess = np.where(wmaxlimena > 0, w - wmaxlim_target, np.nan)
        varset_abs_error = np.where(varsetena > 0, np.abs(var - varset), np.nan)
        varsetpct_target = varmaxinj * (varsetpct / 100.0)
        varsetpct_abs_error = np.where(varsetena > 0, np.abs(var - varsetpct_target), np.nan)
        data["wset_abs_error"] = wset_abs_error.astype(np.float32)
        data["wsetpct_target"] = wsetpct_target.astype(np.float32)
        data["wsetpct_abs_error"] = wsetpct_abs_error.astype(np.float32)
        data["wmaxlim_target"] = wmaxlim_target.astype(np.float32)
        data["wmaxlim_excess"] = wmaxlim_excess.astype(np.float32)
        data["varset_abs_error"] = varset_abs_error.astype(np.float32)
        data["varsetpct_target"] = varsetpct_target.astype(np.float32)
        data["varsetpct_abs_error"] = varsetpct_abs_error.astype(np.float32)
        data["wset_enabled_far"] = (
            (wsetena > 0) & (wset_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))
        ).astype(np.int8)
        data["wsetpct_enabled_far"] = (
            (wsetena > 0) & (wsetpct_abs_error > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))
        ).astype(np.int8)
        data["wmaxlim_enabled_far"] = (
            (wmaxlimena > 0) & (wmaxlim_excess > np.maximum(50.0, 0.05 * np.nan_to_num(wmaxrtg, nan=0.0)))
        ).astype(np.int8)
        data["varsetpct_enabled_far"] = (
            (varsetena > 0) & (varsetpct_abs_error > np.maximum(20.0, 0.05 * np.nan_to_num(varmaxinj, nan=0.0)))
        ).astype(np.int8)

        self._add_capacity_extension_features(
            data,
            wmaxrtg=wmaxrtg,
            wmax=wmax,
            vamaxrtg=vamaxrtg,
            vamax=vamax,
            varmaxinjrtg=varmaxinjrtg,
            varmaxinj=varmaxinj,
            varmaxabsrtg=varmaxabsrtg,
            varmaxabs=varmaxabs,
            vnomrtg=df["DERCapacity[0].VNomRtg"].to_numpy(float),
            vnom=vnom,
            vmaxrtg=df["DERCapacity[0].VMaxRtg"].to_numpy(float),
            vmax=vmax,
            vminrtg=df["DERCapacity[0].VMinRtg"].to_numpy(float),
            vmin=vmin,
            amaxrtg=df["DERCapacity[0].AMaxRtg"].to_numpy(float),
            amax=amax,
            wcha_rtg=df["DERCapacity[0].WChaRteMaxRtg"].to_numpy(float),
            wdis_rtg=df["DERCapacity[0].WDisChaRteMaxRtg"].to_numpy(float),
            vacha_rtg=df["DERCapacity[0].VAChaRteMaxRtg"].to_numpy(float),
            vadis_rtg=df["DERCapacity[0].VADisChaRteMaxRtg"].to_numpy(float),
            wcha=df["DERCapacity[0].WChaRteMax"].to_numpy(float),
            wdis=df["DERCapacity[0].WDisChaRteMax"].to_numpy(float),
            vacha=df["DERCapacity[0].VAChaRteMax"].to_numpy(float),
            vadis=df["DERCapacity[0].VADisChaRteMax"].to_numpy(float),
            pfover_rtg=df["DERCapacity[0].PFOvrExtRtg"].to_numpy(float),
            pfover=df["DERCapacity[0].PFOvrExt"].to_numpy(float),
            pfunder_rtg=df["DERCapacity[0].PFUndExtRtg"].to_numpy(float),
            pfunder=df["DERCapacity[0].PFUndExt"].to_numpy(float),
        )

        voltage_pct = 100.0 * self._safe_div(llv, vnom)
        line_neutral_voltage_pct = 100.0 * self._safe_div(lnv * math.sqrt(3.0), vnom)
        w_pct = 100.0 * self._safe_div(w, wmaxrtg)
        var_pct = self._var_pct(var, varmaxinj, varmaxabs)

        data["voltage_pct"] = voltage_pct.astype(np.float32)
        data["line_neutral_voltage_pct"] = line_neutral_voltage_pct.astype(np.float32)
        data["w_pct_of_rtg"] = w_pct.astype(np.float32)
        data["var_pct_of_limit"] = var_pct.astype(np.float32)

        enter_state_anomaly, enter_blocked_power, enter_blocked_current = self._add_enter_service_features(
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
        pf_abs_ext_present, pf_abs_rvrt_ext_present = self._add_pf_control_features(
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
            outside, power_when_outside = self._add_trip_block_features(
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

        voltvar_measure = voltage_pct - 100.0 + df["DERVoltVar[0].Crv[0].VRef"].fillna(100.0).to_numpy(float)
        self._add_curve_block_features(
            data,
            name="voltvar",
            raw_idx=df["DERVoltVar[0].AdptCrvRslt"].to_numpy(float),
            curve_x=[np.column_stack([df[f"DERVoltVar[0].Crv[{curve}].Pt[{point}].V"].to_numpy(float) for point in range(4)]) for curve in range(3)],
            curve_y=[np.column_stack([df[f"DERVoltVar[0].Crv[{curve}].Pt[{point}].Var"].to_numpy(float) for point in range(4)]) for curve in range(3)],
            curve_actpt=[df[f"DERVoltVar[0].Crv[{curve}].ActPt"].to_numpy(float) for curve in range(3)],
            curve_meta={
                "deptref": [df[f"DERVoltVar[0].Crv[{curve}].DeptRef"].to_numpy(float) for curve in range(3)],
                "pri": [df[f"DERVoltVar[0].Crv[{curve}].Pri"].to_numpy(float) for curve in range(3)],
                "vref": [df[f"DERVoltVar[0].Crv[{curve}].VRef"].to_numpy(float) for curve in range(3)],
                "vref_auto": [df[f"DERVoltVar[0].Crv[{curve}].VRefAuto"].to_numpy(float) for curve in range(3)],
                "vref_auto_ena": [df[f"DERVoltVar[0].Crv[{curve}].VRefAutoEna"].to_numpy(float) for curve in range(3)],
                "vref_auto_tms": [df[f"DERVoltVar[0].Crv[{curve}].VRefAutoTms"].to_numpy(float) for curve in range(3)],
                "rsp": [df[f"DERVoltVar[0].Crv[{curve}].RspTms"].to_numpy(float) for curve in range(3)],
                "readonly": [df[f"DERVoltVar[0].Crv[{curve}].ReadOnly"].to_numpy(float) for curve in range(3)],
            },
            measure_value=voltvar_measure,
            observed_value=var_pct,
        )

        self._add_curve_block_features(
            data,
            name="voltwatt",
            raw_idx=df["DERVoltWatt[0].AdptCrvRslt"].to_numpy(float),
            curve_x=[np.column_stack([df[f"DERVoltWatt[0].Crv[{curve}].Pt[{point}].V"].to_numpy(float) for point in range(2)]) for curve in range(3)],
            curve_y=[np.column_stack([df[f"DERVoltWatt[0].Crv[{curve}].Pt[{point}].W"].to_numpy(float) for point in range(2)]) for curve in range(3)],
            curve_actpt=[df[f"DERVoltWatt[0].Crv[{curve}].ActPt"].to_numpy(float) for curve in range(3)],
            curve_meta={
                "deptref": [df[f"DERVoltWatt[0].Crv[{curve}].DeptRef"].to_numpy(float) for curve in range(3)],
                "rsp": [df[f"DERVoltWatt[0].Crv[{curve}].RspTms"].to_numpy(float) for curve in range(3)],
                "readonly": [df[f"DERVoltWatt[0].Crv[{curve}].ReadOnly"].to_numpy(float) for curve in range(3)],
            },
            measure_value=voltage_pct,
            observed_value=w_pct,
        )

        self._add_curve_block_features(
            data,
            name="wattvar",
            raw_idx=df["DERWattVar[0].AdptCrvRslt"].to_numpy(float),
            curve_x=[np.column_stack([df[f"DERWattVar[0].Crv[{curve}].Pt[{point}].W"].to_numpy(float) for point in range(6)]) for curve in range(3)],
            curve_y=[np.column_stack([df[f"DERWattVar[0].Crv[{curve}].Pt[{point}].Var"].to_numpy(float) for point in range(6)]) for curve in range(3)],
            curve_actpt=[df[f"DERWattVar[0].Crv[{curve}].ActPt"].to_numpy(float) for curve in range(3)],
            curve_meta={
                "deptref": [df[f"DERWattVar[0].Crv[{curve}].DeptRef"].to_numpy(float) for curve in range(3)],
                "pri": [df[f"DERWattVar[0].Crv[{curve}].Pri"].to_numpy(float) for curve in range(3)],
                "readonly": [df[f"DERWattVar[0].Crv[{curve}].ReadOnly"].to_numpy(float) for curve in range(3)],
            },
            measure_value=w_pct,
            observed_value=var_pct,
        )

        self._add_freq_droop_features(data, df, hz=hz, w_pct=w_pct)
        dc_port_type_rare = self._add_dc_features(data, df, w=w, abs_w=abs_w)

        ac_type = df["DERMeasureAC[0].ACType"].to_numpy(float)
        ac_type_is_rare = np.isfinite(ac_type) & (ac_type == 3.0)
        data["ac_type_is_rare"] = ac_type_is_rare.astype(np.int8)

        model_structure_anomaly = data["model_structure_anomaly_any"].astype(np.int8)
        noncanonical_flag = data["noncanonical"] == 1
        common_missing_flag = data["common_missing_any"] == 1
        w_gt_wmax_flag = data["w_gt_wmax_tol"] == 1
        w_gt_wmaxrtg_flag = data["w_gt_wmaxrtg_tol"] == 1
        va_gt_vamax_flag = data["va_gt_vamax_tol"] == 1
        var_gt_injmax_flag = data["var_gt_injmax_tol"] == 1
        var_lt_absmax_flag = data["var_lt_absmax_tol"] == 1
        wset_far_flag = data["wset_enabled_far"] == 1
        wsetpct_far_flag = data["wsetpct_enabled_far"] == 1
        wmaxlim_far_flag = data["wmaxlim_enabled_far"] == 1
        varsetpct_far_flag = data["varsetpct_enabled_far"] == 1
        model_structure_flag = model_structure_anomaly == 1
        ac_type_rare_flag = ac_type_is_rare == 1
        dc_type_rare_flag = dc_port_type_rare == 1
        enter_state_flag = enter_state_anomaly == 1
        enter_blocked_power_flag = enter_blocked_power == 1
        enter_blocked_current_flag = enter_blocked_current == 1
        pf_abs_flag = pf_abs_ext_present == 1
        pf_abs_rvrt_flag = pf_abs_rvrt_ext_present == 1
        trip_power_flag = trip_any_power_when_outside == 1

        hard_rule_flags = np.column_stack(
            [
                noncanonical_flag,
                common_missing_flag,
                w_gt_wmax_flag,
                w_gt_wmaxrtg_flag,
                va_gt_vamax_flag,
                var_gt_injmax_flag,
                var_lt_absmax_flag,
                wset_far_flag,
                wsetpct_far_flag,
                wmaxlim_far_flag,
                varsetpct_far_flag,
                model_structure_flag,
                ac_type_rare_flag,
                dc_type_rare_flag,
                enter_state_flag,
                enter_blocked_power_flag,
                enter_blocked_current_flag,
                pf_abs_flag,
                pf_abs_rvrt_flag,
                trip_power_flag,
            ]
        )
        hard_override_flags = np.column_stack(
            [
                noncanonical_flag,
                common_missing_flag,
                w_gt_wmax_flag,
                w_gt_wmaxrtg_flag,
                va_gt_vamax_flag,
                var_gt_injmax_flag,
                var_lt_absmax_flag,
                wset_far_flag,
                wsetpct_far_flag,
                model_structure_flag,
                ac_type_rare_flag,
                dc_type_rare_flag,
                enter_state_flag,
                pf_abs_flag,
                pf_abs_rvrt_flag,
                trip_power_flag,
            ]
        )
        data["hard_rule_count"] = hard_rule_flags.sum(axis=1).astype(np.int8)
        data["hard_rule_score"] = (
            3.0 * noncanonical_flag.astype(np.float32)
            + 2.5 * common_missing_flag.astype(np.float32)
            + 2.0 * (
                w_gt_wmax_flag.astype(np.float32)
                + w_gt_wmaxrtg_flag.astype(np.float32)
                + va_gt_vamax_flag.astype(np.float32)
                + var_gt_injmax_flag.astype(np.float32)
                + var_lt_absmax_flag.astype(np.float32)
                + model_structure_flag.astype(np.float32)
                + enter_state_flag.astype(np.float32)
                + trip_power_flag.astype(np.float32)
            )
            + 1.5 * (
                wset_far_flag.astype(np.float32)
                + wsetpct_far_flag.astype(np.float32)
                + ac_type_rare_flag.astype(np.float32)
                + dc_type_rare_flag.astype(np.float32)
                + pf_abs_flag.astype(np.float32)
                + pf_abs_rvrt_flag.astype(np.float32)
            )
            + 1.0 * varsetpct_far_flag.astype(np.float32)
            + 0.75 * wmaxlim_far_flag.astype(np.float32)
            + 0.35 * (
                enter_blocked_power_flag.astype(np.float32)
                + enter_blocked_current_flag.astype(np.float32)
            )
        )
        hard_rule_anomaly = hard_rule_flags.any(axis=1).astype(np.int8)
        data["hard_rule_anomaly"] = hard_rule_anomaly
        data["hard_override_anomaly"] = hard_override_flags.any(axis=1).astype(np.int8)
        return pd.DataFrame(data)

    def iter_raw_chunks(
        self,
        zip_path: Path,
        member: str,
        usecols: Sequence[str],
        limit_rows: int = 0,
    ) -> Iterator[pd.DataFrame]:
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
    def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
        best_thr, best_f2 = 0.5, -1.0
        for thr in np.linspace(0.02, 0.98, 97):
            pred = (prob >= thr).astype(int)
            score = fbeta_score(y_true, pred, beta=2)
            if score > best_f2:
                best_thr, best_f2 = float(thr), float(score)
        return best_thr, best_f2

    def _encode_device_family(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["device_family"] = out["device_family"].map(DEVICE_FAMILY_MAP).fillna(-1).astype(np.int8)
        return out

    def _get_surrogate_feature_cols(self, columns: Sequence[str]) -> List[str]:
        excluded = {
            "Id",
            "hard_rule_anomaly",
            "hard_rule_count",
            "hard_rule_score",
            "hard_override_anomaly",
        }
        return [
            col
            for col in columns
            if col not in excluded and col not in SURROGATE_LEAKY_FEATURES
        ]

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
            "throt_pct": self._bucketize(x_df["DERMeasureAC_0_ThrotPct"], scale=5.0, fill_value=-1, dtype=np.int16),
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
        out = x_train.copy()
        y_arr = y_train.to_numpy(np.float32)
        family_series = out["device_family"].astype(str)
        self.family_base_rates = (
            pd.DataFrame({"family": family_series, "y": y_arr})
            .groupby("family")["y"]
            .mean()
            .to_dict()
        )
        keys = self._build_scenario_keys(out)
        output_keys = self._build_scenario_output_keys(out)
        fold_ids = (out["Id"].to_numpy(np.int64) % 5).astype(np.int8)
        scenario_rate = np.zeros(len(out), dtype=np.float32)
        scenario_count = np.zeros(len(out), dtype=np.int32)
        scenario_output_rate = np.zeros(len(out), dtype=np.float32)
        scenario_output_count = np.zeros(len(out), dtype=np.int32)
        global_rate = float(np.mean(y_arr))

        for fold in range(5):
            train_mask = fold_ids != fold
            valid_mask = fold_ids == fold
            if not valid_mask.any():
                continue
            stats = (
                pd.DataFrame({"key": keys[train_mask], "y": y_arr[train_mask]})
                .groupby("key")["y"]
                .agg(["sum", "count"])
            )
            output_stats = (
                pd.DataFrame({"key": output_keys[train_mask], "y": y_arr[train_mask]})
                .groupby("key")["y"]
                .agg(["sum", "count"])
            )
            valid_keys = pd.Series(keys[valid_mask])
            valid_sum = valid_keys.map(stats["sum"]).fillna(0.0).to_numpy(np.float32)
            valid_count = valid_keys.map(stats["count"]).fillna(0).to_numpy(np.int32)
            valid_output_keys = pd.Series(output_keys[valid_mask])
            valid_output_sum = valid_output_keys.map(output_stats["sum"]).fillna(0.0).to_numpy(np.float32)
            valid_output_count = valid_output_keys.map(output_stats["count"]).fillna(0).to_numpy(np.int32)
            valid_family = family_series.loc[valid_mask].tolist()
            prior = np.array([self.family_base_rates.get(name, global_rate) for name in valid_family], dtype=np.float32)
            scenario_rate[valid_mask] = (valid_sum + SCENARIO_SMOOTHING * prior) / (
                valid_count + SCENARIO_SMOOTHING
            )
            scenario_count[valid_mask] = valid_count
            scenario_output_rate[valid_mask] = (valid_output_sum + SCENARIO_SMOOTHING * prior) / (
                valid_output_count + SCENARIO_SMOOTHING
            )
            scenario_output_count[valid_mask] = valid_output_count

        full_stats = (
            pd.DataFrame({"key": keys, "y": y_arr})
            .groupby("key")["y"]
            .agg(["sum", "count"])
        )
        full_output_stats = (
            pd.DataFrame({"key": output_keys, "y": y_arr})
            .groupby("key")["y"]
            .agg(["sum", "count"])
        )
        self.scenario_sum_map = {int(idx): float(val) for idx, val in full_stats["sum"].items()}
        self.scenario_count_map = {int(idx): int(val) for idx, val in full_stats["count"].items()}
        self.scenario_output_sum_map = {
            int(idx): float(val) for idx, val in full_output_stats["sum"].items()
        }
        self.scenario_output_count_map = {
            int(idx): int(val) for idx, val in full_output_stats["count"].items()
        }

        family_prior = family_series.map(self.family_base_rates).fillna(global_rate).to_numpy(np.float32)
        return self._assign_scenario_features(
            out,
            family_prior=family_prior,
            scenario_rate=scenario_rate,
            scenario_count=scenario_count,
            scenario_output_rate=scenario_output_rate,
            scenario_output_count=scenario_output_count,
        )

    def _apply_scenario_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.scenario_count_map:
            return x_df

        out = x_df.copy()
        keys = self._build_scenario_keys(out)
        output_keys = self._build_scenario_output_keys(out)
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
        family_prior = out["device_family"].astype(str).map(self.family_base_rates).fillna(global_rate).to_numpy(np.float32)
        scenario_rate = (sum_values + SCENARIO_SMOOTHING * family_prior) / (count_values + SCENARIO_SMOOTHING)
        scenario_output_rate = (output_sum_values + SCENARIO_SMOOTHING * family_prior) / (
            output_count_values + SCENARIO_SMOOTHING
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
        divisor = 3 if self.sample_rows == 0 else 2
        fit_mask = (ids_arr % divisor) == 0
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

    def _fit_surrogate_models(self, x_train: pd.DataFrame, y_train: pd.Series, valid_mask: pd.Series) -> None:
        self.surrogate_feature_cols = self._get_surrogate_feature_cols(x_train.columns)
        fit_partition = self._surrogate_partition_mask(x_train["Id"], fit_partition=True)
        normal_mask = (
            (y_train == 0)
            & (x_train["hard_override_anomaly"] == 0)
            & (x_train["device_family"] != "other")
            & (~valid_mask.to_numpy())
            & fit_partition
        )
        surrogate_df = x_train.loc[normal_mask].copy()
        if surrogate_df.empty:
            raise RuntimeError("No rows available to train surrogate models.")

        self.surrogate_models = {}
        for family in DEVICE_FAMILY_MAP:
            family_df = surrogate_df.loc[surrogate_df["device_family"] == family].copy()
            if family_df.empty:
                continue
            x_surrogate = self._encode_device_family(family_df[self.surrogate_feature_cols])
            for target_name, (target_col, _) in SURROGATE_TARGETS.items():
                model = self._new_surrogate_model()
                y_target = family_df[target_col].to_numpy(np.float32)
                print(
                    f"[surrogate] training {family}/{target_name} on {len(family_df):,} normal rows"
                )
                model.fit(x_surrogate, y_target)
                self.surrogate_models[(family, target_name)] = model

    def _augment_with_surrogates(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if self.surrogate_feature_cols is None or not self.surrogate_models:
            return x_df

        out = x_df.copy()
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
        x_surrogate = self._encode_device_family(out[self.surrogate_feature_cols])
        for family in DEVICE_FAMILY_MAP:
            family_mask = out["device_family"] == family
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
                out.loc[family_mask, f"pred_{target_name}"] = pred
                out.loc[family_mask, f"resid_{target_name}"] = resid
                out.loc[family_mask, f"abs_resid_{target_name}"] = np.abs(resid).astype(np.float32)
                if scale_col is not None:
                    scale = out.loc[family_mask, scale_col].to_numpy(np.float32)
                    norm_resid = self._safe_div(resid, scale)
                else:
                    scale = np.maximum(0.05, np.abs(actual))
                    norm_resid = (resid / scale).astype(np.float32)
                out.loc[family_mask, f"norm_resid_{target_name}"] = norm_resid.astype(np.float32)
                out.loc[family_mask, f"abs_norm_resid_{target_name}"] = np.abs(norm_resid).astype(np.float32)

        out["resid_energy_total"] = (
            out[["abs_resid_w", "abs_resid_va", "abs_resid_var", "abs_resid_pf", "abs_resid_a"]]
            .sum(axis=1)
            .astype(np.float32)
        )
        out["resid_va_minus_pq"] = (
            out["pred_va"] - np.sqrt(np.square(out["pred_w"]) + np.square(out["pred_var"]))
        ).astype(np.float32)
        out["resid_w_var_ratio"] = self._safe_div(
            out["abs_resid_w"].to_numpy(float),
            out["abs_resid_var"].to_numpy(float) + 1e-3,
        )
        return out

    def _compute_residual_quantiles(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        valid_mask: pd.Series,
    ) -> None:
        calibration_partition = self._surrogate_partition_mask(x_train["Id"], fit_partition=False)
        base_mask = (
            (y_train == 0)
            & (x_train["hard_override_anomaly"] == 0)
            & (x_train["device_family"] != "other")
            & (~valid_mask.to_numpy())
        )
        self.residual_quantiles = {}
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
                family_quantiles[target_name] = {
                    key: max(1e-6, value) for key, value in quantiles.items()
                }
            self.residual_quantiles[family] = family_quantiles

    def _apply_residual_calibration_features(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.residual_quantiles:
            return x_df

        out = x_df.copy()
        for target_name in SURROGATE_TARGETS:
            out[f"tail_resid_{target_name}"] = 0
            out[f"extreme_resid_{target_name}"] = 0
            out[f"ultra_resid_{target_name}"] = 0
            out[f"q99_ratio_resid_{target_name}"] = np.nan

        for family in DEVICE_FAMILY_MAP:
            family_mask = out["device_family"] == family
            if not family_mask.any():
                continue
            family_quantiles = self.residual_quantiles.get(family, {})
            for target_name in SURROGATE_TARGETS:
                abs_norm = out.loc[family_mask, f"abs_norm_resid_{target_name}"].to_numpy(np.float32)
                q = family_quantiles.get(target_name, RESIDUAL_TAIL_FALLBACKS)
                tail = abs_norm >= q["tail"]
                extreme = abs_norm >= q["extreme"]
                ultra = abs_norm >= q["ultra"]
                q99_ratio = self._safe_div(abs_norm, np.full_like(abs_norm, q["extreme"], dtype=np.float32))
                out.loc[family_mask, f"tail_resid_{target_name}"] = tail.astype(np.int8)
                out.loc[family_mask, f"extreme_resid_{target_name}"] = extreme.astype(np.int8)
                out.loc[family_mask, f"ultra_resid_{target_name}"] = ultra.astype(np.int8)
                out.loc[family_mask, f"q99_ratio_resid_{target_name}"] = q99_ratio.astype(np.float32)

        abs_norm_w = np.nan_to_num(out["abs_norm_resid_w"].to_numpy(np.float32), nan=0.0)
        abs_norm_var = np.nan_to_num(out["abs_norm_resid_var"].to_numpy(np.float32), nan=0.0)
        abs_norm_pf = np.nan_to_num(out["abs_norm_resid_pf"].to_numpy(np.float32), nan=0.0)
        abs_norm_a = np.nan_to_num(out["abs_norm_resid_a"].to_numpy(np.float32), nan=0.0)
        pf_mode = np.nan_to_num(out["pf_control_any_enabled"].to_numpy(np.float32), nan=0.0) > 0
        voltvar_mode = (
            np.nan_to_num(out["DERVoltVar_0_Ena"].to_numpy(np.float32), nan=0.0) > 0
        ) & np.isfinite(out["voltvar_curve_expected"].to_numpy(np.float32))
        voltwatt_mode = (
            np.nan_to_num(out["DERVoltWatt_0_Ena"].to_numpy(np.float32), nan=0.0) > 0
        ) & np.isfinite(out["voltwatt_curve_expected"].to_numpy(np.float32))
        wattvar_mode = (
            np.nan_to_num(out["DERWattVar_0_Ena"].to_numpy(np.float32), nan=0.0) > 0
        ) & np.isfinite(out["wattvar_curve_expected"].to_numpy(np.float32))
        droop_mode = np.nan_to_num(out["freqdroop_outside_deadband"].to_numpy(np.float32), nan=0.0) > 0
        enter_idle_mode = np.nan_to_num(out["enter_service_should_idle"].to_numpy(np.float32), nan=0.0) > 0

        out["mode_resid_pf_pf"] = (abs_norm_pf * pf_mode).astype(np.float32)
        out["mode_resid_var_pf"] = (abs_norm_var * pf_mode).astype(np.float32)
        out["mode_resid_var_voltvar"] = (abs_norm_var * voltvar_mode).astype(np.float32)
        out["mode_resid_w_voltwatt"] = (abs_norm_w * voltwatt_mode).astype(np.float32)
        out["mode_resid_var_wattvar"] = (abs_norm_var * wattvar_mode).astype(np.float32)
        out["mode_resid_w_droop"] = (abs_norm_w * droop_mode).astype(np.float32)
        out["mode_resid_w_enter_idle"] = (abs_norm_w * enter_idle_mode).astype(np.float32)
        out["mode_resid_a_enter_idle"] = (abs_norm_a * enter_idle_mode).astype(np.float32)
        out["mode_curve_var_resid"] = (abs_norm_var * (voltvar_mode | wattvar_mode | pf_mode)).astype(np.float32)
        out["mode_dispatch_w_resid"] = (abs_norm_w * (voltwatt_mode | droop_mode | enter_idle_mode)).astype(np.float32)
        out["mode_extreme_var_curve"] = (
            np.nan_to_num(out["extreme_resid_var"].to_numpy(np.float32), nan=0.0)
            * (voltvar_mode | wattvar_mode | pf_mode)
        ).astype(np.int8)
        out["mode_extreme_w_dispatch"] = (
            np.nan_to_num(out["extreme_resid_w"].to_numpy(np.float32), nan=0.0)
            * (voltwatt_mode | droop_mode | enter_idle_mode)
        ).astype(np.int8)
        out["mode_tail_count"] = (
            out[
                [
                    "mode_extreme_var_curve",
                    "mode_extreme_w_dispatch",
                ]
            ]
            .sum(axis=1)
            .astype(np.int8)
        )
        out["resid_tail_count"] = (
            out[
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
        out["resid_extreme_count"] = (
            out[
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
        out["resid_ultra_count"] = (
            out[
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
        out["resid_quantile_score"] = (
            out[
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
        return out

    def _tune_family_thresholds(
        self,
        y_true: np.ndarray,
        prob: np.ndarray,
        families: Sequence[str],
        default_threshold: float,
    ) -> Dict[str, float]:
        family_arr = np.asarray(families)
        per_family_best: Dict[str, float] = {}
        for family in DEVICE_FAMILY_MAP:
            mask = family_arr == family
            if mask.sum() == 0:
                per_family_best[family] = default_threshold
                continue
            thr, _ = self.tune_threshold(y_true[mask], prob[mask])
            per_family_best[family] = thr

        lower = max(FAMILY_THRESHOLD_FLOOR, default_threshold - FAMILY_THRESHOLD_MAX_DELTA)
        upper = min(0.98, default_threshold + FAMILY_THRESHOLD_MAX_DELTA)
        canon10_center = float(np.clip(
            default_threshold + FAMILY_THRESHOLD_SHRINK * (per_family_best["canon10"] - default_threshold),
            lower,
            upper,
        ))
        canon100_center = float(np.clip(
            default_threshold + FAMILY_THRESHOLD_SHRINK * (per_family_best["canon100"] - default_threshold),
            lower,
            upper,
        ))

        mask10 = family_arr == "canon10"
        mask100 = family_arr == "canon100"
        if not mask10.any() or not mask100.any():
            return {"canon10": canon10_center, "canon100": canon100_center}

        y10 = y_true[mask10]
        y100 = y_true[mask100]
        prob10 = prob[mask10]
        prob100 = prob[mask100]
        pos10 = int(y10.sum())
        pos100 = int(y100.sum())
        neg10 = int(len(y10) - pos10)
        neg100 = int(len(y100) - pos100)

        def family_counts(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            pred = probs[None, :] >= thresholds[:, None]
            tp = (pred & (labels[None, :] == 1)).sum(axis=1)
            fp = (pred & (labels[None, :] == 0)).sum(axis=1)
            fn = int(labels.sum()) - tp
            return tp.astype(np.int64), fp.astype(np.int64), fn.astype(np.int64)

        coarse10 = np.linspace(max(0.05, canon10_center - 0.10), min(0.40, canon10_center + 0.10), 25)
        coarse100 = np.linspace(max(0.05, canon100_center - 0.10), min(0.40, canon100_center + 0.10), 25)
        tp10_c, fp10_c, fn10_c = family_counts(prob10, y10, coarse10)
        tp100_c, fp100_c, fn100_c = family_counts(prob100, y100, coarse100)

        best_score = -1.0
        best_t10 = canon10_center
        best_t100 = canon100_center
        for i, t10 in enumerate(coarse10):
            for j, t100 in enumerate(coarse100):
                tp = tp10_c[i] + tp100_c[j]
                fp = fp10_c[i] + fp100_c[j]
                fn = fn10_c[i] + fn100_c[j]
                score = (5.0 * tp) / (5.0 * tp + 4.0 * fn + fp) if tp > 0 else 0.0
                if score > best_score:
                    best_score = score
                    best_t10 = float(t10)
                    best_t100 = float(t100)

        fine10 = np.linspace(max(0.05, best_t10 - 0.03), min(0.40, best_t10 + 0.03), 25)
        fine100 = np.linspace(max(0.05, best_t100 - 0.03), min(0.40, best_t100 + 0.03), 25)
        tp10_f, fp10_f, fn10_f = family_counts(prob10, y10, fine10)
        tp100_f, fp100_f, fn100_f = family_counts(prob100, y100, fine100)
        for i, t10 in enumerate(fine10):
            for j, t100 in enumerate(fine100):
                tp = tp10_f[i] + tp100_f[j]
                fp = fp10_f[i] + fp100_f[j]
                fn = fn10_f[i] + fn100_f[j]
                score = (5.0 * tp) / (5.0 * tp + 4.0 * fn + fp) if tp > 0 else 0.0
                if score > best_score:
                    best_score = score
                    best_t10 = float(t10)
                    best_t100 = float(t100)

        return {"canon10": best_t10, "canon100": best_t100}

    def _apply_family_thresholds(self, prob: np.ndarray, families: Sequence[str]) -> np.ndarray:
        family_arr = np.asarray(families)
        pred = (prob >= self.threshold).astype(np.int8)
        for family, thr in self.family_thresholds.items():
            mask = family_arr == family
            if mask.any():
                pred[mask] = (prob[mask] >= thr).astype(np.int8)
        return pred

    def load_train_sample(self, zip_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        parts: List[pd.DataFrame] = []
        labels: List[pd.Series] = []
        total = 0
        for i, chunk in enumerate(self.iter_raw_chunks(zip_path, "train.csv", USECOLS_TRAIN, self.sample_rows)):
            parts.append(self.build_features(chunk))
            labels.append(chunk["Label"].astype(int))
            total += len(chunk)
            if i % 5 == 0:
                print(f"[train] processed {total:,} rows")
        x_train = pd.concat(parts, ignore_index=True)
        y_train = pd.concat(labels, ignore_index=True).astype(int)
        return x_train, y_train

    def fit(self, zip_path: Path) -> ValidationReport:
        x_train, y_train = self.load_train_sample(zip_path)
        valid_mask = x_train["Id"] % 5 == 0
        self._fit_surrogate_models(x_train, y_train, valid_mask)
        x_train = self._augment_with_surrogates(x_train)
        self._compute_residual_quantiles(x_train, y_train, valid_mask)
        x_train = self._apply_residual_calibration_features(x_train)
        x_train = self._fit_transform_scenario_features(x_train, y_train)
        x_train = self._add_family_interaction_features(x_train)

        hard_pred = x_train.loc[valid_mask, "hard_override_anomaly"].astype(int).to_numpy()
        y_valid_full = y_train.loc[valid_mask].to_numpy()
        hard_rule_valid_f2 = float(fbeta_score(y_valid_full, hard_pred, beta=2))
        hard_rule_valid_precision = float(precision_score(y_valid_full, hard_pred))
        hard_rule_valid_recall = float(recall_score(y_valid_full, hard_pred))

        x_model = x_train.copy()
        y_model = y_train.copy()
        residual_valid = x_model["Id"] % 5 == 0
        residual_train = ~residual_valid

        feature_cols = [col for col in x_model.columns if col not in ["Id", "hard_override_anomaly"]]
        self.feature_cols = feature_cols
        xtr = x_model.loc[residual_train, feature_cols].copy()
        xv = x_model.loc[residual_valid, feature_cols].copy()
        ytr = y_model.loc[residual_train].to_numpy()
        wtr = self._build_sample_weights(x_model.loc[residual_train], ytr)

        xtr = self._encode_device_family(xtr)
        xv = self._encode_device_family(xv)

        self.model = self._new_classifier()
        print(f"[fit] training single model on {len(xtr):,} rows with {len(feature_cols):,} features")
        self.model.fit(xtr, ytr, sample_weight=wtr)

        prob_valid = self.model.predict_proba(xv)[:, 1]

        full_valid = x_train.loc[valid_mask].copy()
        prob_full = prob_valid.copy()
        hard_override_valid = full_valid["hard_override_anomaly"].astype(np.int8).to_numpy()
        prob_for_threshold = prob_full.copy()
        prob_for_threshold[hard_override_valid == 1] = 1.0

        self.threshold, best_f2 = self.tune_threshold(y_valid_full, prob_for_threshold)
        self.family_thresholds = self._tune_family_thresholds(
            y_valid_full[hard_override_valid == 0],
            prob_full[hard_override_valid == 0],
            full_valid.loc[hard_override_valid == 0, "device_family"].tolist(),
            self.threshold,
        )
        pred_full = hard_override_valid.copy()
        model_full_mask = hard_override_valid == 0
        if model_full_mask.any():
            pred_full[model_full_mask] = self._apply_family_thresholds(
                prob_full[model_full_mask],
                full_valid.loc[model_full_mask, "device_family"].tolist(),
            )
        best_f2 = float(fbeta_score(y_valid_full, pred_full, beta=2))
        precision = float(precision_score(y_valid_full, pred_full))
        recall = float(recall_score(y_valid_full, pred_full))

        report = ValidationReport(
            threshold=self.threshold,
            f2=float(best_f2),
            precision=precision,
            recall=recall,
            canon10_threshold=self.family_thresholds.get("canon10", self.threshold),
            canon100_threshold=self.family_thresholds.get("canon100", self.threshold),
            hard_rule_valid_f2=hard_rule_valid_f2,
            hard_rule_valid_precision=hard_rule_valid_precision,
            hard_rule_valid_recall=hard_rule_valid_recall,
            sample_rows=int(len(x_train)),
            residual_train_rows=int(len(xtr)),
            residual_valid_rows=int(len(xv)),
            feature_count=int(len(feature_cols)),
            surrogate_feature_count=int(len(self.surrogate_feature_cols or [])),
        )
        self.validation_report = report
        print(
            "[fit] validation "
            f"F2={report.f2:.6f} at threshold={report.threshold:.3f}; "
            f"precision={report.precision:.4f}, recall={report.recall:.4f}"
        )
        return report

    def predict_test(self, zip_path: Path, out_csv: Path) -> None:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError("Model is not fitted.")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        residual_rows = 0
        with out_csv.open("w", encoding="utf-8") as fh:
            fh.write("Id,Label\n")
            for i, chunk in enumerate(self.iter_raw_chunks(zip_path, "test.csv", USECOLS_TEST, 0)):
                feats = self.build_features(chunk)
                feats = self._augment_with_surrogates(feats)
                feats = self._apply_residual_calibration_features(feats)
                feats = self._apply_scenario_features(feats)
                feats = self._add_family_interaction_features(feats)
                pred = feats["hard_override_anomaly"].astype(int).to_numpy()
                model_mask = feats["hard_override_anomaly"] == 0
                if model_mask.any():
                    xres = feats.loc[model_mask, self.feature_cols].copy()
                    xres = self._encode_device_family(xres)
                    prob = self.model.predict_proba(xres)[:, 1]
                    pred[model_mask.to_numpy()] = self._apply_family_thresholds(
                        prob,
                        feats.loc[model_mask, "device_family"].tolist(),
                    )
                    residual_rows += int(model_mask.sum())
                out = pd.DataFrame({"Id": feats["Id"].astype(int), "Label": pred.astype(int)})
                out.to_csv(fh, index=False, header=False)
                total_rows += len(out)
                if i % 10 == 0:
                    print(f"[test] wrote {total_rows:,} predictions")
        print(f"[test] done; total_rows={total_rows:,}, residual_rows_scored={residual_rows:,}")

    def save(self, model_path: Path, report_path: Path) -> None:
        if self.model is None or self.validation_report is None or self.feature_cols is None:
            raise RuntimeError("Nothing to save; fit the model first.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        surrogate_dir = model_path.parent / "surrogates"
        surrogate_dir.mkdir(parents=True, exist_ok=True)
        for (family, target_name), model in self.surrogate_models.items():
            model.save_model(surrogate_dir / f"{family}_{target_name}.json")
        payload = self.validation_report.as_dict()
        payload["feature_cols"] = self.feature_cols
        payload["family_thresholds"] = self.family_thresholds
        payload["surrogate_feature_cols"] = self.surrogate_feature_cols
        payload["residual_quantiles"] = self.residual_quantiles
        payload["family_base_rates"] = self.family_base_rates
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


DEFAULT_RUN_CONFIG = RunConfig()


def run_pipeline(config: RunConfig = DEFAULT_RUN_CONFIG) -> ValidationReport:
    seed_everything(config.seed)
    baseline = config.create_baseline()
    report = baseline.fit(config.zip_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / MODEL_FILENAME
    report_path = config.output_dir / REPORT_FILENAME
    baseline.save(model_path, report_path)
    final_solution_path = report_path
    final_solution_label = "validation_report"
    if config.write_test_predictions:
        submission_path = config.output_dir / f"submission_semantic_v4_single_sample{report.sample_rows}.csv"
        baseline.predict_test(config.zip_path, submission_path)
        final_solution_path = submission_path
        final_solution_label = "submission"
    print(
        f"[solution] {final_solution_label}_sha256={file_sha256(final_solution_path)} "
        f"path={final_solution_path}"
    )
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
