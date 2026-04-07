"""Schema-owned metadata for raw DER columns and structured block layouts."""

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def dedupe(columns: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(columns))


def prefixed(prefix: str, fields: Sequence[str]) -> List[str]:
    return [f"{prefix}.{field}" for field in fields]


ADAPTIVE_CURVE_HEADER_FIELDS = (
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


def build_adaptive_curve_columns(
    prefix: str,
    *,
    curve_scalar_fields: Sequence[str],
    point_fields: Sequence[str],
    point_count: int,
    curve_count: int = 3,
) -> List[str]:
    cols = prefixed(prefix, ADAPTIVE_CURVE_HEADER_FIELDS)
    for curve in range(curve_count):
        curve_prefix = f"{prefix}.Crv[{curve}]"
        cols.extend(f"{curve_prefix}.{field}" for field in curve_scalar_fields)
        for point in range(point_count):
            cols.extend(f"{curve_prefix}.Pt[{point}].{field}" for field in point_fields)
    return cols


def build_repeated_child_columns(
    prefix: str,
    *,
    header_fields: Sequence[str],
    child_label: str,
    child_fields: Sequence[str],
    child_count: int,
) -> List[str]:
    cols = prefixed(prefix, header_fields)
    for child in range(child_count):
        child_prefix = f"{prefix}.{child_label}[{child}]"
        cols.extend(f"{child_prefix}.{field}" for field in child_fields)
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


@dataclass(frozen=True)
class CurveFeatureSpec:
    """Static layout for one adaptive curve block."""

    prefix: str
    point_count: int
    x_field: str
    y_field: str
    meta_fields: Dict[str, str]


COMMON_FIELDS = "Mn Md Opt Vr SN".split()
COMMON_STR = prefixed("common[0]", COMMON_FIELDS)
COMMON_COLUMNS = prefixed("common[0]", ["ID", "L", *COMMON_FIELDS, "DA"])

MEASURE_AC_FIELDS = """
ID L ACType W VA Var PF A LLV LNV Hz TmpAmb TmpCab TmpSnk TmpTrns TmpSw TmpOt
ThrotPct ThrotSrc WL1 WL2 WL3 VAL1 VAL2 VAL3 VarL1 VarL2 VarL3 PFL1 PFL2 PFL3
AL1 AL2 AL3 VL1L2 VL2L3 VL3L1 VL1 VL2 VL3
""".split()
MEASURE_AC_COLUMNS = prefixed("DERMeasureAC[0]", MEASURE_AC_FIELDS)

CAPACITY_FIELDS = """
ID L WMaxRtg VAMaxRtg VarMaxInjRtg VarMaxAbsRtg WChaRteMaxRtg WDisChaRteMaxRtg
VAChaRteMaxRtg VADisChaRteMaxRtg VNomRtg VMaxRtg VMinRtg AMaxRtg PFOvrExtRtg
PFUndExtRtg NorOpCatRtg AbnOpCatRtg IntIslandCatRtg WMax WMaxOvrExt WOvrExtPF
WMaxUndExt WUndExtPF VAMax VarMaxInj VarMaxAbs WChaRteMax WDisChaRteMax
VAChaRteMax VADisChaRteMax VNom VMax VMin AMax PFOvrExt PFUndExt CtrlModes
IntIslandCat
""".split()
CAPACITY_COLUMNS = prefixed("DERCapacity[0]", CAPACITY_FIELDS)

ENTER_SERVICE_FIELDS = "ID L ES ESVHi ESVLo ESHzHi ESHzLo ESDlyTms ESRndTms ESRmpTms ESDlyRemTms".split()
ENTER_SERVICE_COLUMNS = prefixed("DEREnterService[0]", ENTER_SERVICE_FIELDS)

CTL_AC_FIELDS = """
ID L PFWInjEna PFWInjEnaRvrt PFWInjRvrtTms PFWInjRvrtRem PFWAbsEna PFWAbsEnaRvrt
PFWAbsRvrtTms PFWAbsRvrtRem WMaxLimPctEna WMaxLimPct WMaxLimPctRvrt
WMaxLimPctEnaRvrt WMaxLimPctRvrtTms WMaxLimPctRvrtRem WSetEna WSetMod WSet
WSetRvrt WSetPct WSetPctRvrt WSetEnaRvrt WSetRvrtTms WSetRvrtRem VarSetEna
VarSetMod VarSetPri VarSet VarSetRvrt VarSetPct VarSetPctRvrt VarSetEnaRvrt
VarSetRvrtTms VarSetRvrtRem WRmp WRmpRef VarRmp AntiIslEna PFWInj.PF PFWInj.Ext
PFWInjRvrt.PF PFWInjRvrt.Ext PFWAbs.Ext PFWAbsRvrt.Ext
""".split()
CTL_AC_COLUMNS = prefixed("DERCtlAC[0]", CTL_AC_FIELDS)

VOLT_VAR_COLUMNS = build_adaptive_curve_columns(
    "DERVoltVar[0]",
    curve_scalar_fields=("ActPt", "DeptRef", "Pri", "VRef", "VRefAuto", "VRefAutoEna", "VRefAutoTms", "RspTms", "ReadOnly"),
    point_fields=("V", "Var"),
    point_count=4,
)
VOLT_WATT_COLUMNS = build_adaptive_curve_columns(
    "DERVoltWatt[0]",
    curve_scalar_fields=("ActPt", "DeptRef", "RspTms", "ReadOnly"),
    point_fields=("V", "W"),
    point_count=2,
)
FREQ_DROOP_COLUMNS = build_repeated_child_columns(
    "DERFreqDroop[0]",
    header_fields=("ID", "L", "Ena", "AdptCtlReq", "AdptCtlRslt", "NCtl", "RvrtTms", "RvrtRem", "RvrtCtl"),
    child_label="Ctl",
    child_fields=("DbOf", "DbUf", "KOf", "KUf", "RspTms", "PMin", "ReadOnly"),
    child_count=3,
)
WATT_VAR_COLUMNS = build_adaptive_curve_columns(
    "DERWattVar[0]",
    curve_scalar_fields=("ActPt", "DeptRef", "Pri", "ReadOnly"),
    point_fields=("W", "Var"),
    point_count=6,
)

TRIP_SPECS: Dict[str, Tuple[str, str, str]] = {
    "lv": ("DERTripLV[0]", "V", "low"),
    "hv": ("DERTripHV[0]", "V", "high"),
    "lf": ("DERTripLF[0]", "Hz", "low"),
    "hf": ("DERTripHF[0]", "Hz", "high"),
}
TRIP_COLUMNS = {short_name: build_trip_columns(prefix, axis_name) for short_name, (prefix, axis_name, _) in TRIP_SPECS.items()}

# These expected IDs and lengths belong with the raw schema definition rather
# than with model-training configuration.
EXPECTED_MODEL_META = {
    "common": ("common[0].ID", "common[0].L", 1.0, 66.0),
    "measure_ac": ("DERMeasureAC[0].ID", "DERMeasureAC[0].L", 701.0, 153.0),
    "capacity": ("DERCapacity[0].ID", "DERCapacity[0].L", 702.0, 50.0),
    "enter_service": ("DEREnterService[0].ID", "DEREnterService[0].L", 703.0, 17.0),
    "measure_dc": ("DERMeasureDC[0].ID", "DERMeasureDC[0].L", 714.0, 68.0),
}

CURVE_FEATURE_SPECS = {
    "voltvar": CurveFeatureSpec(
        prefix="DERVoltVar[0]",
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
    ),
    "voltwatt": CurveFeatureSpec(
        prefix="DERVoltWatt[0]",
        point_count=2,
        x_field="V",
        y_field="W",
        meta_fields={
            "deptref": "DeptRef",
            "rsp": "RspTms",
            "readonly": "ReadOnly",
        },
    ),
    "wattvar": CurveFeatureSpec(
        prefix="DERWattVar[0]",
        point_count=6,
        x_field="W",
        y_field="Var",
        meta_fields={
            "deptref": "DeptRef",
            "pri": "Pri",
            "readonly": "ReadOnly",
        },
    ),
}

MEASURE_DC_FIELDS = """
ID L NPrt DCA DCW Prt[0].PrtTyp Prt[0].ID Prt[0].DCA Prt[0].DCV Prt[0].DCW
Prt[0].Tmp Prt[1].PrtTyp Prt[1].ID Prt[1].DCA Prt[1].DCV Prt[1].DCW Prt[1].Tmp
""".split()
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

CURVE_BLOCK_META_FIELDS = "Ena AdptCrvReq AdptCrvRslt NPt NCrv RvrtTms RvrtRem RvrtCrv".split()
FREQ_DROOP_META_FIELDS = "Ena AdptCtlReq AdptCtlRslt NCtl RvrtTms RvrtRem RvrtCtl".split()
TRIP_META_FIELDS = "Ena AdptCrvReq AdptCrvRslt NPt NCrvSet".split()

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

SAFE_RAW = {c: re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in RAW_NUMERIC}
SAFE_STR = {c: re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in RAW_STRING_COLUMNS}
