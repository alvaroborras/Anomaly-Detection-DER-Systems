# Cyber-Physical Anomaly Detection for DER Systems

## Objective
Build a binary classifier that detects anomalous DER (Distributed Energy Resource) operating states from cyber-physical simulator telemetry.

- Class `0`: normal / compliant operation
- Class `1`: anomalous / compromised operation

The competition uses **F2 score**, so recall on anomalies matters more than precision.

## Verified Local Dataset Snapshot
The local CSVs below were checked directly against the files in `data/`.

| File | Purpose | Rows (excluding header) | Columns | Verified schema |
|---|---|---:|---:|---|
| `data/train.csv` | Training data | 2,363,163 | 725 | `Id + 723 raw features + Label` |
| `data/test.csv` | Inference data | 1,012,785 | 724 | `Id + 723 raw features` |
| `data/sample_submission.csv` | Submission example | 100 | 2 | First 100 `test.csv` ids only |

Exact checks against the local files:

- `Label` distribution in `train.csv`: `0 -> 1,161,352`, `1 -> 1,201,811`
- Train anomaly rate: `50.8560%`
- `Id` is unique in both `train.csv` and `test.csv`
- `test.csv` schema is exactly `train.csv` minus `Label`
- `sample_submission.csv` matches the first 100 `Id` values from `test.csv`
- Rows with all 723 feature values missing: `0` in train, `0` in test
- The same `182` feature columns are completely null in both train and test
- Per-row missing-feature counts are tightly concentrated between `183` and `192` in both splits, with modes at `184` and `186`

## Data Structure
Based on the headers and the reference review summarized in `der_eda_findings.md`, the feature space is structured telemetry from the DERSec DER Simulator, organized around SunSpec DER information-model blocks rather than anonymous tabular features.

In practice, each row mixes:

- device identity metadata,
- AC and DC measurements,
- nameplate ratings and active capability limits,
- direct control setpoints and enable flags,
- protection / ride-through curves,
- curve adaptation and reversion state.

The local CSVs do **not** include separate `DerSimControls` or `DeviceType` columns. Device-family information is embedded mainly in the `common[0]` identity block plus the model-specific feature families below.

## Schema Relationship

- Raw feature count: `723`
- `train.csv`: `Id + 723 features + Label = 725 columns`
- `test.csv`: `Id + 723 features = 724 columns`
- Target column name: `Label`

## Feature Families
These counts are verified from the actual CSV header.

| Prefix | Model context | Column count |
|---|---|---:|
| `common[0]` | SunSpec common model | 8 |
| `DERMeasureAC[0]` | Model 701 AC measurements | 66 |
| `DERCapacity[0]` | Model 702 capacity / settings | 51 |
| `DEREnterService[0]` | Model 703 enter-service logic | 13 |
| `DERCtlAC[0]` | Model 704 AC controls | 53 |
| `DERVoltVar[0]` | Model 705 Volt-Var | 64 |
| `DERVoltWatt[0]` | Model 706 Volt-Watt | 37 |
| `DERTripLV[0]` | Model 707 low-voltage trip / ride-through | 77 |
| `DERTripHV[0]` | Model 708 high-voltage trip / ride-through | 77 |
| `DERTripLF[0]` | Model 709 low-frequency trip / ride-through | 77 |
| `DERTripHF[0]` | Model 710 high-frequency trip / ride-through | 77 |
| `DERFreqDroop[0]` | Model 711 frequency-droop | 33 |
| `DERWattVar[0]` | Model 712 Watt-Var | 60 |
| `DERMeasureDC[0]` | Model 714 DC measurements | 30 |

## Verified Structural Facts

- There are zero fully empty rows, but there are `182` fully empty feature columns shared by both splits. Those are different facts and should not be conflated.
- The dataset is dominated by two canonical `common[0]` identity tuples: `DERSec / DER Simulator / 10 kW DER / 1.2.3 / SN-Three-Phase` and `DERSec / DER Simulator 100 kW / 1.2.3.1 / 1.0.0 / 1100058974`.
- Those two tuples account for `2,349,822 / 2,363,163` train rows and `1,007,099 / 1,012,785` test rows.
- The remaining `13,341` non-canonical train rows are all labeled anomalous. The analogous non-canonical tail also exists in test (`5,686` rows).
- Rows with at least one missing identity field are rare but high-value: `526` such rows exist in train and all are anomalies; `228` such rows exist in test.
- In practice, missing identity fields occur in `common[0].Mn`, `common[0].Md`, `common[0].Vr`, and `common[0].SN`. `common[0].Opt` is part of the identity tuple but has `0` missing values in both splits.
- In a 200k-row pandas sample, only seven columns are inferred as string/object-like:
  - `common[0].Mn`
  - `common[0].Md`
  - `common[0].Opt`
  - `common[0].Vr`
  - `common[0].SN`
  - `DERMeasureDC[0].Prt[0].IDStr`
  - `DERMeasureDC[0].Prt[1].IDStr`

## Practical Modeling Notes
These points are consistent with the EDA memo, but they are modeling guidance rather than file-format guarantees.

- Drop only the `182` all-null feature columns. Do not drop rows for being fully empty, because none are.
- Preserve informative missingness in the remaining columns. Rare missing patterns, especially in identity and control-related fields, are anomaly-enriched.
- Treat the 5-field `common[0]` tuple as an opaque device fingerprint rather than perfectly normalized metadata.
- Keep identity values as strings. In particular, `1100058974` and `1100058974.0` are distinct serial-like tokens, and collapsing them would destroy useful anomaly signal.
- `_SF` columns should be treated as features / metadata, not blindly re-applied as scale multipliers to the exported CSV values. The local values already look like engineering units.
- Expect useful anomaly signal from cross-field consistency:
  - measurements vs nameplate / capacity,
  - control state vs observed behavior,
  - identity metadata consistency,
  - AC / DC and phase-to-phase consistency,
  - curve completeness and activation logic.

## Evaluation Metric
Competition ranking uses **F2 score** with `beta = 2`:

\[
F_2 = \frac{5 \cdot (\text{precision} \cdot \text{recall})}{4 \cdot \text{precision} + \text{recall}}
\]

Operational implication:

- false negatives are more expensive than false positives,
- threshold tuning should optimize recall-aware behavior rather than pure accuracy.

## Submission Format
Submission must contain:

```csv
Id,Label
313994,1
899738,0
```

- `Id`: copied from `test.csv`
- `Label`: predicted class `0` or `1`

`sample_submission.csv` is only a format example for the first 100 test rows, not a full-length submission template.

## References and Resources
Competition and dataset:

- Kaggle competition overview: https://www.kaggle.com/competitions/cyber-physical-anomaly-detection-for-der-systems/overview

DERSec simulator documentation:

- DERSec DER Simulator docs index: https://dersec.io/docs/dersim/index.html
- Getting started and simulator CLI/API: https://dersec.io/docs/dersim/getting_started.html
- SunSpec Modbus interface in simulator: https://dersec.io/docs/dersim/comm/sunspec.html
- Measurements (Model 701 context): https://dersec.io/docs/dersim/power/measurements.html
- Settings (Model 702 context): https://dersec.io/docs/dersim/power/settings.html
- AC Controls (Model 704): https://dersec.io/docs/dersim/power/ac_controls.html
- Volt-Var behavior: https://dersec.io/docs/dersim/power/volt_var.html
- Volt-Watt behavior: https://dersec.io/docs/dersim/power/volt_watt.html
- Watt-Var behavior: https://dersec.io/docs/dersim/power/watt_var.html
- Frequency-droop behavior: https://dersec.io/docs/dersim/power/frequency_droop.html
- Compromised/data-falsification modes: https://dersec.io/docs/dersim/hacked/data_falsification.html

SunSpec and standards:

- SunSpec DER Information Model Specification (official PDF): https://sunspec.org/wp-content/uploads/2021/02/SunSpec-DER-Information-Model-Specification-V1-0-02-01-2021.pdf
- SunSpec model registry reference (community docs): https://docs.rs/sunspec/latest/sunspec/models/index.html
- IEEE 1547-2018 standard page: https://standards.ieee.org/standard/1547-2018.html
- SunSpec certification entry for DERSec DER Simulator (model coverage): https://sunspec.org/contributing-members/der-security-corp/
