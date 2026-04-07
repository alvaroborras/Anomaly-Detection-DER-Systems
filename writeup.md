I would like to thank the organizers for putting together the competition, and for addressing the early dataset issues so quickly.

Most of the public leaderboard eventually converged around `0.9115` (public) / `0.9117` (private), so I was happy to finish a bit above that score.

It was a fun competition, and I learned a lot from it. Thank you!

**TLDR**: [This notebook](https://www.kaggle.com/code/alvaroborras/anomaly-detection-cyber-der-final-submission) reproduces my final submission, and the full implementation is available in my [personal repository](https://github.com/alvaroborras/Anomaly-Detection-DER-Systems).

# Overview of the solution

The dataset is highly structured, and in the original CSV files it is dominated by two exact identity tuples.

More concretely, I build a 5-field fingerprint from `common[0].Mn`, `common[0].Md`, `common[0].Opt`, `common[0].Vr`, and `common[0].SN`. The two dominant tuples correspond to the 10 kW and 100 kW simulator rows, which I refer to as `canon10` and `canon100`, and I use that fingerprint to split the data into `canon10`, `canon100`, and a small noncanonical bucket.

That split drives the rest of the pipeline. The two canonical families are modeled separately, while the noncanonical rows are handled outside the main learned-model path because they already form a very strong anomaly signal in training.

From `665` selected source columns, I build semantic features that make the DER structure explicit: missingness and schema-integrity signals, measurement-vs-setting consistency, control-compliance features, protection and curve features, and AC/DC consistency checks. 

For each canonical family, the final predictor is a small ensemble of XGBoost on the semantic numeric table and CatBoost on a narrower raw + categorical table, with the blend and threshold tuned directly for F2.

![High level overview of the final pipeline](diagrams/overall_pipeline.png)

*High level view of the pipeline. Rows are first split by device fingerprint; the two canonical families go through family-specific feature engineering and models, while the small noncanonical bucket is handled separately.*

## Observations about the dataset

The raw dataset has `723` features, but many are fully empty or not especially useful for the final model. In particular, `182` features are completely null in both train and test, while there are no fully empty rows. In practice, the useful signal is not in dropping rows, but in being selective about columns and preserving informative missingness.

A second key observation is that the data is overwhelmingly concentrated around two exact fingerprints from the original CSV identity columns:

- `('DERSec', 'DER Simulator', '10 kW DER', '1.2.3', 'SN-Three-Phase')`  ->  `canon10`
- `('DERSec', 'DER Simulator 100 kW', '1.2.3.1', '1.0.0', '1100058974')`  ->  `canon100`

| Bucket | Definition | Train rows | Test rows | How I use it |
|---|---|---:|---:|---|
| `canon10` | Exact match to the canonical 10 kW simulator fingerprint | 1,181,604 | 505,993 | Family-specific learned models + overrides |
| `canon100` | Exact match to the canonical 100 kW simulator fingerprint | 1,168,218 | 501,106 | Family-specific learned models + overrides |
| `other` / noncanonical | Anything else | 13,341 | 5,686 | Identity-based anomaly bucket |

This matters because all `13,341` noncanonical training rows are anomalous. So I did not treat `other` as a third modeling family. Instead, I used it as a strong identity-based anomaly signal.

## Feature engineering

The main feature groups are:

1. **Identity and missingness**: device fingerprint, missing identity fields, per-block missingness counts, and missingness patterns.
2. **Schema integrity**: expected model IDs and lengths, used to detect malformed or structurally inconsistent rows.
3. **Physical consistency**: measurements compared against ratings and limits, plus phase-sum and power-factor consistency checks.
4. **Control compliance**: whether the reported active or reactive control state matches the observed output.
5. **Protection and curves**: enter-service logic, trip blocks, Volt-Var / Volt-Watt / Watt-Var behavior, and frequency-droop features.
6. **AC/DC consistency**: agreement between AC and DC measurements, per-port values, and sign patterns.
7. **Residual and scenario features**: surrogate-model residuals and smoothed anomaly rates for recurring operating scenarios.

## Hard overrides

I also used a small set of high-confidence hard overrides. They are simply anomaly signatures that were clean enough in training to justify overriding the learned model.

The most important case is the noncanonical bucket itself. If the 5-field fingerprint does not exactly match one of the two canonical simulator identities, the row is assigned to `other`, which acts as an identity-based anomaly bucket rather than a learned family.

Inside the canonical families, the overrides are limited to a few high-precision patterns such as:

- clear electrical envelope violations,
- large mismatches between enabled controls and measured output,
- rare metadata or AC/DC patterns,
- enter-service contradictions,
- producing power while already outside a must-trip region.

An important implementation detail is that candidate overrides are audited on the training data. Only the most precise ones remain true hard overrides; the rest can still help as features without forcing the final prediction.

## Family-specific models and ensembling

For `canon10` and `canon100`, I train separate models. This worked better than a single global classifier because the two canonical devices operate in different regimes and have different local patterns.

The main model is **XGBoost** on the full semantic numeric feature set. The companion model is **CatBoost** on a narrower raw + categorical table containing selected raw numeric columns, raw identity fields, the full device fingerprint, and additional identity / missingness-engineered features.

I also add family-specific regressors trained on normal rows to predict quantities such as active power, apparent power, reactive power, power factor, and current. Their residuals become new features for the final classifiers. For validation, I use both a primary 5-fold split based on `Id` and an audit split based on a hashed representation of the operating scenario.

## Threshold optimization

Since the competition metric is F2, I did not use the default `0.5` threshold. For each canonical family, I search directly over out-of-fold probabilities to find the threshold that maximizes F2. I do the same for the XGBoost / CatBoost blend weight, and keep the family-specific operating point that performs well on both the primary and audit splits.

## Tools used

To develop the solution, I used [uv](https://docs.astral.sh/uv/), [ruff](https://github.com/astral-sh/ruff), [Docker](https://www.docker.com/), and [Codex CLI](https://developers.openai.com/codex/cli). Most of the work was done on a MacBook Pro, without a GPU.

## Reproducibility

To keep the results reproducible, I seeded the full pipeline and used a [Kaggle official Docker image](https://console.cloud.google.com/artifacts/docker/kaggle-images/us/gcr.io/python/sha256:02c72a7c98e5e0895056901d9c715d181cd30eae392491235dfea93e6d0de3ed) for local development. In the GitHub repository, the reproducible workflow is intentionally simple: build the pinned Docker image and run the fixed `run_docker.sh` script, which executes the same repo entrypoint inside the container and writes the final `submission.csv`.

## About me

I am a software developer with a background in applied mathematics, currently based in Madrid. My interests are AI, machine learning, numerical simulation, and optimization.

If you would like to reach out, you can find me on [LinkedIn](https://www.linkedin.com/in/alvaro-borras/), on [GitHub](https://github.com/alvaroborras), or by email at alvaroborrasf (at) gmail (dot) com.
