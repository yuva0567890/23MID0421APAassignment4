

## Probabilistic Customer Segmentation and Segment Prediction Using Demographic, Psychographic, and Behavioral Data with Naive Bayes Classifiers

This repository contains the notebook, report, results, and artifacts for Lab 04 of the MDI3003 Advanced
Predictive Analytics course (SCOPE, VIT Vellore), supervised by Dr. Durgesh Kumar.

The task is a **supervised, leakage-safe customer-segment classification** problem, not clustering: every
customer already carries a predefined segment label from an existing business process, and the goal is to
predict that label for new customers using Naive Bayes classifiers.

---

## Datasets

Two datasets are used. They are **never merged** — each has a different schema, population, and target.

| Dataset | Role | Source | Target |
|---|---|---|---|
| **A — JanataHack Customer Segmentation** | Core dataset (all four required models) | [Kaggle mirror](https://www.kaggle.com/datasets/vetrirah/customer) / [Analytics Vidhya](https://www.analyticsvidhya.com/datahack/contest/janatahack-customer-segmentation) | Predefined `Segmentation` (A/B/C/D) |
| **C — UCI Online Retail II** | Research extension (behavioral RFM features) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii), DOI 10.24432/C5CG6D | No predefined target — frozen RFM-quartile pseudo-label constructed for the extension only |

Raw data files are **not** committed to this repository (see `.gitignore`) due to size and licence terms.
Download them from the links above and place them in `data/raw/` before running the notebook — see
[Setup](#setup--how-to-run).

---

## Repository Structure

```
.
├── README.md
├── notebook/
│   └── RegistrationNumber_Lab04_CustomerSegmentation.ipynb
├── report/
│   └── RegistrationNumber_Lab04_Report.pdf
├── data/
│   ├── raw/                # place downloaded CSVs here (not committed)
│   └── dataset_card.md     # dataset card: licence, schema, checksum, label definition
├── artifacts/
│   ├── split_manifest.csv
│   ├── feature_manifest.json
│   └── versions.json
├── models/
│   └── selected_pipeline.joblib
├── results/
│   ├── cv_results.csv
│   ├── classification_report.csv
│   ├── test_predictions.csv
│   ├── error_analysis.csv
│   └── data_audit.csv
├── figures/
│   └── *.png                # class distribution, confusion matrices, CV comparison, etc.
└── requirements.txt
```

---

## Models

| Model | Representation | Role |
|---|---|---|
| `DummyClassifier` | Any encoded input | Trivial most-frequent-class baseline |
| `GaussianNB` | Continuous numeric features only | Numeric-only baseline |
| `BernoulliNB` | One-hot / binarised indicators | Binary-indicator model |
| `CategoricalNB` | Non-negative integer category codes (mixed-feature) | **Core model** — natively consumes the full mixed-type feature set |

All models are compared on **identical, stratified 5-fold cross-validation** using training data only. Model
selection (by mean CV macro F1) happens **before** the locked test set is touched.

---

## Methodology Summary

1. Load and validate the dataset; confirm target and schema.
2. Remove direct identifiers (`ID` / `Customer ID`) from predictors.
3. Audit missingness, duplicates, class balance, and label provenance (circularity check).
4. Assign features to demographic / psychographic / behavioral groups.
5. Create one locked, stratified 80/20 train–test split; save the split manifest.
6. Build model-specific, leakage-safe `ColumnTransformer` preprocessing pipelines (fit on training data only).
7. Cross-validate the Dummy baseline, GaussianNB, BernoulliNB, and CategoricalNB on identical folds.
8. Run a feature-group ablation (demographic / psychographic / behavioral / combined) using one predeclared core model.
9. Select the final model from CV evidence only, then evaluate once on the locked test set.
10. Analyse errors, posterior confidence, and a validation-selected review/abstention threshold.
11. Provide a `predict_customer_segment()` function for new, unseen customer profiles.

Full technical detail — Bayes' theorem, the Naive Bayes decision rule, model configurations, all
[result tables](#results-snapshot), discussion/viva answers, and the responsible-analytics review — is in
[`report/RegistrationNumber_Lab04_Report.pdf`](report/).

---

## Results Snapshot

*(Illustrative figures — see the report and `results/` for the exact numbers from this run.)*

| Model | CV Macro F1 (mean ± SD) | Locked-Test Macro F1 |
|---|---|---|
| Dummy_most_frequent | 0.107 ± 0.004 | — |
| GaussianNB (numeric only) | 0.351 ± 0.018 | — |
| BernoulliNB | 0.462 ± 0.021 | — |
| **CategoricalNB (selected)** | **0.503 ± 0.017** | **0.496** |

Feature-group ablation (CategoricalNB): **combined features > demographic-only > behavioral-only >
psychographic-only (Var_1)**.

---

## Setup / How to Run

```bash
# 1. Clone the repository
git clone <this-repo-url>
cd <this-repo>

# 2. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download the datasets and place them in data/raw/
#    - Dataset A: customer_segmentation.csv  (from the Kaggle link above)
#    - Dataset C: online_retail_II.csv       (from the UCI link above)

# 4. Run the notebook top to bottom in a clean runtime
jupyter notebook notebook/RegistrationNumber_Lab04_CustomerSegmentation.ipynb
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
joblib
```

---

## Responsible Use

- Predictions are **posterior probabilities under a Naive Bayes model**, not certainties — low-confidence
  predictions are flagged for human review rather than auto-actioned.
- Sensitive demographic attributes (Gender, Age, marital status) are reviewed before use; the feature-group
  ablation is used to check whether behavioral features alone can carry most of the predictive signal.
- The Dataset C extension label is **derived from the same features used to predict it** and is disclosed as a
  representation-recoverability check, not as evidence of real-world predictive skill.
- This model is intended to **support**, not replace, human decision-making for campaign targeting, retention
  analysis, and service personalisation — not for autonomous pricing, exclusion, or denial-of-service decisions.

See `report/` for the full dataset cards, label-provenance/circularity audit, and risk register.

---

## Academic Integrity

External code or AI assistance used in preparing this notebook/report is acknowledged here:
`_____________________________________`. All submitted components were understood, tested, and can be
explained by the author.

## Author

**Name:** ______________________
**Registration Number:** ______________________
**Course:** MDI3003 — Advanced Predictive Analytics, SCOPE, VIT Vellore
**Instructor:** Dr. Durgesh Kumar
