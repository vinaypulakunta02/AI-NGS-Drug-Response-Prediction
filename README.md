# AI-Driven NGS-Based Drug Response Prediction for Precision Oncology

## Overview

This project explores whether cancer cell-line gene expression profiles can be used to predict response to an anticancer drug using machine learning.

RNA-seq gene expression data from **DepMap** were combined with **PRISM Repurposing Secondary Screen** drug-response data to predict **Trametinib response**.

The project also uses feature selection, model comparison, explainable AI (SHAP), candidate-gene prioritization, and pathway analysis.

> **Note:** This is a preclinical computational proof-of-concept using cancer cell-line data. It is not a clinical treatment recommendation or a validated clinical prediction model.

---

## Research Question

**Can NGS-derived gene expression profiles be used to predict cancer cell-line response to Trametinib, and which genes contribute to those predictions?**

---

## Objectives

- Integrate DepMap RNA-seq expression data with PRISM drug-response data.
- Predict Trametinib response from gene-expression profiles.
- Reduce the high-dimensional gene-expression feature space.
- Compare Random Forest and XGBoost regression models.
- Evaluate model performance using a locked test set and 5-fold cross-validation.
- Identify important predictive genes using feature importance and SHAP.
- Prioritize candidate genes for further biological investigation.
- Explore biological pathways associated with the prioritized genes.

---

## Datasets

### DepMap RNA-seq Expression

**Source:** DepMap Public 26Q1

- 1,719 cancer cell lines
- 19,215 gene features
- Cell-line identifier: `ModelID`

Official source: https://depmap.org/portal/

### PRISM Drug Response

**Source:** PRISM Repurposing Secondary Screen

- Drug analysed: **Trametinib**
- Target variable: **Trametinib AUC**
- 727 cell lines in the analysed PRISM subset
- 721 common cell lines between DepMap and PRISM
- 714 cell lines with complete Trametinib AUC after matching

The original datasets are not included in this repository because of their size.

---

## Workflow

```text
DepMap RNA-seq
      ↓
Data preparation
      ↓
Match with PRISM
      ↓
714 cell lines
      ↓
Train / Test Split
      ↓
Variance Filtering
      ↓
Top 500 Feature Selection
      ↓
Random Forest + XGBoost
      ↓
Model Evaluation
      ↓
Feature Importance + SHAP
      ↓
Candidate Gene Prioritization
      ↓
Pathway Analysis
```

---

## Data Preprocessing

The original expression dataset contained **19,215 genes**.

After matching the DepMap and PRISM datasets and removing cell lines without Trametinib AUC values, **714 cell lines** remained.

The data were divided into:

- **Training:** 571 cell lines
- **Test:** 143 cell lines

The test set was kept separate and was not used for model selection.

### Feature reduction

A variance filter of:

```text
variance > 0.05
```

reduced the feature space to approximately **17,254 genes**.

The top **500 genes** based on absolute training-set correlation with Trametinib AUC were then selected for the main modelling analysis.

---

## Machine Learning Models

### Random Forest

Random Forest was used to capture nonlinear relationships between gene expression and Trametinib response.

It was selected as the **primary model** because it performed better on the locked test set.

### XGBoost

XGBoost was used as a second tree-based regression model for comparison.

---

## Model Performance

### Locked Test Set

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **0.110** | **0.136** | **0.279** |
| XGBoost | 0.114 | 0.143 | 0.205 |

### 5-Fold Cross-Validation

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| Random Forest | 0.1246 ± 0.0091 | 0.1548 ± 0.0117 | 0.2795 ± 0.0574 |
| XGBoost | 0.1250 ± 0.0064 | 0.1546 ± 0.0094 | 0.2794 ± 0.0679 |

The two models showed very similar average performance during leakage-safe cross-validation. Random Forest was retained as the primary model because it performed better on the locked test set.

---

## Important Predictive Genes

The final Random Forest feature-importance analysis highlighted genes including:

**ITGA6, CLSTN2, SH3TC2, ETV4, MED29, DIP2C, TIGD7, SPRY4, RASL10B, and DUSP6.**

These genes are **predictive features**, not proven causes of Trametinib resistance or sensitivity.

---

## Explainable AI — SHAP

**SHAP (SHapley Additive exPlanations)** was used to examine how features influenced model predictions.

SHAP helps explain the contribution of individual features to model predictions.

The analysis highlighted genes including:

**CLSTN2, ITGA6, ANXA2R, RASL10B, ETV4, SH3TC2, DIP2C, ANKRD13C, ZFP3, and TAOK2.**

SHAP explains **model behaviour** and does not establish biological causation.

---

## Candidate Gene Prioritization

Candidate genes were compared using:

1. Training-set correlation with Trametinib AUC
2. Random Forest feature importance
3. SHAP importance

A consensus ranking was used to prioritize genes that repeatedly appeared as useful signals.

Top candidates included:

**SH3TC2, ETV4, CLSTN2, ITGA6, RASL10B, LYSMD1, UCHL1, PTOV1, SYNC, and MED29.**

These candidates are **hypothesis-generating signals** requiring further validation.

---

## Biological Interpretation

Several prioritized genes have previous cancer-related evidence.

**ETV4** was an important predictive signal in this analysis and has supporting experimental evidence related to Trametinib response.

**ITGA6** has been associated with cancer progression and resistance-related signalling.

**SH3TC2** has been linked to cancer biology and MAPK-related signalling.

**CLSTN2** was among the stronger computational signals but has comparatively limited direct evidence connecting it to Trametinib response.

These observations provide biological context for further investigation and do not demonstrate that these genes cause drug resistance.

---

## Pathway Analysis

Pathway enrichment was performed on the **20 consensus candidate genes** using the **17,254 variance-filtered genes** as the background.

The analysis included:

- Gene Ontology Biological Process
- KEGG
- Reactome

Observed themes included:

- MAPK regulation
- Apoptotic processes
- ERBB signalling
- Cancer-related transcriptional regulation
- ERK/MAPK-related processes

**KEGG transcriptional misregulation in cancer** was the main pathway that reached an adjusted significance level below 0.05.

Other pathway signals were treated as exploratory because they did not pass the same multiple-testing threshold.

---

## Key Findings

- Gene-expression profiles contained measurable predictive information for Trametinib AUC in the analysed cell lines.
- Random Forest achieved **R² = 0.279** on the locked test set.
- Random Forest and XGBoost showed very similar average performance in leakage-safe cross-validation.
- **SH3TC2, ETV4, CLSTN2, and ITGA6** appeared prominently across feature-analysis approaches.
- SHAP provided an interpretable view of features influencing model predictions.
- The analysis generated candidate biological signals for further investigation.

---

## Limitations

- The analysis uses cancer cell-line data rather than patient tumour samples.
- Only one drug, Trametinib, was analysed.
- Model performance was moderate, with substantial response variation remaining unexplained.
- Statistical association does not imply causation.
- Candidate genes require independent experimental and/or external dataset validation.
- Cancer cell lines do not fully reproduce the biological complexity of human tumours.

---

## Repository Structure

```text
AI-NGS-Drug-Response-Prediction/
│
├── README.md
│
├── code/
│   └── drug_response_prediction.py
│
├── results/
│   ├── model_comparison.csv
│   ├── random_forest_test_predictions.csv
│   ├── xgboost_test_predictions.csv
│   ├── random_forest_5fold_cv.csv
│   ├── random_forest_cv_summary.csv
│   ├── xgboost_5fold_cv.csv
│   ├── xgboost_cv_summary.csv
│   ├── final_random_forest_feature_importance.csv
│   ├── random_forest_feature_importance.csv
│   ├── shap_feature_importance.csv
│   ├── combined_candidate_gene_analysis.csv
│   ├── top_500_selected_genes.csv
│   └── project_summary.csv
│
├── figures/
   ├── actual_vs_predicted_random_forest.png
   ├── actual_vs_predicted_xgboost.png
   ├── final_random_forest_feature_importance_top20.png
   └── shap_summary_top20.png
```

---

## Data Availability

The original DepMap and PRISM datasets are not included in this repository because of their size.

The analysis uses publicly available datasets from the respective sources.

DepMap: https://depmap.org/portal/

---

## Conclusion

This project connects **NGS-derived gene expression, machine learning, explainable AI, and biological interpretation** for drug-response prediction.

The results suggest that gene-expression profiles contain predictive information for Trametinib response in the analysed cancer cell lines.

More importantly, combining machine learning with SHAP helped identify **interpretable candidate genes and biological signals** that can be investigated further.

The project is intended as a **hypothesis-generating computational study**, not as a clinical prediction system.

---

## Disclaimer

This project is a **preclinical computational analysis** and should not be used to make clinical treatment decisions.

The identified genes and model predictions require independent validation before any clinical or therapeutic interpretation.
