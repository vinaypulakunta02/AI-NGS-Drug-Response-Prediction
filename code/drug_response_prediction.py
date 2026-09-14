# ============================================================
# AI-DRIVEN NGS-BASED DRUG RESPONSE PREDICTION
# Precision Oncology - Trametinib
#
# GitHub version of the analysis performed in Google Colab.
# This script reflects the actual workflow used in the project.
# ============================================================

# Required packages:
# pandas, numpy, matplotlib, scikit-learn, xgboost, shap, gseapy
#
# In Google Colab, install with:
# # !pip -q install xgboost shap gseapy
#
# The input datasets are public DepMap/PRISM files and are not
# included in this repository because of their size.

# ============================================================
# AI-DRIVEN NGS-BASED DRUG RESPONSE PREDICTION
# Precision Oncology - Trametinib
#
# COMPLETE END-TO-END PIPELINE
# ============================================================


# ============================================================
# 0. INSTALL / IMPORT REQUIRED LIBRARIES
# ============================================================

# !pip -q install xgboost shap gseapy

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import shap
import gseapy as gp


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_path = "/content/drive/MyDrive/AI_NGS_DRUG_RESPONSE"

expression_path = os.path.join(
    project_path,
    "Expression_(Short-read)_Public_26Q1_subsetted.csv"
)

drug_path = os.path.join(
    project_path,
    "Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv"
)

results_path = os.path.join(project_path, "results")
figures_path = os.path.join(project_path, "figures")

os.makedirs(results_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

print("=" * 70)
print("PROJECT INITIALIZATION")
print("=" * 70)

print("Expression file:", expression_path)
print("Drug-response file:", drug_path)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

expression_df = pd.read_csv(expression_path)
drug_df = pd.read_csv(drug_path)

print("Expression dataset shape:", expression_df.shape)
print("Drug-response dataset shape:", drug_df.shape)


# ============================================================
# 3. INSPECT DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: BASIC DATA INSPECTION")
print("=" * 70)

print("\nExpression dataset:")
print(expression_df.head())

print("\nDrug-response dataset:")
print(drug_df.head())

print("\nExpression columns:")
print(expression_df.columns[:10].tolist())

print("\nDrug-response columns:")
print(drug_df.columns[:10].tolist())


# ============================================================
# 4. IDENTIFY CELL-LINE ID COLUMN
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: IDENTIFYING CELL-LINE IDs")
print("=" * 70)

# The first column contains DepMap model IDs
expression_df = expression_df.rename(
    columns={expression_df.columns[0]: "ModelID"}
)

drug_df = drug_df.rename(
    columns={drug_df.columns[0]: "ModelID"}
)

print("Expression ID column:", expression_df.columns[0])
print("Drug-response ID column:", drug_df.columns[0])


# ============================================================
# 5. IDENTIFY TRAMETINIB COLUMN
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: IDENTIFYING TRAMETINIB RESPONSE")
print("=" * 70)

trametinib_columns = [
    col for col in drug_df.columns
    if "TRAMETINIB" in str(col).upper()
]

print("Trametinib columns found:")
for col in trametinib_columns:
    print(col)

if len(trametinib_columns) == 0:
    raise ValueError("Trametinib column was not found.")

trametinib_col = trametinib_columns[0]

print("\nSelected target column:")
print(trametinib_col)


# ============================================================
# 6. MATCH RNA-SEQ AND DRUG RESPONSE CELL LINES
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: MATCHING CELL LINES")
print("=" * 70)

expression_ids = set(expression_df["ModelID"])
drug_ids = set(drug_df["ModelID"])

common_ids = sorted(
    expression_ids.intersection(drug_ids)
)

print("RNA-seq cell lines:", len(expression_ids))
print("PRISM cell lines:", len(drug_ids))
print("Common cell lines:", len(common_ids))


# Keep only matched cell lines
expression_matched = (
    expression_df[
        expression_df["ModelID"].isin(common_ids)
    ]
    .copy()
)

drug_matched = (
    drug_df[
        drug_df["ModelID"].isin(common_ids)
    ]
    .copy()
)

# Sort both datasets by ModelID
expression_matched = expression_matched.sort_values("ModelID")
drug_matched = drug_matched.sort_values("ModelID")

expression_matched = expression_matched.reset_index(drop=True)
drug_matched = drug_matched.reset_index(drop=True)

print("\nMatched expression shape:",
      expression_matched.shape)

print("Matched drug-response shape:",
      drug_matched.shape)


# ============================================================
# 7. EXTRACT TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: EXTRACTING TRAMETINIB AUC")
print("=" * 70)

target_df = drug_matched[
    ["ModelID", trametinib_col]
].copy()

target_df = target_df.rename(
    columns={trametinib_col: "Trametinib_AUC"}
)

print(target_df.head())

print("\nMissing Trametinib values:",
      target_df["Trametinib_AUC"].isna().sum())


# ============================================================
# 8. COMBINE EXPRESSION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: PREPARING ML DATASET")
print("=" * 70)

# Merge using ModelID
merged_df = expression_matched.merge(
    target_df,
    on="ModelID",
    how="inner"
)

# Remove samples with missing target
merged_df = merged_df.dropna(
    subset=["Trametinib_AUC"]
)

merged_df = merged_df.reset_index(drop=True)

print("Final merged dataset:",
      merged_df.shape)


# ============================================================
# 9. CREATE X AND y
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: CREATING FEATURES AND TARGET")
print("=" * 70)

gene_columns = [
    col for col in expression_matched.columns
    if col != "ModelID"
]

X = merged_df[gene_columns].copy()

y = merged_df["Trametinib_AUC"].copy()

sample_ids = merged_df["ModelID"].copy()

# Make sure expression values are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Missing expression values
print("Number of missing expression values:",
      X.isna().sum().sum())

# If any missing values exist, replace with column median
if X.isna().sum().sum() > 0:
    X = X.fillna(X.median())

print("Number of samples:", X.shape[0])
print("Number of genes:", X.shape[1])
print("Target size:", y.shape)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    sample_ids,
    test_size=0.20,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 11. VARIANCE FILTERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 10: VARIANCE FILTERING")
print("=" * 70)

# Calculate variance ONLY on training data
gene_variances = X_train.var()

# Remove zero / near-zero variance genes
variance_threshold = 0.05

selected_variance_genes = gene_variances[
    gene_variances > variance_threshold
].index.tolist()

X_train_var = X_train[selected_variance_genes].copy()
X_test_var = X_test[selected_variance_genes].copy()

print("Genes before variance filtering:",
      X_train.shape[1])

print("Genes after variance filtering:",
      X_train_var.shape[1])


# ============================================================
# 12. CORRELATION-BASED FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 11: CORRELATION-BASED FEATURE SELECTION")
print("=" * 70)

# Correlation calculated ONLY using training data
correlations = X_train_var.corrwith(y_train)

correlations = correlations.dropna()

absolute_correlations = correlations.abs()

# Rank all eligible genes
correlation_ranking = (
    absolute_correlations
    .sort_values(ascending=False)
)

print("\nTop 20 genes by absolute correlation:")

print(
    pd.DataFrame({
        "Gene": correlation_ranking.head(20).index,
        "Correlation": correlations[
            correlation_ranking.head(20).index
        ].values
    })
)


# ============================================================
# 13. SELECT TOP 500 GENES
# ============================================================

TOP_K = 500

top_500_genes = (
    correlation_ranking
    .head(TOP_K)
    .index
    .tolist()
)

X_train_selected = X_train_var[top_500_genes].copy()
X_test_selected = X_test_var[top_500_genes].copy()

print("\nSelected genes:", len(top_500_genes))
print("Training matrix:", X_train_selected.shape)
print("Testing matrix:", X_test_selected.shape)


# Save selected genes
pd.DataFrame({
    "Gene": top_500_genes
}).to_csv(
    os.path.join(
        results_path,
        "top_500_selected_genes.csv"
    ),
    index=False
)


# ============================================================
# 14. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_train_data, X_test_data,
                   y_train_data, y_test_data):

    model.fit(X_train_data, y_train_data)

    predictions = model.predict(X_test_data)

    mae = mean_absolute_error(
        y_test_data,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_data,
            predictions
        )
    )

    r2 = r2_score(
        y_test_data,
        predictions
    )

    return model, predictions, mae, rmse, r2


# ============================================================
# 15. COMPARE FEATURE NUMBERS
# ============================================================

print("\n" + "=" * 70)
print("STEP 12: RANDOM FOREST FEATURE COMPARISON")
print("=" * 70)

feature_sizes = [50, 100, 200, 500]

rf_feature_results = []

for k in feature_sizes:

    genes_k = top_500_genes[:k]

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train_selected[genes_k],
        y_train
    )

    pred = rf.predict(
        X_test_selected[genes_k]
    )

    rf_feature_results.append({
        "Features": k,
        "MAE": mean_absolute_error(
            y_test,
            pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        ),
        "R2": r2_score(
            y_test,
            pred
        )
    })

rf_feature_results_df = pd.DataFrame(
    rf_feature_results
)

print(rf_feature_results_df)


# ============================================================
# 16. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("STEP 13: LINEAR REGRESSION")
print("=" * 70)

linear_results = []

for k in feature_sizes:

    genes_k = top_500_genes[:k]

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(
        X_train_selected[genes_k]
    )

    Xte = scaler.transform(
        X_test_selected[genes_k]
    )

    linear_model = LinearRegression()

    linear_model.fit(
        Xtr,
        y_train
    )

    pred = linear_model.predict(Xte)

    linear_results.append({
        "Model": "Linear Regression",
        "Features": k,
        "MAE": mean_absolute_error(
            y_test,
            pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        ),
        "R2": r2_score(
            y_test,
            pred
        )
    })

linear_results_df = pd.DataFrame(
    linear_results
)

print(linear_results_df)




# ============================================================
# LEAKAGE-SAFE 5-FOLD CROSS-VALIDATION
# RANDOM FOREST REGRESSION
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. Create 5-fold cross-validation
# ------------------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 2. Store results from each fold
# ------------------------------------------------------------

rf_cv_results = []


# ------------------------------------------------------------
# 3. Run 5-fold CV
# ------------------------------------------------------------

for fold, (train_idx, val_idx) in enumerate(
    kf.split(X), start=1
):

    print("\n" + "="*60)
    print(f"RANDOM FOREST - FOLD {fold}")
    print("="*60)


    # --------------------------------------------------------
    # Split into training and validation data
    # --------------------------------------------------------

    X_fold_train = X.iloc[train_idx]
    X_fold_val = X.iloc[val_idx]

    y_fold_train = y.iloc[train_idx]
    y_fold_val = y.iloc[val_idx]


    # --------------------------------------------------------
    # STEP 1: Variance filtering
    # --------------------------------------------------------
    # IMPORTANT:
    # Fit the variance filter ONLY on the training portion.
    #
    # This prevents information from the validation fold
    # from influencing feature selection.

    variances = X_fold_train.var()

    variance_genes = variances[
        variances > 0.05
    ].index.tolist()

    X_fold_train_var = X_fold_train[
        variance_genes
    ]

    X_fold_val_var = X_fold_val[
        variance_genes
    ]


    print(
        "Genes after variance filtering:",
        len(variance_genes)
    )


    # --------------------------------------------------------
    # STEP 2: Select top 500 genes
    # --------------------------------------------------------
    # f_regression uses ONLY the training fold.
    #
    # Therefore, the validation fold is still completely
    # unseen during feature selection.

    selector = SelectKBest(
        score_func=f_regression,
        k=500
    )

    X_fold_train_selected = selector.fit_transform(
        X_fold_train_var,
        y_fold_train
    )

    X_fold_val_selected = selector.transform(
        X_fold_val_var
    )


    print(
        "Genes selected:",
        X_fold_train_selected.shape[1]
    )


    # --------------------------------------------------------
    # STEP 3: Train Random Forest
    # --------------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )


    rf_model.fit(
        X_fold_train_selected,
        y_fold_train
    )


    # --------------------------------------------------------
    # STEP 4: Predict validation fold
    # --------------------------------------------------------

    y_pred = rf_model.predict(
        X_fold_val_selected
    )


    # --------------------------------------------------------
    # STEP 5: Calculate evaluation metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_fold_val,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_fold_val,
            y_pred
        )
    )

    r2 = r2_score(
        y_fold_val,
        y_pred
    )


    # --------------------------------------------------------
    # Store fold results
    # --------------------------------------------------------

    rf_cv_results.append({
        "Fold": fold,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")


# ============================================================
# 4. Convert results to DataFrame
# ============================================================

rf_cv_df = pd.DataFrame(
    rf_cv_results
)


# ============================================================
# 5. Calculate mean ± standard deviation
# ============================================================

rf_cv_summary = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],

    "Mean": [
        rf_cv_df["MAE"].mean(),
        rf_cv_df["RMSE"].mean(),
        rf_cv_df["R2"].mean()
    ],

    "Std": [
        rf_cv_df["MAE"].std(),
        rf_cv_df["RMSE"].std(),
        rf_cv_df["R2"].std()
    ]
})


# ============================================================
# 6. Display fold results
# ============================================================

print("\n" + "="*60)
print("RANDOM FOREST 5-FOLD CV RESULTS")
print("="*60)

print(rf_cv_df)


# ============================================================
# 7. Display summary
# ============================================================

print("\n" + "="*60)
print("RANDOM FOREST CV SUMMARY")
print("="*60)

print(rf_cv_summary)


# ============================================================
# 8. Save results to Google Drive
# ============================================================

results_dir = (
    "/content/drive/MyDrive/"
    "AI_NGS_DRUG_RESPONSE/results"
)

rf_cv_df.to_csv(
    f"{results_dir}/random_forest_5fold_cv.csv",
    index=False
)

rf_cv_summary.to_csv(
    f"{results_dir}/random_forest_cv_summary.csv",
    index=False
)


print("\nResults saved successfully!")
print(
    f"{results_dir}/random_forest_5fold_cv.csv"
)
print(
    f"{results_dir}/random_forest_cv_summary.csv"
)

# ============================================================
# FINAL RANDOM FOREST MODEL ANALYSIS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. TRAIN FINAL RANDOM FOREST
# ============================================================

print("=" * 60)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 60)

final_rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Use the variables that actually exist in your notebook
final_rf.fit(
    X_train_selected,
    y_train
)

print("Final Random Forest training completed.")


# ============================================================
# 2. PREDICT ON LOCKED TEST SET
# ============================================================

y_pred_rf = final_rf.predict(
    X_test_selected
)

print("Test-set prediction completed.")


# ============================================================
# 3. CALCULATE FINAL TEST METRICS
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    y_pred_rf
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_rf
    )
)

rf_r2 = r2_score(
    y_test,
    y_pred_rf
)


print("\n" + "=" * 60)
print("FINAL RANDOM FOREST TEST PERFORMANCE")
print("=" * 60)

print(f"MAE  : {rf_mae:.6f}")
print(f"RMSE : {rf_rmse:.6f}")
print(f"R²   : {rf_r2:.6f}")


# ============================================================
# 4. CREATE RESULTS DIRECTORIES
# ============================================================

results_dir = (
    "/content/drive/MyDrive/"
    "AI_NGS_DRUG_RESPONSE/results"
)

figures_dir = (
    "/content/drive/MyDrive/"
    "AI_NGS_DRUG_RESPONSE/figures"
)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# ============================================================
# 5. SAVE TEST PREDICTIONS
# ============================================================

rf_predictions = pd.DataFrame({
    "ModelID": ids_test,
    "Actual_Trametinib_AUC": y_test.values,
    "Predicted_Trametinib_AUC": y_pred_rf
})

rf_predictions.to_csv(
    f"{results_dir}/random_forest_test_predictions.csv",
    index=False
)


# ============================================================
# 6. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.7
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    y_pred_rf.min()
)

max_value = max(
    y_test.max(),
    y_pred_rf.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Trametinib AUC")
plt.ylabel("Predicted Trametinib AUC")

plt.title(
    "Random Forest: Actual vs Predicted Trametinib AUC"
)

plt.text(
    0.05,
    0.95,
    f"R² = {rf_r2:.4f}\n"
    f"MAE = {rf_mae:.4f}\n"
    f"RMSE = {rf_rmse:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    f"{figures_dir}/actual_vs_predicted_random_forest.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({
    "Gene": X_train_selected.columns,
    "Importance": final_rf.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 8. DISPLAY TOP 20 GENES
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 RANDOM FOREST FEATURES")
print("=" * 60)

print(
    rf_importance.head(20)
)


# ============================================================
# 9. SAVE FEATURE IMPORTANCE
# ============================================================

rf_importance.to_csv(
    f"{results_dir}/final_random_forest_feature_importance.csv",
    index=False
)


# ============================================================
# 10. PLOT TOP 20 FEATURES
# ============================================================

top20 = rf_importance.head(20).sort_values(
    "Importance"
)

plt.figure(figsize=(8, 7))

plt.barh(
    top20["Gene"],
    top20["Importance"]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Gene")

plt.title(
    "Top 20 Predictive Genes — Final Random Forest"
)

plt.tight_layout()

plt.savefig(
    f"{figures_dir}/final_random_forest_feature_importance_top20.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL RANDOM FOREST ANALYSIS COMPLETE")
print("=" * 60)

print(f"Test MAE  : {rf_mae:.6f}")
print(f"Test RMSE : {rf_rmse:.6f}")
print(f"Test R²   : {rf_r2:.6f}")

print("\nFiles saved successfully:")
print("random_forest_test_predictions.csv")
print("final_random_forest_feature_importance.csv")
print("actual_vs_predicted_random_forest.png")
print("final_random_forest_feature_importance_top20.png")

# ============================================================
# CORRECT PATHWAY ENRICHMENT ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import gseapy as gp
import os

# ------------------------------------------------------------
# 1. Define the top candidate genes
# ------------------------------------------------------------

top_genes = [
    'SH3TC2', 'ETV4', 'CLSTN2', 'ITGA6', 'RASL10B',
    'LYSMD1', 'UCHL1', 'PTOV1', 'SYNC', 'MED29',
    'UBFD1', 'ANKRD13C', 'ZNF730', 'ZNF248', 'TIGD7',
    'DUSP6', 'TMEM25', 'DNAH10OS', 'CASTOR2', 'GADD45G'
]

print("Number of genes for pathway analysis:", len(top_genes))
print("\nTop genes:")
print(top_genes)


# ------------------------------------------------------------
# 2. Recreate the CORRECT background gene universe
# ------------------------------------------------------------
# IMPORTANT:
# The background is NOT the 500 selected genes.
#
# We start with the training expression data.
# The variance filter was:
#
#     variance > 0.05
#
# This gives approximately 17,254 genes.

variance = X_train.var()

selected_variance_genes = variance[variance > 0.05].index.tolist()

print("\n" + "="*60)
print("BACKGROUND GENE UNIVERSE")
print("="*60)

print("Number of background genes:",
      len(selected_variance_genes))


# ------------------------------------------------------------
# 3. Check that candidate genes are actually in background
# ------------------------------------------------------------

valid_genes = [
    gene for gene in top_genes
    if gene in selected_variance_genes
]

missing_genes = [
    gene for gene in top_genes
    if gene not in selected_variance_genes
]

print("\nCandidate genes found in background:",
      len(valid_genes))

if missing_genes:
    print("\nGenes NOT found in background:")
    print(missing_genes)

else:
    print("\nAll candidate genes are present in background.")


# ------------------------------------------------------------
# 4. Create output directory
# ------------------------------------------------------------

results_dir = "/content/drive/MyDrive/AI_NGS_DRUG_RESPONSE/results"
figures_dir = "/content/drive/MyDrive/AI_NGS_DRUG_RESPONSE/figures"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# ------------------------------------------------------------
# 5. Run pathway enrichment
# ------------------------------------------------------------
#
# organism = "human"
# NOT "Human"
#
# We use the 17,254 variance-filtered genes as background.

pathway_results = {}

gene_sets = {
    "GO_Biological_Process_2023":
        "GO_Biological_Process_2023",

    "KEGG_2021_Human":
        "KEGG_2021_Human",

    "Reactome_2022":
        "Reactome_2022"
}


for name, gene_set in gene_sets.items():

    print("\n" + "="*60)
    print("Running:", name)
    print("="*60)

    try:

        enr = gp.enrichr(
            gene_list=valid_genes,
            gene_sets=gene_set,
            organism="human",
            background=selected_variance_genes,
            outdir=None
        )

        result = enr.results

        pathway_results[name] = result

        print("\nPathway analysis successful!")

        print("\nTop enriched pathways:")

        print(
            result[
                [
                    "Term",
                    "Overlap",
                    "P-value",
                    "Adjusted P-value",
                    "Combined Score"
                ]
            ].head(10)
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        output_file = os.path.join(
            results_dir,
            f"{name}_enrichment.csv"
        )

        result.to_csv(output_file, index=False)

        print("\nSaved to:")
        print(output_file)

    except Exception as e:

        print("\nERROR:")
        print(type(e).__name__, ":", e)


# ------------------------------------------------------------
# 6. Check final status
# ------------------------------------------------------------

print("\n" + "="*60)
print("PATHWAY ANALYSIS SUMMARY")
print("="*60)

if pathway_results:

    print(
        "Successfully completed:",
        list(pathway_results.keys())
    )

else:

    print(
        "No pathway analysis completed successfully."
    )

print("\nCorrect background:",
      len(selected_variance_genes),
      "genes")

print("Candidate genes:",
      len(valid_genes))
