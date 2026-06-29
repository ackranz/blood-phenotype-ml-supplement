import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_path = "merged_variant_data_78_a.csv"
out_dir = "collinearity_check"
os.makedirs(out_dir, exist_ok=True)

FEATURE_SETS = {
    "CTRLV_VS_NULLV": [
        "Delta_Hydrophobicity",
        "Delta_Sidechain Volume",
        "SurfaceAccessibility",
        "MedusaFlexWeighted",
        "Conservation",
        "MissenseScore",
    ],
    "CTRLV_VS_AG": [
        "Hydrophobicity",
        "Sidechain Volume",
        "SurfaceAccessibility",
        "Conservation",
        "MedusaFlexWeighted",
    ],
}

def compute_tables(df, features, prefix):
    sub = df[features].copy()
    for c in features:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna().reset_index(drop=True)

    corr = sub.corr(method="pearson")
    corr.to_csv(os.path.join(out_dir, f"{prefix}_corr.csv"))

    rows = []
    for i, a in enumerate(features):
        for j, b in enumerate(features):
            if j <= i:
                continue
            rows.append({
                "Feature_1": a,
                "Feature_2": b,
                "Pearson_r": corr.loc[a, b],
                "abs_r": abs(corr.loc[a, b]),
            })
    corr_long = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    corr_long.to_csv(os.path.join(out_dir, f"{prefix}_corr_long.csv"), index=False)

    X = sub.values.astype(float)
    vif = pd.DataFrame({
        "Feature": features,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    vif.to_csv(os.path.join(out_dir, f"{prefix}_vif.csv"), index=False)

    print(f"\n=== {prefix} ===")
    print("n complete:", len(sub))
    print("\nVIF:")
    print(vif)
    print("\nTop correlations:")
    print(corr_long.head(10))

df_all = pd.read_csv(file_path)

# CTRLV vs NullV
df1 = df_all[df_all["Phenotype"].isin(["CtrlV", "NullV"])].copy()
compute_tables(df1, FEATURE_SETS["CTRLV_VS_NULLV"], "ctrlv_vs_nullv_final")
if "Delta_Polarity" in df1.columns:
    compute_tables(df1, FEATURE_SETS["CTRLV_VS_NULLV"] + ["Delta_Polarity"], "ctrlv_vs_nullv_plus_delta_polarity")

# CTRLV vs Ag
df2 = df_all[df_all["Phenotype"].isin(["CtrlV", "AgV", "AgWT"])].copy()
compute_tables(df2, FEATURE_SETS["CTRLV_VS_AG"], "ctrlv_vs_ag_final")
if "Polarity" in df2.columns:
    compute_tables(df2, FEATURE_SETS["CTRLV_VS_AG"] + ["Polarity"], "ctrlv_vs_ag_plus_polarity")