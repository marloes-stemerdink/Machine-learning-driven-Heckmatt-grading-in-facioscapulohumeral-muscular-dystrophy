#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference-only script for Heckmatt grading
- Uses pretrained XGBoost model
- Aggregates 3 images per muscle
- Outputs predicted and manual Heckmatt scores
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# =========================================
# CONFIGURATION
# =========================================
DATA_DIR = "/home/marloes.stemerdink@mydre.org/Documents/analysis"

FEATURE_JSON = os.path.join(
    DATA_DIR,
    "results/feature_extraction_output",
    "segmentation_summary_knet_swin_mod_muscle_specific.json"
)

HECKMAP_XLSX = os.path.join(
    DATA_DIR,
    "data",
    "dummy_heckMapPlusCharacteristics.xlsx"
)

MODEL_PATH = os.path.join(
    DATA_DIR,
    "Machine-learning-driven-Heckmatt-grading-in-facioscapulohumeral-muscular-dystrophy",
    "XGboost_fitted_models",
    "best_model_fold2.pkl"   # adjust filename if needed
)

OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "results/Heckmatt/inference_Heckmatt_results2.csv"
)

# =========================================
# Muscle code mapping (same as training)
# =========================================
class_to_code = {
    'Biceps_brachii': '001',
    'Deltoideus': '002',
    'Depressor_anguli_oris': '003',
    'Digastricus': '004',
    'Extensor_digitorum_brevis': '005',
    'Flexor_carpi_radialis': '006',
    'Flexor_digitorum_profundus': '007',
    'Gastrocnemius_medial_head': '008',
    'Geniohyoideus': '009',
    'Levator_labii_superior': '010',
    'Masseter': '011',
    'Mentalis': '012',
    'Orbicularis_oris': '013',
    'Peroneus_tertius': '014',
    'Rectus_abdominis': '015',
    'Rectus_femoris': '016',
    'Temporalis': '017',
    'Tibialis_anterior': '018',
    'Trapezius': '019',
    'Vastus_lateralis': '020',
    'Zygomaticus': '021'
}

# =========================================
# 1. Load feature JSON
# =========================================
with open(FEATURE_JSON, "r") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data)

df["muscle_code"] = df["Muscle"].map(class_to_code)
df = df.dropna(subset=["muscle_code"])

# =========================================
# 2. Aggregate 3 images per muscle
# =========================================
rows = []

group_cols = ["subject", "muscle_code", "side"]

for (subject, muscle, side), g in df.groupby(group_cols):

    # Collect feature dicts
    main_dicts = [
        x for x in g["features_img_pred"]
        if isinstance(x,dict)
    ]   # prevent error when mask is not found

    low_dicts = [
        x for x in g["features_img_pred_not"]
        if isinstance(x,dict)
    ]   # prevent error when mask is not found

    # skip this muscle if no valid features exist
    if len(main_dicts) == 0 or len(low_dicts) == 0:
        print(f"Skipping {subject}, muscle {muscle}, side {side} (no valid features)")
        continue

    feat_main = pd.DataFrame(main_dicts)
    feat_low = pd.DataFrame(low_dicts)

    # Force numeric conversion
    feat_main = feat_main.apply(pd.to_numeric, errors="coerce")
    feat_low = feat_low.apply(pd.to_numeric, errors="coerce")


    # Mean across images
    feat_main_mean = feat_main.mean(skipna=True)
    feat_low_mean = feat_low.mean(skipna=True)

    # Combine features (same naming as training)
    combined = feat_main_mean.to_dict()
    combined.update({f"{k}_not": v for k, v in feat_low_mean.items()})

    combined["subject"] = subject
    combined["muscle"] = muscle
    combined["side"] = side

    rows.append(combined)

X = pd.DataFrame(rows)
X = X.set_index(["subject", "muscle", "side"])
X = X.astype(float)

print(f"Inference samples: {X.shape[0]}")

# =========================================
# 5. Load manual Heckmatt (comparison only)
# =========================================
heckmap = pd.read_excel(HECKMAP_XLSX)
heckmap = heckmap.iloc[5:, :]

heckmap["Code"] = (
    heckmap["Code"]
    .astype(str)
    .astype(float)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# Pretrained model adds certain features, make a match
meta = heckmap[["Code", "Age", "Sex", "BMI"]].copy()

meta["Code"] = (
    meta["Code"].astype(str).astype(float).astype(int).astype(str).str.zfill(5)
)
meta = meta.rename(columns={
    "Code": "subject",
    "Age": "age",
    "Sex": "sex",
    "BMI": "bmi"
})

meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
meta["bmi"] = pd.to_numeric(meta["bmi"], errors="coerce")
meta["sex"] = pd.to_numeric(meta["sex"], errors="coerce")

# =========================================
# 3. Load pretrained model
# =========================================
clf = joblib.load(MODEL_PATH)

X = X.reset_index()
X = X.merge(meta, on="subject", how="left")
X = X.set_index(["subject", "muscle", "side"])

expected_features=clf.get_booster().feature_names
# add required clinical covariates if missing
for col in ["age", "sex", "bmi", "muscleN"]:
    if col not in X.columns:
        X[col] = np.nan

X = X.reindex(columns=expected_features)
X["muscleN"]=(
    X.index.get_level_values("muscle").astype(str).astype(int)
)
print(X["muscleN"].describe())

# =========================================
# 4. Predict Heckmatt class (0–2 → 1–3)
# =========================================
y_pred = clf.predict(X)
y_pred = y_pred.astype(int) + 1

df_out = pd.DataFrame(
    {"predicted_heckmatt": y_pred},
    index=X.index
)

manual = []


for (subject, muscle, side) in df_out.index:
    col = f"{muscle}_{side}"
    row = heckmap.loc[heckmap["Code"] == subject]

    if not row.empty and col in row.columns:
        manual.append(row[col].values[0])
    else:
        manual.append(np.nan)

df_out["manual_heckmatt"] = manual

# =========================================
# 6. Save results
# =========================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_out.to_csv(OUTPUT_CSV)

print("Inference completed.")
print(f"Results saved to:\n{OUTPUT_CSV}")