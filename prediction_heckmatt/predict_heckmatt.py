#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heckmatt Score Prediction - Inference Only
==========================================
Loads 10 pre-trained XGBoost fold models and predicts Heckmatt scores
from ultrasound image segmentation features.

NO training is performed. Predictions from all 10 models are averaged
(soft voting) to produce a final prediction per sample.

Required inputs
---------------
1. segmentation_summary_knet_swin_mod_pred.json
      Radiomics / segmentation features extracted per ultrasound image.

2. heckMapPlusCharacteristics.xlsx
      Patient demographics (age, sex, BMI).
      Heckmatt labels column can be filled with dummy values (e.g. 1)
      if you don't have ground-truth yet.

3. XGBoost_fitted_models/
      Folder containing fold_0.pkl … fold_9.pkl  (10 trained pipelines).

Outputs
-------
predictions.csv   – one row per sample, columns:
                    subject | muscle | side | predicted_h_score (1-3)
                    + per-class probabilities (prob_class1, prob_class2, prob_class3)
"""

import os
import json
import pickle
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# ─────────────────────────────────────────────
# CONFIGURE PATHS HERE
# ─────────────────────────────────────────────
DATA_DIR        = '/path/to/your/data'          # <-- change this
MODELS_DIR      = '/path/to/XGBoost_fitted_models'  # <-- change this
OUTPUT_DIR      = '/path/to/output'             # <-- change this

JSON_FILE       = os.path.join(DATA_DIR, 'segmentation_summary_knet_swin_mod_pred.json')
EXCEL_FILE      = os.path.join(DATA_DIR, 'heckMapPlusCharacteristics.xlsx')
OUTPUT_CSV      = os.path.join(OUTPUT_DIR, 'predictions.csv')

# ─────────────────────────────────────────────
# LOOKUP TABLES  (identical to original script)
# ─────────────────────────────────────────────
class_to_code = {
    'Biceps_brachii': '001', 'Deltoideus': '002', 'Depressor_anguli_oris': '003',
    'Digastricus': '004', 'Extensor_digitorum_brevis': '005', 'Flexor_carpi_radialis': '006',
    'Flexor_digitorum_profundus': '007', 'Gastrocnemius_medial_head': '008',
    'Geniohyoideus': '009', 'Levator_labii_superior': '010', 'Masseter': '011',
    'Mentalis': '012', 'Orbicularis_oris': '013', 'Peroneus_tertius': '014',
    'Rectus_abdominis': '015', 'Rectus_femoris': '016', 'Temporalis': '017',
    'Tibialis_anterior': '018', 'Trapezius': '019', 'Vastus_lateralis': '020',
    'Zygomaticus': '021'
}

code_to_class_original = {
    '001': 'BB', '002': 'DEL', '003': 'DA', '004': 'DIG', '005': 'EDB',
    '006': 'FCR', '007': 'FDP', '008': 'GM', '009': 'GH', '010': 'LLS',
    '011': 'MAS', '012': 'MNT', '013': 'OO', '014': 'PT', '015': 'RA',
    '016': 'RF', '017': 'TEM', '018': 'TA', '019': 'TRAP', '020': 'VL',
    '021': 'ZYG'
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS  (identical to original)
# ─────────────────────────────────────────────

def mean_features_with_less_variation(group):
    feature_columns = group["features_img_gt"].iloc[0].keys()
    mean_features = {}
    for feature in feature_columns:
        values = [float(entry[feature]) for entry in group["features_img_gt"]]
        mean_features[feature] = sum(values) / len(values)
    return pd.Series(mean_features)

def mean_features_with_less_variation_nan(group):
    feature_columns = group["features_img_gt"].iloc[0].keys()
    mean_features = {}
    for feature in feature_columns:
        values = [float(entry[feature]) for entry in group["features_img_gt"]]
        variation = max(values) - min(values)
        if variation / abs(np.mean(values) + 1e-10) < 0.5:
            mean_features[feature] = sum(values) / len(values)
        else:
            mean_features[feature] = np.nan
    return pd.Series(mean_features)

def mean_features_with_less_variation_not(group):
    feature_columns = group["features_img_gt_not"].iloc[0].keys()
    mean_features = {}
    for feature in feature_columns:
        values = [float(entry[feature]) for entry in group["features_img_gt_not"]]
        mean_features[feature] = sum(values) / len(values)
    return pd.Series(mean_features)

def mean_features_with_less_variation_nan_not(group):
    feature_columns = group["features_img_gt_not"].iloc[0].keys()
    mean_features = {}
    for feature in feature_columns:
        values = [float(entry[feature]) for entry in group["features_img_gt_not"]]
        variation = max(values) - min(values)
        if variation / abs(np.mean(values) + 1e-10) < 0.5:
            mean_features[feature] = sum(values) / len(values)
        else:
            mean_features[feature] = np.nan
    return pd.Series(mean_features)


# ─────────────────────────────────────────────
# STEP 1 — LOAD & MERGE DATA
# ─────────────────────────────────────────────
print("Loading JSON feature file …")
with open(JSON_FILE, 'r') as f:
    data = json.load(f)
df = pd.DataFrame.from_dict(data)
del data
gc.collect()

print("Loading demographics from Excel …")
HeckMap = pd.read_excel(EXCEL_FILE)
HeckMap = HeckMap.iloc[5:, :]

HeckMap['Code']     = HeckMap['Code'].apply(
    lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) and x != '' else '')
HeckMap['Code']     = HeckMap['Code'].astype(str)
HeckMap['Sex']      = HeckMap['Sex'].astype(str)
HeckMap['FSHD_age'] = HeckMap['FSHD_age'].astype(str)
HeckMap['FSHD_BMI'] = HeckMap['FSHD_BMI'].astype(str)

df['muscle_code'] = df['class_gt'].map(class_to_code)

df1 = pd.merge(df, HeckMap[['Code', 'Sex', 'FSHD_age', 'FSHD_BMI']],
               left_on='subject', right_on='Code', how='left')
df1 = df1.drop('Code', axis=1)
df1.rename(columns={'Sex': 'sex', 'FSHD_age': 'age', 'FSHD_BMI': 'bmi'}, inplace=True)

df1['muscle_code'] = df1['muscle_code'].astype(str)
df1['side']        = df1['side'].astype(str)
df1['muscle_side'] = df1['muscle_code'] + '_' + df1['side']

# ── Assign Heckmatt labels (use dummy = 1 if you have no ground truth) ──
print("Assigning Heckmatt labels (dummy = 1 where not found) …")
for idx, row in df1.iterrows():
    column_name  = row['muscle_side']
    subject_name = row['subject']
    if column_name in HeckMap.columns and subject_name in HeckMap['Code'].values:
        find_idx = HeckMap['Code'].loc[lambda x: x == subject_name].index[0]
        df1.loc[idx, 'manual_h_score'] = HeckMap.loc[find_idx, column_name]
    else:
        # DUMMY LABEL — keeps the pipeline running without ground truth
        df1.loc[idx, 'manual_h_score'] = 1

# ─────────────────────────────────────────────
# STEP 2 — PRE-PROCESS FEATURES
# ─────────────────────────────────────────────
print("Pre-processing features …")
df_hPred = df1.copy()
df_hPred.replace('mask not found', np.nan, inplace=True)
df_hPred = df_hPred.dropna(axis=0)

df_hPred['subject'] = pd.to_numeric(df_hPred['subject'], errors='coerce')
df_hPred['age']     = pd.to_numeric(df_hPred['age'],     errors='coerce')
df_hPred['bmi']     = pd.to_numeric(df_hPred['bmi'],     errors='coerce')
df_hPred['muscleN'] = pd.to_numeric(df_hPred['muscle'],  errors='coerce')

df_gt = df_hPred[['subject', 'muscle', 'side', 'age', 'bmi', 'sex',
                   'muscleN', 'features_img_gt', 'features_img_gt_not', 'manual_h_score']]

grouped_df     = df_gt.groupby(['subject', 'muscle', 'side']).apply(mean_features_with_less_variation).reset_index()
grouped_df_nan = df_gt.groupby(['subject', 'muscle', 'side']).apply(mean_features_with_less_variation_nan).reset_index()

grouped_df_not     = df_gt.groupby(['subject', 'muscle', 'side']).apply(mean_features_with_less_variation_not).reset_index()
grouped_df_nan_not = df_gt.groupby(['subject', 'muscle', 'side']).apply(mean_features_with_less_variation_nan_not).reset_index()

threshold       = grouped_df.shape[0] * 0.1
filtered_df     = grouped_df.loc[:, grouped_df_nan.isna().sum() <= threshold]
filtered_df_not = grouped_df_not.loc[:, grouped_df_nan_not.isna().sum() <= threshold]

filtered_df     = filtered_df.drop(['subject', 'muscle', 'side'], axis=1)
filtered_df_not = filtered_df_not.drop(['subject', 'muscle', 'side'], axis=1)

data_dict     = filtered_df.to_dict(orient='records')
data_dict_not = filtered_df_not.to_dict(orient='records')

df_hPred_group = df_hPred.groupby(['subject', 'muscle', 'side']).agg('first').reset_index()
df_hPred_group["manual_h_score"] = df_hPred_group["manual_h_score"].replace(4, 3)

feat_names_gt  = list(data_dict[0].keys())
feat_names_not = list(data_dict_not[0].keys())

scaled_df     = pd.DataFrame(data=data_dict,     columns=feat_names_gt)
scaled_df_not = pd.DataFrame(data=data_dict_not, columns=feat_names_not)

filtered_df     = scaled_df
filtered_df_not = scaled_df_not.add_suffix('_not')

dfX = df_hPred_group[['subject', 'muscle', 'side', 'manual_h_score']].copy()
dfX = dfX.set_index(['subject', 'muscle', 'side'])
dfX["manual_h_score"] = dfX["manual_h_score"].astype("category")

filtered_df.index     = dfX.index
filtered_df_not.index = dfX.index

dfX = pd.concat([dfX, filtered_df, filtered_df_not], axis=1)

X = dfX.drop(columns=['manual_h_score'])
Y = dfX[['manual_h_score']].astype('uint8') - 1   # 0-indexed for model; still dummy if no GT

print(f"Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────
# STEP 3 — LOAD MODELS & PREDICT
# ─────────────────────────────────────────────
print("\nLoading pre-trained fold models and predicting …")

n_classes  = 3
prob_accum = np.zeros((X.shape[0], n_classes))   # accumulate probabilities across folds

for fold_idx in range(10):
    model_path = os.path.join(MODELS_DIR, f'fold_{fold_idx}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Expected files named fold_0.pkl … fold_9.pkl in {MODELS_DIR}\n"
            f"Rename your pkl files to match this pattern if they differ."
        )

    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    probs       = pipeline.predict_proba(X)   # shape (n_samples, 3)
    prob_accum += probs
    print(f"  Fold {fold_idx} done.")

# Average probabilities (soft voting across 10 folds)
prob_avg       = prob_accum / 10
predicted_idx  = np.argmax(prob_avg, axis=1)       # 0-indexed class
predicted_score = predicted_idx + 1                 # convert to Heckmatt scale 1-3

# ─────────────────────────────────────────────
# STEP 4 — BUILD OUTPUT DATAFRAME & SAVE
# ─────────────────────────────────────────────
print("\nSaving predictions …")

results = X.copy()
results = results[[]]           # keep only the index (subject / muscle / side)
results = results.reset_index()

results['predicted_h_score'] = predicted_score
results['prob_class1']        = prob_avg[:, 0]
results['prob_class2']        = prob_avg[:, 1]
results['prob_class3']        = prob_avg[:, 2]

# Human-readable muscle names
results['muscle_name'] = results['muscle'].map(code_to_class_original)

os.makedirs(OUTPUT_DIR, exist_ok=True)
results.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone! Predictions written to:\n  {OUTPUT_CSV}")
print(f"\nPrediction distribution:")
print(results['predicted_h_score'].value_counts().sort_index())
print(f"\nFirst 10 rows:")
print(results.head(10).to_string(index=False))
