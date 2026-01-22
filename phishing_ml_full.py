# ==================================================
# Phishing Website Detection
# ML Models + Soft Voting Fusion + ROC & AUC
# ==================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("results/roc_curves", exist_ok=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
DATA_PATH = "data/final_phishing_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Remove duplicate columns (important for your dataset)
df = df.loc[:, ~df.columns.duplicated()]

# --------------------------------------------------
# TARGET & FEATURES
# --------------------------------------------------
TARGET_COLUMN = "Result"   # 1 = Phishing, -1 = Legitimate

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].replace(-1, 0)   # Convert -1 → 0

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model
}

results = []

# --------------------------------------------------
# TRAIN & EVALUATE INDIVIDUAL MODELS
# --------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results.append([name, acc, prec, rec, f1, roc_auc])

    # ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"results/roc_curves/{name.lower().replace(' ', '_')}_roc.png")
    plt.close()

# --------------------------------------------------
# 🔥 SOFT VOTING FUSION MODEL
# --------------------------------------------------
print("\nApplying Soft Voting Fusion Model...")

# Get probabilities from both models
lr_prob = lr_model.predict_proba(X_test)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Soft Voting (Average)
fusion_prob = (lr_prob + rf_prob) / 2

# Final prediction
fusion_pred = (fusion_prob >= 0.5).astype(int)

# Metrics
fusion_acc = accuracy_score(y_test, fusion_pred)
fusion_prec = precision_score(y_test, fusion_pred)
fusion_rec = recall_score(y_test, fusion_pred)
fusion_f1 = f1_score(y_test, fusion_pred)

# ROC & AUC for Fusion
fpr_fusion, tpr_fusion, _ = roc_curve(y_test, fusion_prob)
fusion_auc = auc(fpr_fusion, tpr_fusion)

results.append([
    "Fusion Model (Soft Voting)",
    fusion_acc,
    fusion_prec,
    fusion_rec,
    fusion_f1,
    fusion_auc
])

# ROC for Fusion
plt.figure()
plt.plot(fpr_fusion, tpr_fusion, label=f"Fusion Model (AUC={fusion_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fusion Model")
plt.legend()
plt.grid()
plt.savefig("results/roc_curves/fusion_model_roc.png")
plt.close()

# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------
summary_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
)

summary_df.to_csv("results/model_performance_summary.csv", index=False)

print("\nMODEL PERFORMANCE SUMMARY")
print(summary_df)
print("\nFusion model completed successfully.")
