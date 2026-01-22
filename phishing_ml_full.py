# ==================================================
# Phishing Website Detection - ML + ROC & AUC
# Dataset-based (Result column: -1, 1)
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

print("Dataset loaded successfully")
print("Shape:", df.shape)

# --------------------------------------------------
# HANDLE DUPLICATE COLUMNS (IMPORTANT FOR YOUR DATA)
# --------------------------------------------------
df = df.loc[:, ~df.columns.duplicated()]

# --------------------------------------------------
# TARGET & FEATURES
# --------------------------------------------------
TARGET_COLUMN = "Result"   # 1 = Phishing, -1 = Legitimate

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Convert labels: -1 → 0, 1 → 1 (REQUIRED FOR ROC)
y = y.replace(-1, 0)

# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

# --------------------------------------------------
# TRAIN, EVALUATE, ROC & AUC
# --------------------------------------------------
summary = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    summary.append([
        model_name,
        acc,
        prec,
        rec,
        f1,
        roc_auc
    ])

    # --------------------------------------------------
    # ROC CURVE PLOT
    # --------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.2f})",
        linewidth=2
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)

    roc_file = f"results/roc_curves/{model_name.lower().replace(' ', '_')}_roc.png"
    plt.savefig(roc_file)
    plt.close()

    print(f"ROC curve saved → {roc_file}")

# --------------------------------------------------
# SAVE PERFORMANCE SUMMARY
# --------------------------------------------------
summary_df = pd.DataFrame(
    summary,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "AUC"
    ]
)

summary_path = "results/model_performance_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nMODEL PERFORMANCE SUMMARY")
print(summary_df)
print(f"\nSummary saved → {summary_path}")
print("ROC & AUC generation completed successfully.")
