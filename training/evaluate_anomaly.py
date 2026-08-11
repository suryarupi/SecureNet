import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "./data/processed/cicids2017_binary.csv"

MODEL_PATH = "./models/isolation_forest.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Isolation Forest...")

artifact = joblib.load(
    MODEL_PATH
)

# Your train_anomaly.py saved a dictionary
isolation_model = artifact["model"]

# These are the exact features used during training
model_features = artifact["features"]

print("Isolation Forest loaded successfully.")

print(
    f"Number of model features: "
    f"{len(model_features)}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nPreparing features...")

X = df.drop(
    columns=[
        "Attack Type",
        "Binary_Label"
    ]
).copy()

y_attack_type = df["Attack Type"]


# Make sure the feature order is exactly
# the same as during Isolation Forest training.

X = X.reindex(
    columns=model_features
)


# ============================================================
# CLEAN FEATURES
# ============================================================

# Replace infinity values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Convert to numeric

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Fill missing values

missing_values = X.isna().sum().sum()

print(
    f"Missing values: {missing_values}"
)

if missing_values > 0:

    print(
        "Filling missing values with "
        "column medians..."
    )

    X = X.fillna(
        X.median()
    )

# Remaining NaN values, if any

X = X.fillna(0)


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET ATTACK DISTRIBUTION")
print("=" * 70)

print(
    y_attack_type.value_counts()
)


# ============================================================
# RUN ISOLATION FOREST
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING ISOLATION FOREST")
print("=" * 70)

print(
    "\nEvaluating all traffic samples..."
)

# Isolation Forest:
#
#  1  = Normal
# -1  = Anomaly

predictions = (
    isolation_model.predict(X)
)

# Decision function:
# Higher = more normal
# Lower  = more anomalous

anomaly_scores = (
    isolation_model.decision_function(X)
)


# ============================================================
# CONVERT PREDICTION
# ============================================================

# Convert Isolation Forest output:
#
#  1  -> Normal
# -1  -> Attack / Anomaly

predicted_anomaly = (
    predictions == -1
)


# ============================================================
# OVERALL COUNTS
# ============================================================

total_samples = len(
    predictions
)

total_anomalies = (
    predicted_anomaly.sum()
)

total_normal = (
    total_samples -
    total_anomalies
)


print("\n")
print("=" * 70)
print("OVERALL ANOMALY RESULTS")
print("=" * 70)

print(
    f"\nTotal samples       : "
    f"{total_samples}"
)

print(
    f"Predicted normal    : "
    f"{total_normal}"
)

print(
    f"Predicted anomaly   : "
    f"{total_anomalies}"
)

print(
    f"Anomaly percentage  : "
    f"{(total_anomalies / total_samples) * 100:.2f}%"
)


# ============================================================
# NORMAL TRAFFIC EVALUATION
# ============================================================

normal_mask = (
    y_attack_type ==
    "Normal Traffic"
)

normal_predictions = (
    predicted_anomaly[
        normal_mask
    ]
)

normal_anomaly_count = (
    normal_predictions.sum()
)

normal_total = (
    normal_mask.sum()
)

normal_anomaly_rate = (
    normal_anomaly_count /
    normal_total
)


print("\n")
print("=" * 70)
print("NORMAL TRAFFIC EVALUATION")
print("=" * 70)

print(
    f"\nTotal normal samples : "
    f"{normal_total}"
)

print(
    f"Normal → Normal      : "
    f"{normal_total - normal_anomaly_count}"
)

print(
    f"Normal → Anomaly     : "
    f"{normal_anomaly_count}"
)

print(
    f"False Positive Rate  : "
    f"{normal_anomaly_rate:.4f}"
)

print(
    f"False Positive %     : "
    f"{normal_anomaly_rate * 100:.2f}%"
)


# ============================================================
# ATTACK TRAFFIC EVALUATION
# ============================================================

attack_mask = (
    y_attack_type !=
    "Normal Traffic"
)

attack_predictions = (
    predicted_anomaly[
        attack_mask
    ]
)

attack_anomaly_count = (
    attack_predictions.sum()
)

attack_total = (
    attack_mask.sum()
)

attack_detection_rate = (
    attack_anomaly_count /
    attack_total
)


print("\n")
print("=" * 70)
print("ATTACK TRAFFIC EVALUATION")
print("=" * 70)

print(
    f"\nTotal attack samples : "
    f"{attack_total}"
)

print(
    f"Attack → Anomaly     : "
    f"{attack_anomaly_count}"
)

print(
    f"Attack → Normal      : "
    f"{attack_total - attack_anomaly_count}"
)

print(
    f"Attack Detection Rate: "
    f"{attack_detection_rate:.4f}"
)

print(
    f"Attack Detection %   : "
    f"{attack_detection_rate * 100:.2f}%"
)


# ============================================================
# PER-CLASS ANOMALY DETECTION
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS ANOMALY DETECTION")
print("=" * 70)

class_results = []


for attack_type in sorted(
    y_attack_type.unique()
):

    class_mask = (
        y_attack_type ==
        attack_type
    )

    class_predictions = (
        predicted_anomaly[
            class_mask
        ]
    )

    class_total = (
        class_mask.sum()
    )

    class_anomalies = (
        class_predictions.sum()
    )

    class_detection_rate = (
        class_anomalies /
        class_total
    )

    class_results.append({

        "Attack Type":
            attack_type,

        "Total Samples":
            class_total,

        "Anomalies Detected":
            class_anomalies,

        "Anomaly Rate":
            class_detection_rate
    })


class_results_df = pd.DataFrame(
    class_results
)


# Display percentage nicely

display_df = (
    class_results_df.copy()
)

display_df["Anomaly Rate"] = (
    display_df["Anomaly Rate"] * 100
).round(2)


print()

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BINARY GROUND TRUTH
# ============================================================

# For evaluation we define:
#
# 0 = Normal Traffic
# 1 = Attack
#
# This lets us calculate standard
# binary classification metrics.

y_true_binary = (
    y_attack_type !=
    "Normal Traffic"
).astype(int)


# Isolation Forest:
#
# Normal   -> 0
# Anomaly  -> 1

y_pred_binary = (
    predicted_anomaly
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true_binary,
    y_pred_binary,
    labels=[0, 1]
)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nClass order:"
)

print(
    "[0 = Normal, 1 = Attack]"
)

print("\n")

print(cm)


# Extract values

tn, fp, fn, tp = (
    cm.ravel()
)


print("\n")
print("True Negatives  :", tn)
print("False Positives :", fp)
print("False Negatives :", fn)
print("True Positives  :", tp)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true_binary,
    y_pred_binary
)

precision = precision_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

recall = recall_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

f1 = f1_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)


print("\n")
print("=" * 70)
print("ISOLATION FOREST METRICS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        y_true_binary,
        y_pred_binary,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    "./models/"
    "isolation_forest_evaluation.csv"
)

class_results_df.to_csv(
    results_path,
    index=False
)


print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nPer-class results saved at:"
)

print(
    results_path
)

print("\n")