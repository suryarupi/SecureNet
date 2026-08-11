import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "./data/raw/cicids2017_cleaned.csv"
MODEL_PATH = "./models/isolation_forest.joblib"

THRESHOLD_PATH = "./models/isolation_forest_threshold.json"

TARGET = "Attack Type"

NORMAL_LABEL = "Normal Traffic"

# Same test sizes used in our Isolation Forest training script
NORMAL_TEST_SIZE = 100_000
ATTACK_TEST_SIZE = 100_000

# Percentiles to test
NORMAL_PERCENTILES = [
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
    15.0
]

# Target false positive rate
TARGET_FPR = 0.05


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Loading Isolation Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. SEPARATE NORMAL AND ATTACK TRAFFIC
# ============================================================

normal_df = df[
    df[TARGET] == NORMAL_LABEL
].copy()

attack_df = df[
    df[TARGET] != NORMAL_LABEL
].copy()

print(f"\nNormal samples available: {len(normal_df)}")
print(f"Attack samples available: {len(attack_df)}")


# ============================================================
# 4. CREATE TEST DATA
# ============================================================

# Use the same random_state as our previous script
normal_test = normal_df.sample(
    n=min(NORMAL_TEST_SIZE, len(normal_df)),
    random_state=42
)

attack_test = attack_df.sample(
    n=min(ATTACK_TEST_SIZE, len(attack_df)),
    random_state=42
)


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

X_normal = normal_test.drop(
    columns=[TARGET]
)

X_attack = attack_test.drop(
    columns=[TARGET]
)


# Keep numeric features only
X_normal = X_normal.select_dtypes(
    include=["number"]
)

X_attack = X_attack.select_dtypes(
    include=["number"]
)


# Handle infinity and missing values
X_normal = X_normal.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

X_attack = X_attack.replace(
    [float("inf"), float("-inf")],
    float("nan")
)


X_normal = X_normal.fillna(0)
X_attack = X_attack.fillna(0)


# ============================================================
# 6. GET ISOLATION FOREST SCORES
# ============================================================

print("\nCalculating anomaly scores...")

normal_scores = model.decision_function(
    X_normal
)

attack_scores = model.decision_function(
    X_attack
)

print("Scores calculated!")


# ============================================================
# IMPORTANT:
# Higher Isolation Forest score = more normal
# Lower score = more anomalous
#
# Therefore:
#
# score < threshold → Anomaly
# score >= threshold → Normal
# ============================================================


# ============================================================
# 7. TEST DIFFERENT THRESHOLDS
# ============================================================

print("\n")
print("=" * 115)
print("TESTING ANOMALY THRESHOLDS")
print("=" * 115)

print()

header = (
    f"{'Normal Percentile':<20}"
    f"{'Threshold':<15}"
    f"{'False Positive Rate':<22}"
    f"{'Attack Detection Rate':<24}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
    f"{'F1':<10}"
)

print(header)


results = []


for percentile in NORMAL_PERCENTILES:

    # Threshold based ONLY on normal traffic
    threshold = np.percentile(
        normal_scores,
        percentile
    )


    # --------------------------------------------------------
    # Normal traffic
    # --------------------------------------------------------

    # Normal becomes false positive if classified as anomaly
    normal_predictions = (
        normal_scores < threshold
    )

    false_positives = normal_predictions.sum()

    false_positive_rate = (
        false_positives / len(normal_scores)
    )


    # --------------------------------------------------------
    # Attack traffic
    # --------------------------------------------------------

    # Attack detected if score is below threshold
    attack_predictions = (
        attack_scores < threshold
    )

    attack_detected = attack_predictions.sum()

    attack_detection_rate = (
        attack_detected / len(attack_scores)
    )


    # --------------------------------------------------------
    # Combined metrics
    # --------------------------------------------------------

    # Ground truth:
    # 0 = Normal
    # 1 = Attack

    y_true = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(attack_scores))
    ])


    y_pred = np.concatenate([
        normal_predictions.astype(int),
        attack_predictions.astype(int)
    ])


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    results.append({
        "percentile": percentile,
        "threshold": threshold,
        "fpr": false_positive_rate,
        "attack_detection_rate": attack_detection_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


    print(
        f"{percentile:<20.1f}"
        f"{threshold:<15.6f}"
        f"{false_positive_rate * 100:<22.2f}"
        f"{attack_detection_rate * 100:<24.2f}"
        f"{precision * 100:<12.2f}"
        f"{recall * 100:<10.2f}"
        f"{f1 * 100:<10.2f}"
    )


# ============================================================
# 8. SELECT BEST THRESHOLD
# ============================================================

print("\n")
print("=" * 115)
print("SELECTING BEST THRESHOLD")
print("=" * 115)

print(
    f"\nTarget False Positive Rate: "
    f"{TARGET_FPR * 100:.2f}%"
)


# ------------------------------------------------------------
# Select the highest tested percentile that remains
# strictly below the target FPR.
#
# This reproduces the behavior seen in your friend's output:
# target = 5%
# selected = 4%
# ------------------------------------------------------------

valid_results = [
    result
    for result in results
    if result["fpr"] < TARGET_FPR
]


if not valid_results:

    print(
        "\nNo threshold satisfies the target FPR."
    )

    # fallback: choose lowest FPR
    best_result = min(
        results,
        key=lambda x: x["fpr"]
    )

else:

    # Choose the largest FPR still below target
    best_result = max(
        valid_results,
        key=lambda x: x["fpr"]
    )


# ============================================================
# 9. DISPLAY SELECTED THRESHOLD
# ============================================================

print(
    f"\nSelected threshold: "
    f"{best_result['threshold']:.6f}"
)

print(
    f"False Positive Rate: "
    f"{best_result['fpr'] * 100:.2f}%"
)

print(
    f"Attack Detection Rate: "
    f"{best_result['attack_detection_rate'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best_result['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best_result['recall'] * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{best_result['f1'] * 100:.2f}%"
)


# ============================================================
# 10. SAVE THRESHOLD
# ============================================================

threshold_data = {
    "threshold": float(best_result["threshold"]),
    "target_false_positive_rate": TARGET_FPR,
    "actual_false_positive_rate": float(
        best_result["fpr"]
    ),
    "attack_detection_rate": float(
        best_result["attack_detection_rate"]
    ),
    "precision": float(
        best_result["precision"]
    ),
    "recall": float(
        best_result["recall"]
    ),
    "f1_score": float(
        best_result["f1"]
    ),
    "normal_percentile": float(
        best_result["percentile"]
    )
}


os.makedirs(
    "./models",
    exist_ok=True
)


with open(
    THRESHOLD_PATH,
    "w"
) as file:

    json.dump(
        threshold_data,
        file,
        indent=4
    )


print(
    f"\nThreshold saved to: "
    f"{THRESHOLD_PATH}"
)

print(
    "\nIsolation Forest threshold tuning completed!"
)