import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# Configuration
# ============================================================

DATA_PATH = "./data/raw/cicids2017_cleaned.csv"
MODEL_PATH = "./models/isolation_forest.joblib"

TARGET = "Attack Type"

# Number of normal and attack samples used for evaluation
NORMAL_SAMPLE_SIZE = 100_000
ATTACK_SAMPLE_SIZE = 100_000

# Thresholds to evaluate
THRESHOLDS = [
    -0.020,
    -0.015,
    -0.010,
    -0.005,
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030
]


# ============================================================
# 1. Load Model
# ============================================================

print("Loading Isolation Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# 2. Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. Check Class Distribution
# ============================================================

print("\nClass distribution:")

print(
    df[TARGET].value_counts()
)


# ============================================================
# 4. Separate Normal and Attack Traffic
# ============================================================

normal_df = df[
    df[TARGET] == "Normal Traffic"
].copy()

attack_df = df[
    df[TARGET] != "Normal Traffic"
].copy()

print(
    f"\nNormal samples available: "
    f"{len(normal_df)}"
)

print(
    f"Attack samples available: "
    f"{len(attack_df)}"
)


# ============================================================
# 5. Create Evaluation Dataset
# ============================================================

# Sample normal traffic

if len(normal_df) > NORMAL_SAMPLE_SIZE:

    normal_test = normal_df.sample(
        n=NORMAL_SAMPLE_SIZE,
        random_state=42
    )

else:

    normal_test = normal_df


# Sample attack traffic

if len(attack_df) > ATTACK_SAMPLE_SIZE:

    attack_test = attack_df.sample(
        n=ATTACK_SAMPLE_SIZE,
        random_state=42
    )

else:

    attack_test = attack_df


# Combine

test_df = pd.concat(
    [
        normal_test,
        attack_test
    ],
    ignore_index=True
)


print(
    f"\nEvaluation dataset size: "
    f"{len(test_df)}"
)


# ============================================================
# 6. Prepare Features
# ============================================================

X_test = test_df.drop(
    columns=[TARGET]
)

X_test = X_test.select_dtypes(
    include=["number"]
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.fillna(0)


print(
    f"Number of features: "
    f"{X_test.shape[1]}"
)


# ============================================================
# 7. Calculate Isolation Forest Scores
# ============================================================

print("\nCalculating anomaly scores...")

scores = model.decision_function(
    X_test
)

print("Anomaly scores calculated!")


# ============================================================
# 8. Ground Truth
# ============================================================

# Normal = 0
# Attack = 1

y_true = (
    test_df[TARGET] != "Normal Traffic"
).astype(int)


# ============================================================
# 9. Evaluate Thresholds
# ============================================================

results = []


print("\n")
print("=" * 80)
print("             ISOLATION FOREST THRESHOLD EVALUATION")
print("=" * 80)

print(
    f"{'Threshold':>12}"
    f"{'Attack Det.':>15}"
    f"{'False Pos.':>15}"
    f"{'Precision':>15}"
    f"{'Recall':>15}"
    f"{'F1':>15}"
)

print("-" * 80)


for threshold in THRESHOLDS:

    # Isolation Forest:
    # score < threshold = anomaly

    y_pred = (
        scores < threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Attack detection rate
    # --------------------------------------------------------

    attack_mask = (
        y_true == 1
    )

    attack_detected = (
        y_pred[attack_mask] == 1
    ).sum()

    total_attacks = (
        attack_mask.sum()
    )

    attack_detection_rate = (
        attack_detected / total_attacks
        if total_attacks > 0
        else 0
    )

    # --------------------------------------------------------
    # False positive rate
    # --------------------------------------------------------

    normal_mask = (
        y_true == 0
    )

    false_positives = (
        y_pred[normal_mask] == 1
    ).sum()

    total_normal = (
        normal_mask.sum()
    )

    false_positive_rate = (
        false_positives / total_normal
        if total_normal > 0
        else 0
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "threshold": threshold,
            "attack_detection": attack_detection_rate,
            "false_positive_rate": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )

    print(
        f"{threshold:12.6f}"
        f"{attack_detection_rate:14.2%}"
        f"{false_positive_rate:14.2%}"
        f"{precision:14.4f}"
        f"{recall:14.4f}"
        f"{f1:14.4f}"
    )


# ============================================================
# 10. Find Best Threshold
# ============================================================

results_df = pd.DataFrame(results)

best_result = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n")
print("=" * 80)
print("                    BEST THRESHOLD")
print("=" * 80)

print(
    f"Threshold           : "
    f"{best_result['threshold']:.6f}"
)

print(
    f"Attack Detection    : "
    f"{best_result['attack_detection']:.2%}"
)

print(
    f"False Positive Rate : "
    f"{best_result['false_positive_rate']:.2%}"
)

print(
    f"Precision            : "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall               : "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1 Score             : "
    f"{best_result['f1']:.4f}"
)


# ============================================================
# 11. Per Attack-Type Evaluation
# ============================================================

best_threshold = (
    best_result["threshold"]
)

best_predictions = (
    scores < best_threshold
).astype(int)


evaluation_df = test_df.copy()

evaluation_df["Anomaly"] = (
    best_predictions
)


print("\n")
print("=" * 80)
print("                ATTACK TYPE DETECTION")
print("=" * 80)


attack_types = sorted(
    attack_df[TARGET].unique()
)


for attack_type in attack_types:

    subset = evaluation_df[
        evaluation_df[TARGET] == attack_type
    ]

    total = len(subset)

    detected = (
        subset["Anomaly"] == 1
    ).sum()

    detection_rate = (
        detected / total
        if total > 0
        else 0
    )

    print(
        f"{attack_type:20s}"
        f"{detected:8d}/"
        f"{total:<8d}"
        f"({detection_rate:.2%})"
    )


# ============================================================
# 12. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    best_predictions
)

print("\n")
print("=" * 80)
print("                    CONFUSION MATRIX")
print("=" * 80)

print(
    "                 Predicted"
)

print(
    "              Normal  Anomaly"
)

print(
    f"Actual Normal  {cm[0][0]:7d} "
    f"{cm[0][1]:7d}"
)

print(
    f"Actual Attack  {cm[1][0]:7d} "
    f"{cm[1][1]:7d}"
)


# ============================================================
# 13. Save Evaluation Results
# ============================================================

OUTPUT_PATH = (
    "./models/isolation_forest_threshold_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nThreshold results saved to:")
print(OUTPUT_PATH)

print("\nIsolation Forest evaluation completed!")