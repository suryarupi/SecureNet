import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "./data/processed/cicids2017_binary.csv"

MODEL_OUTPUT_PATH = (
    "./models/isolation_forest_improved.pkl"
)

THRESHOLD_OUTPUT_PATH = (
    "./models/anomaly_threshold.pkl"
)

RANDOM_STATE = 42

# Number of trees
N_ESTIMATORS = 200

# We want to keep the false-positive rate around
# or below this value.
TARGET_FPR = 0.05


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading CICIDS2017 dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 2. SEPARATE NORMAL AND ATTACK TRAFFIC
# ============================================================

print("\nSeparating normal and attack traffic...")

normal_df = df[
    df["Attack Type"] == "Normal Traffic"
].copy()

attack_df = df[
    df["Attack Type"] != "Normal Traffic"
].copy()


print(
    f"Normal samples : {len(normal_df)}"
)

print(
    f"Attack samples : {len(attack_df)}"
)


# ============================================================
# 3. SPLIT NORMAL TRAFFIC
# ============================================================

print("\nSplitting normal traffic...")

normal_train, normal_validation = train_test_split(
    normal_df,
    test_size=0.20,
    random_state=RANDOM_STATE
)


print(
    f"Normal training samples   : "
    f"{len(normal_train)}"
)

print(
    f"Normal validation samples : "
    f"{len(normal_validation)}"
)


# ============================================================
# 4. CREATE FEATURES
# ============================================================

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in [
        "Attack Type",
        "Binary_Label"
    ]
]


print(
    f"\nNumber of features: "
    f"{len(FEATURE_COLUMNS)}"
)


X_normal_train = normal_train[
    FEATURE_COLUMNS
].copy()


X_normal_validation = normal_validation[
    FEATURE_COLUMNS
].copy()


X_attack = attack_df[
    FEATURE_COLUMNS
].copy()


# ============================================================
# 5. CLEAN FEATURES
# ============================================================

print("\nCleaning features...")

def clean_features(X, medians=None):

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if medians is None:
        medians = X.median()

    X = X.fillna(medians)

    X = X.fillna(0)

    return X, medians


# IMPORTANT:
# Calculate medians ONLY from training data.

X_normal_train, training_medians = clean_features(
    X_normal_train
)

X_normal_validation, _ = clean_features(
    X_normal_validation,
    training_medians
)

X_attack, _ = clean_features(
    X_attack,
    training_medians
)


print("Feature cleaning completed.")


# ============================================================
# 6. TRAIN ISOLATION FOREST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING IMPROVED ISOLATION FOREST")
print("=" * 70)

print(
    "\nTraining ONLY on the normal-training data..."
)

isolation_forest = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


isolation_forest.fit(
    X_normal_train
)


print(
    "\nIsolation Forest training completed!"
)


# ============================================================
# 7. GENERATE ANOMALY SCORES
# ============================================================

print("\nGenerating anomaly scores...")

# IMPORTANT:
#
# decision_function:
# Higher score = more normal
# Lower score  = more anomalous
#
# Therefore:
#
# score < threshold → anomaly

normal_scores = (
    isolation_forest.decision_function(
        X_normal_validation
    )
)


attack_scores = (
    isolation_forest.decision_function(
        X_attack
    )
)


print("Anomaly scores generated.")


# ============================================================
# 8. TEST DIFFERENT THRESHOLDS
# ============================================================

print("\n")
print("=" * 70)
print("TESTING ANOMALY THRESHOLDS")
print("=" * 70)


# We calculate thresholds from the unseen NORMAL
# validation data.
#
# For example:
#
# 95th percentile from the lower side gives us
# approximately 5% normal traffic flagged.
#
# Since lower score = more anomalous,
# we use low percentiles.

percentiles = [
    0.5,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15
]


thresholds = [
    np.percentile(
        normal_scores,
        percentile
    )
    for percentile in percentiles
]


results = []


# Combined validation data

all_scores = np.concatenate(
    [
        normal_scores,
        attack_scores
    ]
)


# Ground truth:
#
# 0 = Normal
# 1 = Attack

y_true = np.concatenate(
    [
        np.zeros(
            len(normal_scores),
            dtype=int
        ),

        np.ones(
            len(attack_scores),
            dtype=int
        )
    ]
)


for percentile, threshold in zip(
    percentiles,
    thresholds
):

    # Lower score than threshold
    # means anomaly.

    y_pred = (
        all_scores < threshold
    ).astype(int)


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()


    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )


    # --------------------------------------------------------
    # Attack Detection Rate / Recall
    # --------------------------------------------------------

    attack_detection_rate = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    results.append({

        "Normal Percentile":
            percentile,

        "Threshold":
            threshold,

        "False Positive Rate":
            false_positive_rate,

        "Attack Detection Rate":
            attack_detection_rate,

        "Precision":
            precision,

        "Recall":
            attack_detection_rate,

        "F1":
            f1
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# 9. DISPLAY THRESHOLD RESULTS
# ============================================================

display_df = results_df.copy()


for column in [
    "False Positive Rate",
    "Attack Detection Rate",
    "Precision",
    "Recall",
    "F1"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)


display_df["Threshold"] = (
    display_df["Threshold"]
    .round(6)
)


print("\n")

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# 10. FIND BEST THRESHOLD
# ============================================================

print("\n")
print("=" * 70)
print("SELECTING BEST THRESHOLD")
print("=" * 70)


# Only consider thresholds whose FPR
# is <= our target FPR.

acceptable_thresholds = (
    results_df[
        results_df[
            "False Positive Rate"
        ] <= TARGET_FPR
    ]
)


if len(acceptable_thresholds) == 0:

    print(
        "\nNo threshold satisfied "
        f"the target FPR of "
        f"{TARGET_FPR * 100:.1f}%."
    )

    # Fall back to threshold with
    # lowest FPR.

    best_row = (
        results_df
        .sort_values(
            by="False Positive Rate"
        )
        .iloc[0]
    )

else:

    # Among acceptable thresholds,
    # maximize attack detection rate.
    #
    # If tied, maximize F1.

    best_row = (
        acceptable_thresholds
        .sort_values(
            by=[
                "Attack Detection Rate",
                "F1"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )


best_threshold = (
    best_row["Threshold"]
)

best_fpr = (
    best_row["False Positive Rate"]
)

best_detection_rate = (
    best_row["Attack Detection Rate"]
)

best_precision = (
    best_row["Precision"]
)

best_recall = (
    best_row["Recall"]
)

best_f1 = (
    best_row["F1"]
)


# ============================================================
# 11. PRINT SELECTED THRESHOLD
# ============================================================

print(
    f"\nTarget False Positive Rate: "
    f"{TARGET_FPR * 100:.2f}%"
)

print(
    f"\nSelected threshold: "
    f"{best_threshold:.6f}"
)

print(
    f"False Positive Rate: "
    f"{best_fpr * 100:.2f}%"
)

print(
    f"Attack Detection Rate: "
    f"{best_detection_rate * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best_precision * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best_recall * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{best_f1 * 100:.2f}%"
)


# ============================================================
# 12. FINAL CONFUSION MATRIX
# ============================================================

final_predictions = (
    all_scores < best_threshold
).astype(int)


final_cm = confusion_matrix(
    y_true,
    final_predictions,
    labels=[0, 1]
)


tn, fp, fn, tp = (
    final_cm.ravel()
)


print("\n")
print("=" * 70)
print("FINAL CONFUSION MATRIX")
print("=" * 70)

print(
    "\nClass order:"
)

print(
    "[0 = Normal, 1 = Attack]"
)

print("\n")

print(final_cm)

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# 13. PER-CLASS ATTACK EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS ATTACK ANOMALY DETECTION")
print("=" * 70)


per_class_results = []


for attack_type in sorted(
    attack_df["Attack Type"].unique()
):

    class_mask = (
        attack_df["Attack Type"]
        .values
        == attack_type
    )

    class_scores = (
        attack_scores[
            class_mask
        ]
    )

    class_anomalies = (
        class_scores <
        best_threshold
    ).sum()

    class_total = (
        len(class_scores)
    )

    class_detection_rate = (
        class_anomalies /
        class_total
        if class_total > 0
        else 0
    )


    per_class_results.append({

        "Attack Type":
            attack_type,

        "Total Samples":
            class_total,

        "Anomalies Detected":
            class_anomalies,

        "Detection Rate":
            class_detection_rate
    })


per_class_df = pd.DataFrame(
    per_class_results
)


per_class_display = (
    per_class_df.copy()
)


per_class_display[
    "Detection Rate"
] = (
    per_class_display[
        "Detection Rate"
    ] * 100
).round(2)


print(
    per_class_display.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE MODEL ARTIFACT
# ============================================================

model_artifact = {

    "model":
        isolation_forest,

    "features":
        FEATURE_COLUMNS,

    "medians":
        training_medians.to_dict(),

    "threshold":
        float(best_threshold),

    "target_fpr":
        TARGET_FPR,

    "random_state":
        RANDOM_STATE
}


joblib.dump(
    model_artifact,
    MODEL_OUTPUT_PATH
)


# ============================================================
# 15. SAVE THRESHOLD SEPARATELY
# ============================================================

threshold_artifact = {

    "threshold":
        float(best_threshold),

    "target_fpr":
        TARGET_FPR,

    "false_positive_rate":
        float(best_fpr),

    "attack_detection_rate":
        float(best_detection_rate),

    "precision":
        float(best_precision),

    "recall":
        float(best_recall),

    "f1":
        float(best_f1)
}


joblib.dump(
    threshold_artifact,
    THRESHOLD_OUTPUT_PATH
)


# ============================================================
# 16. SAVE EVALUATION TABLES
# ============================================================

results_df.to_csv(
    "./models/"
    "anomaly_threshold_results.csv",
    index=False
)


per_class_df.to_csv(
    "./models/"
    "anomaly_per_class_results.csv",
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("IMPROVED ISOLATION FOREST COMPLETED")
print("=" * 70)

print(
    "\nModel saved at:"
)

print(
    MODEL_OUTPUT_PATH
)

print(
    "\nThreshold saved at:"
)

print(
    THRESHOLD_OUTPUT_PATH
)

print(
    "\nThreshold:"
)

print(
    f"{best_threshold:.6f}"
)

print(
    "\nTarget FPR:"
)

print(
    f"{TARGET_FPR * 100:.2f}%"
)

print("\n")