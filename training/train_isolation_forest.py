import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "./data/raw/cicids2017_cleaned.csv"
MODEL_PATH = "./models/isolation_forest.joblib"

TARGET = "Attack Type"

# Fraction of normal traffic used for training.
# Keep this manageable because CICIDS2017 is large.
NORMAL_SAMPLE_SIZE = 500_000


# ============================================================
# 2. Load Dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. Separate Normal and Attack Traffic
# ============================================================

normal_df = df[df[TARGET] == "Normal Traffic"].copy()
attack_df = df[df[TARGET] != "Normal Traffic"].copy()

print(f"\nNormal samples available: {len(normal_df)}")
print(f"Attack samples available: {len(attack_df)}")


# ============================================================
# 4. Sample Normal Traffic for Training
# ============================================================

if len(normal_df) > NORMAL_SAMPLE_SIZE:
    normal_train = normal_df.sample(
        n=NORMAL_SAMPLE_SIZE,
        random_state=42
    )
else:
    normal_train = normal_df

print(f"Normal samples used for training: {len(normal_train)}")


# ============================================================
# 5. Prepare Features
# ============================================================

X_normal = normal_train.drop(columns=[TARGET])

X_normal = X_normal.select_dtypes(include=["number"])

X_normal = X_normal.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

X_normal = X_normal.fillna(0)


# ============================================================
# 6. Train Isolation Forest
# ============================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_normal)

print("Training completed!")


# ============================================================
# 7. Save Model
# ============================================================

os.makedirs("./models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")


# ============================================================
# 8. Test on Normal + Attack Traffic
# ============================================================

print("\nTesting anomaly detection...")

# Use a manageable sample for evaluation
normal_test = normal_df.drop(
    normal_train.index,
    errors="ignore"
)

if len(normal_test) > 100_000:
    normal_test = normal_test.sample(
        n=100_000,
        random_state=42
    )

if len(attack_df) > 100_000:
    attack_test = attack_df.sample(
        n=100_000,
        random_state=42
    )
else:
    attack_test = attack_df


test_df = pd.concat(
    [normal_test, attack_test],
    ignore_index=True
)

X_test = test_df.drop(columns=[TARGET])

X_test = X_test.select_dtypes(include=["number"])

X_test = X_test.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

X_test = X_test.fillna(0)


# ============================================================
# 9. Isolation Forest Prediction
# ============================================================

predictions = model.predict(X_test)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

test_df["Prediction"] = predictions

test_df["Predicted_Label"] = test_df["Prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})


# ============================================================
# 10. Evaluation
# ============================================================

# Ground truth:
# Normal Traffic = 1
# Attack = -1

y_true = test_df[TARGET].apply(
    lambda x: 1 if x == "Normal Traffic" else -1
)

y_pred = test_df["Prediction"]


print("\n======================================")
print("       ISOLATION FOREST RESULTS")
print("======================================")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[1, -1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[1, -1]
    )
)


# ============================================================
# 11. Attack Detection by Type
# ============================================================

print("\nAttack Detection by Type:")

attack_results = test_df[
    test_df[TARGET] != "Normal Traffic"
].copy()

for attack_type in sorted(
    attack_results[TARGET].unique()
):

    subset = attack_results[
        attack_results[TARGET] == attack_type
    ]

    detected = (
        subset["Prediction"] == -1
    ).sum()

    total = len(subset)

    recall = detected / total if total > 0 else 0

    print(
        f"{attack_type:20s} "
        f"{detected:6d}/{total:6d} "
        f"({recall:.2%})"
    )


print("\nIsolation Forest training and evaluation completed!")