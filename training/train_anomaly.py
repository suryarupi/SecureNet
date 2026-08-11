import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "./data/processed/cicids2017_binary.csv"
)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. SELECT NORMAL TRAFFIC
# ============================================================

print("\nSelecting normal traffic...")

normal_df = df[
    df["Attack Type"] == "Normal Traffic"
].copy()

print(
    f"Normal traffic samples: "
    f"{len(normal_df)}"
)


# ============================================================
# 3. CREATE FEATURES
# ============================================================

X_normal = normal_df.drop(
    columns=[
        "Attack Type",
        "Binary_Label"
    ]
)


# ============================================================
# 4. CLEAN FEATURES
# ============================================================

print("\nCleaning features...")

# Replace infinity values
X_normal = X_normal.replace(
    [np.inf, -np.inf],
    np.nan
)

# Convert features to numeric
X_normal = X_normal.apply(
    pd.to_numeric,
    errors="coerce"
)

# Check missing values
missing_values = X_normal.isna().sum().sum()

print(
    f"Missing values: {missing_values}"
)

# Fill missing values
if missing_values > 0:

    print(
        "Filling missing values "
        "with column medians..."
    )

    X_normal = X_normal.fillna(
        X_normal.median()
    )


# ============================================================
# 5. DISPLAY INFORMATION
# ============================================================

print("\nFeature information:")

print(
    f"Number of features: "
    f"{X_normal.shape[1]}"
)

print(
    f"Number of normal samples: "
    f"{X_normal.shape[0]}"
)


# ============================================================
# 6. CREATE ISOLATION FOREST
# ============================================================

print("\nCreating Isolation Forest...")

isolation_forest = IsolationForest(

    n_estimators=200,

    contamination="auto",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining Isolation Forest...")
print(
    "The model is learning the "
    "normal traffic pattern..."
)

isolation_forest.fit(
    X_normal
)

print(
    "\nIsolation Forest training completed!"
)


# ============================================================
# 8. CHECK TRAINING DATA
# ============================================================

print(
    "\nChecking normal training data..."
)

normal_predictions = (
    isolation_forest.predict(X_normal)
)

normal_anomalies = (
    normal_predictions == -1
).sum()

normal_count = len(
    normal_predictions
)

normal_anomaly_percentage = (
    normal_anomalies /
    normal_count
) * 100


print(
    f"Normal samples classified "
    f"as anomalies: {normal_anomalies}"
)

print(
    f"Percentage flagged as anomaly: "
    f"{normal_anomaly_percentage:.2f}%"
)


# ============================================================
# 9. SAVE MODEL ARTIFACT
# ============================================================

model_artifact = {

    "model": isolation_forest,

    "features": list(
        X_normal.columns
    )
}


model_path = (
    "./models/isolation_forest.pkl"
)

joblib.dump(
    model_artifact,
    model_path
)


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("ISOLATION FOREST MODEL SAVED")
print("=" * 60)

print(
    f"\nModel saved at:\n"
    f"{model_path}"
)

print(
    "\nThe model was trained only "
    "on Normal Traffic."
)

print(
    "\nIt can now be used to detect "
    "traffic that deviates from "
    "normal network behavior."
)

print("\n")