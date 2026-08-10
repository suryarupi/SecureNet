import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# Load Dataset
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv("./data/processed/cicids2017_binary.csv")

print(f"Dataset shape: {df.shape}")


# ==========================================
# Basic Dataset Information
# ==========================================

print("\nColumns:")
print(df.columns.tolist())

print("\nAttack Type Distribution:")
print(df["Attack Type"].value_counts())


# ==========================================
# Features and Target
# ==========================================

# Attack Type is our multi-class target.
#
# Binary_Label was created only for the binary
# classifier, so we remove it here.

X = df.drop(
    columns=["Attack Type", "Binary_Label"]
)

y = df["Attack Type"]


# ==========================================
# Check Feature Types
# ==========================================

print("\nChecking feature data types...")

non_numeric_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

if non_numeric_columns:
    print("\nERROR: Non-numeric feature columns found:")
    print(non_numeric_columns)

    raise ValueError(
        "All model features must be numeric. "
        "Check your data cleaning step."
    )


# ==========================================
# Handle Infinite / Missing Values
# ==========================================

print("\nChecking missing and infinite values...")

# Replace infinite values with NaN
X = X.replace([float("inf"), float("-inf")], pd.NA)

# Convert possible missing values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Count missing values
missing_values = X.isna().sum().sum()

print(f"Missing values found: {missing_values}")

if missing_values > 0:

    print("Filling missing values with column medians...")

    X = X.fillna(X.median())


# ==========================================
# Final Dataset Information
# ==========================================

print("\nFinal feature shape:")
print(X.shape)

print("\nNumber of attack classes:")
print(y.nunique())

print("\nClasses:")

for attack_class in sorted(y.unique()):
    print(f" - {attack_class}")


# ==========================================
# Train-Test Split
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==========================================
# Random Forest Multi-Class Classifier
# ==========================================

print("\nCreating Random Forest model...")

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)


# ==========================================
# Train Model
# ==========================================

print("\nTraining Multi-Class Random Forest...")
print("This may take some time depending on dataset size...\n")

rf.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# Predictions
# ==========================================

print("\nGenerating predictions...")

y_pred = rf.predict(X_test)


# ==========================================
# Evaluation Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# Print Results
# ==========================================

print("\n")
print("=" * 60)
print("MULTI-CLASS RANDOM FOREST RESULTS")
print("=" * 60)

print(f"\nAccuracy        : {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall   : {recall:.4f}")
print(f"Weighted F1       : {f1:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")


# ==========================================
# Confusion Matrix
# ==========================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=rf.classes_
)

print("\nClass order:")
print(list(rf.classes_))

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# Classification Report
# ==========================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# Feature Importance
# ==========================================

print("\n")
print("=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(20).to_string(index=False)
)


# ==========================================
# Save Model
# ==========================================

model_path = "./models/multiclass_random_forest.pkl"

joblib.dump(
    rf,
    model_path
)

print("\n")
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"\nModel saved successfully at:")
print(model_path)


# ==========================================
# Save Feature Importance
# ==========================================

importance_path = (
    "./models/multiclass_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved at:")
print(importance_path)


# ==========================================
# Final Information
# ==========================================

print("\n")
print("=" * 60)
print("MULTI-CLASS TRAINING COMPLETED")
print("=" * 60)

print(f"\nNumber of classes: {len(rf.classes_)}")

print("\nClasses detected:")

for i, class_name in enumerate(rf.classes_):
    print(f"{i}: {class_name}")

print("\n")