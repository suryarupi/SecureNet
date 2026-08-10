import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Load processed binary dataset
# --------------------------------------------------

DATA_PATH = "./data/processed/cicids2017_binary.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

TARGET = "Binary_Label"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# 3. Keep only numeric features
# --------------------------------------------------

X = X.select_dtypes(include=["number"])

print("Number of features:", X.shape[1])


# --------------------------------------------------
# 4. Handle invalid values
# --------------------------------------------------

X = X.replace([float("inf"), float("-inf")], float("nan"))

X = X.fillna(0)


# --------------------------------------------------
# 5. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Create XGBoost model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== XGBoost Results ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

MODEL_PATH = "./models/xgboost_binary.joblib"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")