import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("./data/processed/cicids2017_binary.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(columns=["Attack Type", "Binary_Label"])

y = df["Binary_Label"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# -----------------------------
# Random Forest Model
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

rf.fit(X_train, y_train)

print("Training Completed!")

# -----------------------------
# Predictions
# -----------------------------
y_pred = rf.predict(X_test)

y_prob = rf.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc = roc_auc_score(y_test, y_prob)

print("\n========== RESULTS ==========\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc:.4f}")

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(rf, "./models/random_forest_model.pkl")

print("\nModel saved successfully!")