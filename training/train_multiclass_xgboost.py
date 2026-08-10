import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

from xgboost import XGBClassifier


# ==================================================
# 1. Load Dataset
# ==================================================

DATA_PATH = "./data/raw/cicids2017_cleaned.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ==================================================
# 2. Separate Features and Target
# ==================================================

TARGET = "Attack Type"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ==================================================
# 3. Convert Labels to Numbers
# ==================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClass Mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# ==================================================
# 4. Keep Numeric Features
# ==================================================

X = X.select_dtypes(include=["number"])

print(f"\nNumber of features: {X.shape[1]}")


# ==================================================
# 5. Handle Invalid Values
# ==================================================

X = X.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

X = X.fillna(0)


# ==================================================
# 6. Train/Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==================================================
# 7. Create XGBoost Multiclass Model
# ==================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),

    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,

    subsample=0.8,
    colsample_bytree=0.8,

    eval_metric="mlogloss",

    random_state=42,
    n_jobs=-1
)


# ==================================================
# 8. Train Model
# ==================================================

print("\nTraining Multiclass XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==================================================
# 9. Predictions
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 10. Evaluation
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n======================================")
print("      MULTICLASS XGBOOST RESULTS")
print("======================================")

print(f"Accuracy      : {accuracy:.4f}")
print(f"Macro F1      : {macro_f1:.4f}")
print(f"Macro Recall  : {macro_recall:.4f}")


# ==================================================
# 11. Classification Report
# ==================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==================================================
# 12. Confusion Matrix
# ==================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==================================================
# 13. Save Model
# ==================================================

MODEL_PATH = "./models/xgboost_multiclass.joblib"

joblib.dump(
    model,
    MODEL_PATH
)


# ==================================================
# 14. Save Label Encoder
# ==================================================

ENCODER_PATH = "./models/multiclass_label_encoder.joblib"

joblib.dump(
    label_encoder,
    ENCODER_PATH
)


print(f"\nModel saved to: {MODEL_PATH}")
print(f"Label encoder saved to: {ENCODER_PATH}")

print("\nMulticlass training completed successfully!")