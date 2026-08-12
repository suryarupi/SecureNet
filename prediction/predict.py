import joblib
import pandas as pd
import numpy as np

# ============================================================
# 1. Configuration
# ============================================================

BINARY_MODEL_PATH = "./models/xgboost_binary.joblib"

MULTICLASS_MODEL_PATH = (
    "./models/xgboost_multiclass.joblib"
)

LABEL_ENCODER_PATH = (
    "./models/multiclass_label_encoder.joblib"
)

ISOLATION_MODEL_PATH = (
    "./models/isolation_forest.joblib"
)

# Tuned Isolation Forest threshold
ISOLATION_THRESHOLD = -0.057862


# ============================================================
# 2. Load Models
# ============================================================

print("Loading SecureNet models...")

binary_model = joblib.load(
    BINARY_MODEL_PATH
)

multiclass_model = joblib.load(
    MULTICLASS_MODEL_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

isolation_model = joblib.load(
    ISOLATION_MODEL_PATH
)

print("All models loaded successfully!")


# ============================================================
# 3. Feature Preparation
# ============================================================

def prepare_features(data):
    """
    Prepare network flow features.
    """

    X = data.copy()

    # Remove labels if present
    columns_to_drop = [
        "Attack Type",
        "Binary_Label"
    ]

    X = X.drop(
        columns=[
            col
            for col in columns_to_drop
            if col in X.columns
        ],
        errors="ignore"
    )

    # Keep only numeric features
    X = X.select_dtypes(
        include=["number"]
    )

    # Replace infinities
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X = X.fillna(0)

    return X


# ============================================================
# 4. Prediction Function
# ============================================================

def predict_flow(flow):
    """
    Predict a single network flow.

    Input:
        dict / Series / DataFrame

    Output:
        Dictionary of predictions.
    """

    # --------------------------------------------------------
    # Convert input
    # --------------------------------------------------------

    if isinstance(flow, dict):

        df = pd.DataFrame([flow])

    elif isinstance(flow, pd.Series):

        df = pd.DataFrame([flow])

    elif isinstance(flow, pd.DataFrame):

        df = flow.copy()

    else:

        raise TypeError(
            "Input must be dict, "
            "Series or DataFrame."
        )

    # --------------------------------------------------------
    # Feature preparation
    # --------------------------------------------------------

    X = prepare_features(df)

    if X.empty:

        raise ValueError(
            "No numeric features found."
        )

    # ========================================================
    # Isolation Forest
    # ========================================================

    anomaly_score = float(
        isolation_model.decision_function(X)[0]
    )

    is_anomaly = (
        anomaly_score < ISOLATION_THRESHOLD
    )

    # ========================================================
    # Binary XGBoost
    # ========================================================

    binary_prediction = int(
        binary_model.predict(X)[0]
    )

    binary_probability = float(
        binary_model.predict_proba(X)[0][1]
    )

    if binary_prediction == 1:
        binary_label = "Attack"
    else:
        binary_label = "Normal"

    # ========================================================
    # Multiclass XGBoost
    # ========================================================

    attack_type = None
    attack_confidence = None

    if binary_prediction == 1:

        multiclass_prediction = int(
            multiclass_model.predict(X)[0]
        )

        probabilities = (
            multiclass_model.predict_proba(X)[0]
        )

        attack_confidence = float(
            np.max(probabilities)
        )

        attack_type = (
            label_encoder.inverse_transform(
                [multiclass_prediction]
            )[0]
        )

    # ========================================================
    # Return Results
    # ========================================================

    result = {

        # Isolation Forest
        "anomaly": bool(is_anomaly),
        "anomaly_score": anomaly_score,
        "anomaly_threshold": ISOLATION_THRESHOLD,

        # Binary XGBoost
        "binary_prediction": binary_label,
        "attack_probability": binary_probability,

        # Multiclass XGBoost
        "attack_type": attack_type,
        "attack_confidence": attack_confidence
    }

    return result