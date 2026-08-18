import joblib
import pandas as pd
import numpy as np
import shap

# ============================================================
# Configuration
# ============================================================

BINARY_MODEL_PATH = "./models/xgboost_binary.joblib"
MULTICLASS_MODEL_PATH = "./models/xgboost_multiclass.joblib"
LABEL_ENCODER_PATH = "./models/multiclass_label_encoder.joblib"

DATA_PATH = "./data/raw/cicids2017_cleaned.csv"

# Number of top features to display
TOP_FEATURES = 10


# ============================================================
# Load Models
# ============================================================

print("Loading XGBoost models...")

binary_model = joblib.load(BINARY_MODEL_PATH)
multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Models loaded successfully!")


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(data):
    """
    Prepare network-flow data exactly as required
    by the trained XGBoost models.
    """

    X = data.copy()

    # Remove target columns if present
    columns_to_drop = [
        "Attack Type",
        "Binary_Label"
    ]

    X = X.drop(
        columns=[
            col for col in columns_to_drop
            if col in X.columns
        ],
        errors="ignore"
    )

    # Keep numeric features only
    X = X.select_dtypes(include=["number"])

    # Handle infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Handle missing values
    X = X.fillna(0)

    return X


# ============================================================
# Explain Binary XGBoost
# ============================================================

def explain_binary(flow):
    """
    Explain why Binary XGBoost classified
    the network flow as Normal or Attack.
    """

    # Convert input to DataFrame
    if isinstance(flow, pd.Series):
        df = pd.DataFrame([flow])

    elif isinstance(flow, dict):
        df = pd.DataFrame([flow])

    elif isinstance(flow, pd.DataFrame):
        df = flow.copy()

    else:
        raise TypeError(
            "Input must be dict, Series or DataFrame."
        )

    X = prepare_features(df)

    if X.empty:
        raise ValueError(
            "No numeric features found."
        )

    # Create SHAP Tree Explainer
    explainer = shap.TreeExplainer(binary_model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    # For binary XGBoost, obtain values for the sample
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[0])
    else:
        values = np.asarray(shap_values)

    values = values.reshape(-1)

    # Rank features by absolute SHAP value
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Value": values,
        "Absolute SHAP": np.abs(values)
    })

    feature_importance = feature_importance.sort_values(
        by="Absolute SHAP",
        ascending=False
    )

    return feature_importance.head(TOP_FEATURES)


# ============================================================
# Explain Multiclass XGBoost
# ============================================================

def explain_multiclass(flow):
    """
    Explain why Multiclass XGBoost selected
    the predicted attack type.
    """

    # Convert input to DataFrame
    if isinstance(flow, pd.Series):
        df = pd.DataFrame([flow])

    elif isinstance(flow, dict):
        df = pd.DataFrame([flow])

    elif isinstance(flow, pd.DataFrame):
        df = flow.copy()

    else:
        raise TypeError(
            "Input must be dict, Series or DataFrame."
        )

    X = prepare_features(df)

    if X.empty:
        raise ValueError(
            "No numeric features found."
        )

    # Predict attack class
    prediction = int(
        multiclass_model.predict(X)[0]
    )

    attack_type = label_encoder.inverse_transform(
        [prediction]
    )[0]

    # SHAP Tree Explainer
    explainer = shap.TreeExplainer(multiclass_model)

    shap_values = explainer.shap_values(X)

    # --------------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------------

    values = np.asarray(shap_values)

    if values.ndim == 3:
        # Shape can be:
        # samples × features × classes
        sample_values = values[0, :, prediction]

    elif values.ndim == 2:
        # Possible shape:
        # features × classes
        sample_values = values[:, prediction]

    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    # Rank features
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Value": sample_values,
        "Absolute SHAP": np.abs(sample_values)
    })

    feature_importance = feature_importance.sort_values(
        by="Absolute SHAP",
        ascending=False
    )

    return attack_type, feature_importance.head(TOP_FEATURES)


# ============================================================
# Test SHAP Explainability
# ============================================================

if __name__ == "__main__":

    print("\nLoading test data...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Pick one known attack
    # --------------------------------------------------------

    attack_sample = (
        df[df["Attack Type"] != "Normal Traffic"]
        .iloc[[0]]
    )

    actual_label = attack_sample[
        "Attack Type"
    ].iloc[0]

    print(
        f"\nActual Attack Type: {actual_label}"
    )

    # ========================================================
    # Binary Explanation
    # ========================================================

    print("\n========================================")
    print("       BINARY XGBOOST EXPLANATION")
    print("========================================")

    binary_explanation = explain_binary(
        attack_sample
    )

    print(binary_explanation.to_string(
        index=False
    ))

    # ========================================================
    # Multiclass Explanation
    # ========================================================

    print("\n========================================")
    print("      MULTICLASS XGBOOST EXPLANATION")
    print("========================================")

    attack_type, multiclass_explanation = (
        explain_multiclass(attack_sample)
    )

    print(
        f"\nPredicted Attack Type: {attack_type}"
    )

    print(
        multiclass_explanation.to_string(
            index=False
        )
    )

    print(
        "\nSHAP explainability completed successfully!"
    )