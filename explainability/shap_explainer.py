"""
SecureNet SHAP Explainability

Provides SHAP-based explanations for:
1. Binary XGBoost model
2. Multiclass XGBoost model

Also generates human-readable explanations for:
- NORMAL flows
- SUSPICIOUS flows
- ATTACK flows

Important:
SHAP explanations in this file explain the XGBoost models.
Isolation Forest is used for anomaly detection and is not
directly explained using SHAP here.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

BINARY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "xgboost_binary.joblib"
)

MULTICLASS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "xgboost_multiclass.joblib"
)

LABEL_ENCODER_PATH = os.path.join(
    MODELS_DIR,
    "multiclass_label_encoder.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

binary_model = joblib.load(
    BINARY_MODEL_PATH
)

multiclass_model = joblib.load(
    MULTICLASS_MODEL_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(model):
    """
    Get feature names from an XGBoost model.
    """

    # XGBoost sklearn wrapper
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # XGBoost Booster
    if hasattr(model, "get_booster"):

        booster = model.get_booster()

        if booster.feature_names is not None:
            return list(booster.feature_names)

    raise ValueError(
        "Could not determine feature names from the model."
    )


BINARY_FEATURE_NAMES = get_feature_names(
    binary_model
)

MULTICLASS_FEATURE_NAMES = get_feature_names(
    multiclass_model
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

if len(BINARY_FEATURE_NAMES) != 52:

    raise ValueError(
        f"Binary XGBoost has "
        f"{len(BINARY_FEATURE_NAMES)} features. "
        f"SecureNet expects exactly 52."
    )


if len(MULTICLASS_FEATURE_NAMES) != 52:

    raise ValueError(
        f"Multiclass XGBoost has "
        f"{len(MULTICLASS_FEATURE_NAMES)} features. "
        f"SecureNet expects exactly 52."
    )


if BINARY_FEATURE_NAMES != MULTICLASS_FEATURE_NAMES:

    raise ValueError(
        "Binary and multiclass XGBoost models "
        "do not use the same feature order."
    )


FEATURE_NAMES = BINARY_FEATURE_NAMES


# ============================================================
# SHAP EXPLAINERS
# ============================================================

binary_explainer = shap.TreeExplainer(
    binary_model
)

multiclass_explainer = shap.TreeExplainer(
    multiclass_model
)


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_input(
    flow_data,
    feature_names=None
):
    """
    Convert flow data into a one-row DataFrame
    containing exactly the model features.

    Accepted input:
        - dict
        - pandas Series
        - pandas DataFrame
    """

    if feature_names is None:
        feature_names = FEATURE_NAMES


    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(flow_data, dict):

        df = pd.DataFrame(
            [flow_data]
        )

    elif isinstance(flow_data, pd.Series):

        df = flow_data.to_frame().T

    elif isinstance(flow_data, pd.DataFrame):

        df = flow_data.copy()

    else:

        raise TypeError(
            "flow_data must be a dict, pandas Series, "
            "or pandas DataFrame."
        )


    # --------------------------------------------------------
    # Remove non-model columns
    # --------------------------------------------------------

    columns_to_drop = [
        "Attack Type",
        "Binary_Label"
    ]

    for column in columns_to_drop:

        if column in df.columns:

            df = df.drop(
                columns=[column]
            )


    # --------------------------------------------------------
    # Check for missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features:\n"
            + ", ".join(missing_features)
        )


    # --------------------------------------------------------
    # Keep exact model feature order
    # --------------------------------------------------------

    df = df[
        feature_names
    ].copy()


    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    for column in feature_names:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Replace NaN
    # --------------------------------------------------------

    df = df.fillna(0)


    return df


# ============================================================
# BINARY SHAP VALUE EXTRACTION
# ============================================================

def extract_binary_shap_values(
    shap_values
):
    """
    Normalize SHAP binary output into a
    one-dimensional feature vector.

    Handles:
        - list
        - 1D array
        - 2D array
        - 3D array
    """

    # --------------------------------------------------------
    # Older SHAP versions may return a list
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        # Binary classification:
        # usually [class_0, class_1]
        if len(shap_values) >= 2:

            values = np.asarray(
                shap_values[1]
            )

        else:

            values = np.asarray(
                shap_values[0]
            )

    else:

        values = np.asarray(
            shap_values
        )


    # --------------------------------------------------------
    # 3D
    # --------------------------------------------------------

    if values.ndim == 3:

        # samples x features x classes
        if values.shape[0] == 1:

            # Take the first sample
            values = values[0]

        else:

            # Use first sample
            values = values[0]


    # --------------------------------------------------------
    # 2D
    # --------------------------------------------------------

    if values.ndim == 2:

        # One sample x features
        if values.shape[0] == 1:

            values = values[0]

        # Features x classes
        elif values.shape[1] == 2:

            # Positive / attack class
            values = values[:, 1]

        else:

            # Fallback: first sample
            values = values[0]


    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    values = np.asarray(
        values
    ).reshape(-1)


    return values


# ============================================================
# MULTICLASS SHAP VALUE EXTRACTION
# ============================================================

def extract_multiclass_shap_values(
    shap_values,
    predicted_class,
    num_features,
    num_classes
):
    """
    Extract SHAP values for the predicted
    multiclass class.

    Supports different SHAP output formats.
    """

    # --------------------------------------------------------
    # SHAP returns list
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if predicted_class >= len(
            shap_values
        ):

            raise ValueError(
                f"Predicted class {predicted_class} "
                f"is outside SHAP output range."
            )


        values = np.asarray(
            shap_values[predicted_class]
        )


        if values.ndim == 2:

            return values[0]


        return values.reshape(-1)


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    values = np.asarray(
        shap_values
    )


    # --------------------------------------------------------
    # 3D
    # --------------------------------------------------------

    if values.ndim == 3:

        # samples x features x classes
        if (
            values.shape[0] == 1
            and values.shape[1] == num_features
            and values.shape[2] == num_classes
        ):

            return values[
                0,
                :,
                predicted_class
            ]


        # samples x classes x features
        if (
            values.shape[0] == 1
            and values.shape[1] == num_classes
            and values.shape[2] == num_features
        ):

            return values[
                0,
                predicted_class,
                :
            ]


        # Generic fallback
        values = values[0]


    # --------------------------------------------------------
    # 2D
    # --------------------------------------------------------

    if values.ndim == 2:

        # Features x classes
        if (
            values.shape[0] == num_features
            and values.shape[1] == num_classes
        ):

            return values[
                :,
                predicted_class
            ]


        # Classes x features
        if (
            values.shape[0] == num_classes
            and values.shape[1] == num_features
        ):

            return values[
                predicted_class,
                :
            ]


        # One sample x features
        if values.shape[0] == 1:

            return values[0]


        # Fallback
        return values[0]


    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    values = values.reshape(-1)


    if len(values) != num_features:

        raise ValueError(
            "Unexpected multiclass SHAP output shape."
        )


    return values


# ============================================================
# BINARY EXPLANATION
# ============================================================

def explain_binary(
    flow_data,
    top_n=10
):
    """
    Generate binary XGBoost SHAP explanation.

    Returns:
        pandas DataFrame with:

            feature
            value
            shap_value
    """

    df = prepare_input(
        flow_data,
        BINARY_FEATURE_NAMES
    )


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = (
        binary_explainer.shap_values(
            df
        )
    )


    values = extract_binary_shap_values(
        shap_values
    )


    # --------------------------------------------------------
    # Validate SHAP output
    # --------------------------------------------------------

    if len(values) != len(
        BINARY_FEATURE_NAMES
    ):

        raise ValueError(
            "Binary SHAP output does not match "
            "the number of model features."
        )


    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = pd.DataFrame({

        "feature":
            BINARY_FEATURE_NAMES,

        "value":
            df.iloc[0].values,

        "shap_value":
            values
    })


    # --------------------------------------------------------
    # Sort by importance
    # --------------------------------------------------------

    result["abs_shap"] = (
        result["shap_value"].abs()
    )


    result = result.sort_values(
        "abs_shap",
        ascending=False
    )


    result = result.head(
        top_n
    )


    result = result.drop(
        columns=["abs_shap"]
    )


    return result.reset_index(
        drop=True
    )


# ============================================================
# MULTICLASS EXPLANATION
# ============================================================

def explain_multiclass(
    flow_data,
    predicted_class=None,
    top_n=10
):
    """
    Generate multiclass XGBoost SHAP explanation.

    predicted_class:
        Integer class index.

    If predicted_class is None,
    the multiclass XGBoost model determines it.
    """

    df = prepare_input(
        flow_data,
        MULTICLASS_FEATURE_NAMES
    )


    # --------------------------------------------------------
    # Determine predicted class
    # --------------------------------------------------------

    if predicted_class is None:

        prediction = (
            multiclass_model.predict(
                df
            )
        )

        predicted_class = int(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )


    predicted_class = int(
        predicted_class
    )


    # --------------------------------------------------------
    # Determine number of classes
    # --------------------------------------------------------

    try:

        num_classes = len(
            label_encoder.classes_
        )

    except Exception:

        probabilities = (
            multiclass_model.predict_proba(
                df
            )
        )

        num_classes = len(
            probabilities[0]
        )


    # --------------------------------------------------------
    # Calculate SHAP
    # --------------------------------------------------------

    shap_values = (
        multiclass_explainer.shap_values(
            df
        )
    )


    values = extract_multiclass_shap_values(
        shap_values,
        predicted_class,
        len(MULTICLASS_FEATURE_NAMES),
        num_classes
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(values) != len(
        MULTICLASS_FEATURE_NAMES
    ):

        raise ValueError(
            "Multiclass SHAP output does not match "
            "the number of model features."
        )


    # --------------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame({

        "feature":
            MULTICLASS_FEATURE_NAMES,

        "value":
            df.iloc[0].values,

        "shap_value":
            values
    })


    # --------------------------------------------------------
    # Sort by absolute SHAP value
    # --------------------------------------------------------

    result["abs_shap"] = (
        result["shap_value"].abs()
    )


    result = result.sort_values(
        "abs_shap",
        ascending=False
    )


    result = result.head(
        top_n
    )


    result = result.drop(
        columns=["abs_shap"]
    )


    return result.reset_index(
        drop=True
    )


# ============================================================
# GENERIC SHAP EXPLANATION
# ============================================================

def generate_explanation(
    flow_data,
    prediction_type="binary",
    predicted_class=None,
    top_n=5
):
    """
    Generate SHAP results as a list of dictionaries.
    """

    prediction_type = (
        prediction_type.lower()
    )


    if prediction_type == "binary":

        result = explain_binary(
            flow_data,
            top_n=top_n
        )


    elif prediction_type == "multiclass":

        result = explain_multiclass(
            flow_data,
            predicted_class=predicted_class,
            top_n=top_n
        )


    else:

        raise ValueError(
            "prediction_type must be "
            "'binary' or 'multiclass'."
        )


    return result.to_dict(
        orient="records"
    )


# ============================================================
# FORMAT FEATURE VALUE
# ============================================================

def format_feature_value(
    value
):
    """
    Format feature values for readable
    dashboard explanations.
    """

    try:

        value = float(
            value
        )


        if abs(value) >= 1_000_000:

            return f"{value:,.2f}"


        if abs(value) >= 1_000:

            return f"{value:,.2f}"


        if abs(value) >= 1:

            return f"{value:.2f}"


        return f"{value:.4f}"


    except (
        TypeError,
        ValueError
    ):

        return str(value)


# ============================================================
# BINARY HUMAN-READABLE SUMMARY
# ============================================================

def generate_binary_summary(
    flow_data,
    status=None,
    anomaly=None,
    top_n=5
):
    """
    Generate a human-readable explanation
    for the binary XGBoost model.
    """

    status = (
        str(status).upper()
        if status is not None
        else None
    )


    explanation = explain_binary(
        flow_data,
        top_n=top_n
    )


    # --------------------------------------------------------
    # Features supporting attack classification
    # --------------------------------------------------------

    positive_features = explanation[
        explanation["shap_value"] > 0
    ]


    # --------------------------------------------------------
    # Features supporting normal classification
    # --------------------------------------------------------

    negative_features = explanation[
        explanation["shap_value"] < 0
    ]


    # --------------------------------------------------------
    # Format positive features
    # --------------------------------------------------------

    positive_text = []

    for _, row in positive_features.iterrows():

        positive_text.append(
            f"{row['feature']} "
            f"({format_feature_value(row['value'])})"
        )


    # --------------------------------------------------------
    # Format negative features
    # --------------------------------------------------------

    negative_text = []

    for _, row in negative_features.iterrows():

        negative_text.append(
            f"{row['feature']} "
            f"({format_feature_value(row['value'])})"
        )


    # ========================================================
    # ATTACK
    # ========================================================

    if status == "ATTACK":

        summary = (
            "Binary XGBoost classified this "
            "flow as an attack."
        )


        if positive_text:

            summary += (
                " The strongest features contributing "
                "toward the attack classification include "
                + ", ".join(
                    positive_text[:3]
                )
                + "."
            )


        if negative_text:

            summary += (
                " Features contributing away from the "
                "attack classification include "
                + ", ".join(
                    negative_text[:3]
                )
                + "."
            )


        return [summary]


    # ========================================================
    # SUSPICIOUS
    # ========================================================

    if status == "SUSPICIOUS":

        summary = (
            "Binary XGBoost classified this traffic "
            "as normal, but Isolation Forest detected "
            "anomalous network behaviour."
        )


        if positive_text:

            summary += (
                " SHAP indicates that the strongest "
                "features contributing toward the attack "
                "side of the binary XGBoost model include "
                + ", ".join(
                    positive_text[:3]
                )
                + "."
            )


        return [summary]


    # ========================================================
    # NORMAL
    # ========================================================

    summary = (
        "Binary XGBoost classified the traffic "
        "as normal and no significant anomaly "
        "was detected."
    )


    if negative_text:

        summary += (
            " SHAP indicates that features contributing "
            "away from the attack classification include "
            + ", ".join(
                negative_text[:3]
            )
            + "."
        )


    return [summary]


# ============================================================
# MULTICLASS HUMAN-READABLE SUMMARY
# ============================================================

def generate_multiclass_summary(
    flow_data,
    attack_type=None,
    predicted_class=None,
    top_n=5
):
    """
    Generate a human-readable explanation
    for the multiclass XGBoost model.
    """

    # --------------------------------------------------------
    # Determine predicted class
    # --------------------------------------------------------

    if predicted_class is None:

        prepared = prepare_input(
            flow_data,
            MULTICLASS_FEATURE_NAMES
        )


        prediction = (
            multiclass_model.predict(
                prepared
            )
        )


        predicted_class = int(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )


    predicted_class = int(
        predicted_class
    )


    # --------------------------------------------------------
    # Decode attack type
    # --------------------------------------------------------

    if attack_type is None:

        try:

            attack_type = (
                label_encoder.inverse_transform(
                    [predicted_class]
                )[0]
            )

        except Exception:

            attack_type = str(
                predicted_class
            )


    # --------------------------------------------------------
    # Get SHAP explanation
    # --------------------------------------------------------

    explanation = explain_multiclass(
        flow_data,
        predicted_class=predicted_class,
        top_n=top_n
    )


    positive_features = explanation[
        explanation["shap_value"] > 0
    ]


    negative_features = explanation[
        explanation["shap_value"] < 0
    ]


    # --------------------------------------------------------
    # Positive features
    # --------------------------------------------------------

    positive_text = []

    for _, row in positive_features.iterrows():

        positive_text.append(
            f"{row['feature']} "
            f"({format_feature_value(row['value'])})"
        )


    # --------------------------------------------------------
    # Negative features
    # --------------------------------------------------------

    negative_text = []

    for _, row in negative_features.iterrows():

        negative_text.append(
            f"{row['feature']} "
            f"({format_feature_value(row['value'])})"
        )


    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = (
        f"Multiclass XGBoost classified "
        f"the flow as {attack_type}."
    )


    if positive_text:

        summary += (
            " The strongest features supporting "
            "this classification include "
            + ", ".join(
                positive_text[:3]
            )
            + "."
        )


    if negative_text:

        summary += (
            " Features contributing away from "
            "this classification include "
            + ", ".join(
                negative_text[:3]
            )
            + "."
        )


    return [summary]


# ============================================================
# COMPLETE FLOW SUMMARY
# ============================================================

def generate_flow_summary(
    flow_data,
    status=None,
    attack_type=None,
    anomaly=None,
    predicted_class=None,
    top_n=5
):
    """
    Generate the complete SHAP explanation
    for a SecureNet network flow.

    ATTACK:
        Binary XGBoost explanation
        +
        Multiclass XGBoost explanation

    SUSPICIOUS:
        Binary XGBoost explanation
        +
        Isolation Forest anomaly information

    NORMAL:
        Binary XGBoost explanation
    """

    status = (
        str(status).upper()
        if status is not None
        else None
    )


    # ========================================================
    # ATTACK FLOW
    # ========================================================

    if status == "ATTACK":

        binary_summary = (
            generate_binary_summary(
                flow_data,
                status="ATTACK",
                anomaly=anomaly,
                top_n=top_n
            )
        )


        multiclass_summary = (
            generate_multiclass_summary(
                flow_data,
                attack_type=attack_type,
                predicted_class=predicted_class,
                top_n=top_n
            )
        )


        return (
            binary_summary
            + multiclass_summary
        )


    # ========================================================
    # SUSPICIOUS FLOW
    # ========================================================

    if status == "SUSPICIOUS":

        return generate_binary_summary(
            flow_data,
            status="SUSPICIOUS",
            anomaly=anomaly,
            top_n=top_n
        )


    # ========================================================
    # NORMAL FLOW
    # ========================================================

    return generate_binary_summary(
        flow_data,
        status="NORMAL",
        anomaly=anomaly,
        top_n=top_n
    )


# ============================================================
# TEST / DEBUG
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SecureNet SHAP Explainer")
    print("=" * 60)

    print(
        f"Binary features     : "
        f"{len(BINARY_FEATURE_NAMES)}"
    )

    print(
        f"Multiclass features : "
        f"{len(MULTICLASS_FEATURE_NAMES)}"
    )

    print(
        f"Binary model        : "
        f"{BINARY_MODEL_PATH}"
    )

    print(
        f"Multiclass model    : "
        f"{MULTICLASS_MODEL_PATH}"
    )

    print(
        "SHAP explainers loaded successfully."
    )

    print("=" * 60)