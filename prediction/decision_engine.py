import pandas as pd

from .predict import predict_flow
from .explain import explain_binary, explain_multiclass


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "./data/raw/cicids2017_cleaned.csv"

TOP_FEATURES = 5


# ============================================================
# Decision Engine
# ============================================================

def make_decision(prediction):
    """
    Convert raw ML predictions into a security decision.
    """

    attack_probability = prediction["attack_probability"]
    attack_confidence = prediction["attack_confidence"]
    anomaly = prediction["anomaly"]

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

# --------------------------------------------------------
# NORMAL / ANOMALY CHECK
# --------------------------------------------------------
 
    if prediction["binary_prediction"] == "Normal":

     if anomaly:

        return {
            "status": "SUSPICIOUS",
            "severity": "LOW",
            "attack_type": None,
            "reason": (
                "Binary XGBoost classified the traffic as normal, "
                "but Isolation Forest detected anomalous "
                "network behaviour."
            )
        }

     return {
        "status": "NORMAL",
        "severity": "NONE",
        "attack_type": None,
        "reason": (
            "Binary XGBoost classified the traffic as normal "
            "and no significant anomaly was detected."
        )
    }

    # --------------------------------------------------------
    # HIGH SEVERITY
    # --------------------------------------------------------

    if (
        attack_probability >= 0.90
        and
        attack_confidence is not None
        and
        attack_confidence >= 0.80
    ):

        return {
            "status": "ATTACK",
            "severity": "HIGH",
            "attack_type": prediction["attack_type"],
            "reason": (
                "High-confidence attack detected by "
                "Binary XGBoost and attack type identified "
                "with high confidence by Multiclass XGBoost."
            )
        }

    # --------------------------------------------------------
    # MEDIUM SEVERITY
    # --------------------------------------------------------

    if attack_probability >= 0.50:

        if (
            attack_confidence is not None
            and
            attack_confidence >= 0.50
        ):

            reason = (
                "XGBoost detected an attack and identified "
                "the attack type with moderate confidence."
            )

        else:

            reason = (
                "XGBoost strongly detected an attack, "
                "but the exact attack type has low confidence."
            )

        return {
            "status": "ATTACK",
            "severity": "MEDIUM",
            "attack_type": prediction["attack_type"],
            "reason": reason
        }

    # --------------------------------------------------------
    # LOW SEVERITY / ANOMALY
    # --------------------------------------------------------

    if anomaly:

        return {
            "status": "SUSPICIOUS",
            "severity": "LOW",
            "attack_type": prediction["attack_type"],
            "reason": (
                "Isolation Forest detected anomalous "
                "network behaviour."
            )
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "status": "SUSPICIOUS",
        "severity": "LOW",
        "attack_type": prediction["attack_type"],
        "reason": (
            "Traffic shows some indication of malicious "
            "behaviour but confidence is below the attack threshold."
        )
    }


# ============================================================
# Print Decision
# ============================================================

def print_decision(prediction, decision):
    """
    Print the final SecureNet decision.
    """

    print("\n========================================")
    print("         SECURENET DECISION")
    print("========================================")

    print(
        f"Status             : "
        f"{decision['status']}"
    )

    print(
        f"Severity           : "
        f"{decision['severity']}"
    )

    print(
        f"Attack Type        : "
        f"{decision['attack_type']}"
    )

    print(
        f"Attack Probability : "
        f"{prediction['attack_probability']:.6f}"
    )

    print(
        f"Attack Confidence  : "
        f"{prediction['attack_confidence']}"
    )

    print(
        f"Anomaly            : "
        f"{prediction['anomaly']}"
    )

    print(
        f"Anomaly Score      : "
        f"{prediction['anomaly_score']:.6f}"
    )

    print(
        f"Reason             : "
        f"{decision['reason']}"
    )

    print("========================================")


# ============================================================
# SHAP Explanation
# ============================================================

def print_shap_explanation(flow, prediction):
    """
    Generate and print SHAP explanations for an attack.
    """

    # Only explain attacks
    if prediction["binary_prediction"] != "Attack":
        return

    print("\n----------------------------------------")
    print("       SHAP EXPLAINABILITY")
    print("----------------------------------------")

    # --------------------------------------------------------
    # Binary XGBoost explanation
    # --------------------------------------------------------

    binary_explanation = explain_binary(flow)

    print("\nWhy was this classified as an ATTACK?")

    for _, row in binary_explanation.head(
        TOP_FEATURES
    ).iterrows():

        direction = (
            "toward ATTACK"
            if row["SHAP Value"] > 0
            else "toward NORMAL"
        )

        print(
            f"{row['Feature']:<30} "
            f"{row['SHAP Value']:>10.4f} "
            f"({direction})"
        )

    # --------------------------------------------------------
    # Multiclass explanation
    # --------------------------------------------------------

    if prediction["attack_type"] is not None:

        try:

            predicted_type, multiclass_explanation = (
                explain_multiclass(flow)
            )

            print(
                f"\nWhy was it classified as "
                f"{predicted_type}?"
            )

            for _, row in multiclass_explanation.head(
                TOP_FEATURES
            ).iterrows():

                direction = (
                    "supports class"
                    if row["SHAP Value"] > 0
                    else "opposes class"
                )

                print(
                    f"{row['Feature']:<30} "
                    f"{row['SHAP Value']:>10.4f} "
                    f"({direction})"
                )

        except Exception as error:

            print(
                "\nMulticlass SHAP explanation "
                f"could not be generated: {error}"
            )

    print("----------------------------------------")


# ============================================================
# Main Test Pipeline
# ============================================================

if __name__ == "__main__":

    print("Loading test data...")

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Select test flows
    # --------------------------------------------------------

    normal_samples = (
        df[
            df["Attack Type"] == "Normal Traffic"
        ]
        .head(2)
    )

    attack_samples = (
        df[
            df["Attack Type"] != "Normal Traffic"
        ]
        .groupby("Attack Type")
        .head(2)
    )

    test_samples = pd.concat(
        [
            normal_samples,
            attack_samples
        ]
    )

    print(
        f"Testing {len(test_samples)} "
        "network flows..."
    )

    # --------------------------------------------------------
    # Process each flow
    # --------------------------------------------------------

    for index, row in test_samples.iterrows():

        print("\n\n########################################")
        print(f"FLOW #{index}")
        print("########################################")

        print(
            f"Actual Label: "
            f"{row['Attack Type']}"
        )

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        prediction = predict_flow(row)

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        decision = make_decision(
            prediction
        )

        print_decision(
            prediction,
            decision
        )

        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        print_shap_explanation(
            row,
            prediction
        )