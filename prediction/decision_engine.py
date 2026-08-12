"""
SecureNet Decision Engine

Combines:
    1. Isolation Forest
    2. Binary XGBoost
    3. Multiclass XGBoost

The Decision Engine converts raw ML predictions into
a final security decision.
"""

from predict import predict_flow


# ============================================================
# Configuration
# ============================================================

# Binary XGBoost attack probability thresholds
HIGH_ATTACK_PROBABILITY = 0.90
MEDIUM_ATTACK_PROBABILITY = 0.50

# Multiclass confidence thresholds
HIGH_TYPE_CONFIDENCE = 0.80
MEDIUM_TYPE_CONFIDENCE = 0.50


# ============================================================
# Decision Engine
# ============================================================

def make_decision(prediction):
    """
    Convert raw ML predictions into a final security decision.

    Parameters
    ----------
    prediction : dict
        Output returned by predict_flow()

    Returns
    -------
    dict
        Final security decision.
    """

    # --------------------------------------------------------
    # Extract model results
    # --------------------------------------------------------

    binary_prediction = prediction["binary_prediction"]
    attack_probability = prediction["attack_probability"]

    attack_type = prediction["attack_type"]
    attack_confidence = prediction["attack_confidence"]

    anomaly = prediction["anomaly"]
    anomaly_score = prediction["anomaly_score"]

    # ========================================================
    # CASE 1: Binary XGBoost detected an attack
    # ========================================================

    if binary_prediction == "Attack":

        # ----------------------------------------------------
        # Determine attack severity
        # ----------------------------------------------------

        if attack_probability >= HIGH_ATTACK_PROBABILITY:

            # Strong attack probability
            if (
                attack_confidence is not None
                and attack_confidence >= HIGH_TYPE_CONFIDENCE
            ):
                severity = "HIGH"

            elif (
                attack_confidence is not None
                and attack_confidence >= MEDIUM_TYPE_CONFIDENCE
            ):
                severity = "HIGH"

            else:
                # Binary model is highly confident,
                # but attack type is uncertain.
                severity = "MEDIUM"

        elif attack_probability >= MEDIUM_ATTACK_PROBABILITY:

            severity = "MEDIUM"

        else:

            severity = "LOW"

        # ----------------------------------------------------
        # Determine explanation
        # ----------------------------------------------------

        if attack_confidence is not None:

            if attack_confidence >= HIGH_TYPE_CONFIDENCE:

                reason = (
                    "High-confidence attack detected by "
                    "Binary XGBoost and attack type identified "
                    "with high confidence by Multiclass XGBoost."
                )

            elif attack_confidence >= MEDIUM_TYPE_CONFIDENCE:

                reason = (
                    "XGBoost detected an attack and identified "
                    "the attack type with moderate confidence."
                )

            else:

                reason = (
                    "XGBoost strongly detected an attack, "
                    "but the exact attack type has low confidence."
                )

        else:

            reason = (
                "Binary XGBoost detected attack traffic."
            )

        # ----------------------------------------------------
        # Return attack decision
        # ----------------------------------------------------

        return {
            "status": "ATTACK",
            "severity": severity,
            "attack_type": attack_type,
            "attack_probability": attack_probability,
            "attack_confidence": attack_confidence,
            "anomaly": anomaly,
            "anomaly_score": anomaly_score,
            "reason": reason
        }

    # ========================================================
    # CASE 2: Binary XGBoost says normal,
    #         but Isolation Forest detects anomaly
    # ========================================================

    if anomaly:

        return {
            "status": "SUSPICIOUS",
            "severity": "MEDIUM",
            "attack_type": None,
            "attack_probability": attack_probability,
            "attack_confidence": attack_confidence,
            "anomaly": anomaly,
            "anomaly_score": anomaly_score,
            "reason": (
                "Isolation Forest detected traffic that "
                "significantly differs from learned normal "
                "network behavior."
            )
        }

    # ========================================================
    # CASE 3: Normal traffic
    # ========================================================

    return {
        "status": "NORMAL",
        "severity": "NONE",
        "attack_type": None,
        "attack_probability": attack_probability,
        "attack_confidence": attack_confidence,
        "anomaly": anomaly,
        "anomaly_score": anomaly_score,
        "reason": (
            "No significant attack probability or anomaly "
            "was detected."
        )
    }


# ============================================================
# Human-readable output
# ============================================================

def print_decision(decision):
    """
    Print a security decision in a readable format.
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
        f"{decision['attack_probability']:.6f}"
    )

    print(
        f"Attack Confidence  : "
        f"{decision['attack_confidence']}"
    )

    print(
        f"Anomaly            : "
        f"{decision['anomaly']}"
    )

    print(
        f"Anomaly Score      : "
        f"{decision['anomaly_score']:.6f}"
    )

    print(
        f"Reason             : "
        f"{decision['reason']}"
    )

    print("========================================")


# ============================================================
# Test Decision Engine
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    DATA_PATH = "./data/raw/cicids2017_cleaned.csv"

    print("Loading test data...")

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Select test samples
    # --------------------------------------------------------

    normal_samples = df[
        df["Attack Type"] == "Normal Traffic"
    ].head(2)

    attack_samples = (
        df[df["Attack Type"] != "Normal Traffic"]
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
    # Run prediction + decision
    # --------------------------------------------------------

    for index, row in test_samples.iterrows():

        print("\n\n########################################")
        print(f"FLOW #{index}")
        print("########################################")

        print(
            f"Actual Label: "
            f"{row['Attack Type']}"
        )

        # Run ML prediction pipeline
        prediction = predict_flow(row)

        # Run Decision Engine
        decision = make_decision(prediction)

        # Print final result
        print_decision(decision)