import sys
import os
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(
    PROJECT_ROOT
)


# ============================================================
# IMPORTS
# ============================================================

from prediction2.predictor import (
    NetworkPredictor
)

from decision.decision_engine2 import (
    DecisionEngine
)


# ============================================================
# DATASET
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "cicids2017_binary.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SECURENET DECISION ENGINE TEST")
    print("=" * 80)


    # --------------------------------------------------------
    # Load prediction layer
    # --------------------------------------------------------

    predictor = NetworkPredictor()


    # --------------------------------------------------------
    # Load decision engine
    # --------------------------------------------------------

    decision_engine = (
        DecisionEngine()
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )


    # ========================================================
    # SELECT DIFFERENT TYPES OF TRAFFIC
    # ========================================================

    print(
        "\nSelecting test samples..."
    )


    # --------------------------------------------------------
    # Normal sample
    # --------------------------------------------------------

    normal_samples = df[
        df["Attack Type"]
        == "Normal Traffic"
    ]


    # --------------------------------------------------------
    # Attack sample
    # --------------------------------------------------------

    attack_samples = df[
        df["Attack Type"]
        != "Normal Traffic"
    ]


    if len(normal_samples) == 0:

        raise ValueError(
            "No normal samples found."
        )


    if len(attack_samples) == 0:

        raise ValueError(
            "No attack samples found."
        )


    # --------------------------------------------------------
    # Select small test set
    # --------------------------------------------------------

    test_normal = (
        normal_samples
        .sample(
            n=min(
                10,
                len(normal_samples)
            ),
            random_state=42
        )
    )


    test_attack = (
        attack_samples
        .sample(
            n=min(
                10,
                len(attack_samples)
            ),
            random_state=42
        )
    )


    test_data = pd.concat(
        [
            test_normal,
            test_attack
        ]
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nRunning prediction layer..."
    )


    predictions = predictor.predict(
        test_data
    )


    # ========================================================
    # DECISION ENGINE
    # ========================================================

    print(
        "\nRunning decision engine..."
    )


    decisions = (
        decision_engine.decide_batch(
            predictions
        )
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DECISION RESULTS")
    print("=" * 80)


    for i, (
        actual,
        prediction,
        decision
    ) in enumerate(
        zip(
            test_data[
                "Attack Type"
            ].values,
            predictions,
            decisions
        )
    ):

        print("\n" + "-" * 80)

        print(
            f"Flow {i + 1}"
        )

        print(
            f"Actual Label        : "
            f"{actual}"
        )

        print(
            f"Binary Prediction   : "
            f"{prediction['binary_prediction']}"
        )

        print(
            f"Binary Confidence   : "
            f"{prediction['binary_confidence']:.4f}"
        )

        print(
            f"Multiclass Prediction: "
            f"{prediction['multiclass_prediction']}"
        )

        print(
            f"Multiclass Confidence: "
            f"{prediction['multiclass_confidence']:.4f}"
        )

        print(
            f"Anomaly Score       : "
            f"{prediction['anomaly_score']:.6f}"
        )

        print(
            f"Is Anomaly          : "
            f"{prediction['is_anomaly']}"
        )

        print(
            f"FINAL STATUS        : "
            f"{decision['status']}"
        )

        print(
            f"ATTACK TYPE         : "
            f"{decision['attack_type']}"
        )

        print(
            f"FINAL CONFIDENCE    : "
            f"{decision['final_confidence']:.4f}"
        )

        print(
            f"REASON              : "
            f"{decision['reason']}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DECISION SUMMARY")
    print("=" * 80)


    decision_series = pd.Series(
        [
            result["status"]
            for result in decisions
        ]
    )


    print(
        "\nFinal decision counts:"
    )

    print(
        decision_series.value_counts()
    )


    # ========================================================
    # CHECK ALL VALID STATUS VALUES
    # ========================================================

    allowed_statuses = {
        "Normal Traffic",
        "Known Attack",
        "Potential Unknown Attack"
    }


    invalid_statuses = set(
        decision_series.unique()
    ) - allowed_statuses


    if invalid_statuses:

        print(
            "\nWARNING:"
        )

        print(
            "Invalid statuses found:"
        )

        print(
            invalid_statuses
        )

    else:

        print(
            "\n✓ All decision statuses "
            "are valid."
        )


    print("\n")
    print(
        "Decision Engine test completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()