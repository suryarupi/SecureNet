import sys
import os
import pandas as pd


# ============================================================
# ALLOW IMPORT FROM PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(
    PROJECT_ROOT
)


from prediction2.predictor import (
    NetworkPredictor
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
# MAIN TEST
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("SECURENET PREDICTION LAYER TEST")
    print("=" * 80)


    # --------------------------------------------------------
    # Load predictor
    # --------------------------------------------------------

    predictor = NetworkPredictor()


    # --------------------------------------------------------
    # Load a few dataset samples
    # --------------------------------------------------------

    print("\nLoading test samples...")

    df = pd.read_csv(
        DATA_PATH
    )


    # Take a small sample first

    test_data = df.head(
        20
    ).copy()


    # Keep actual labels separately
    # ONLY for testing.

    actual_labels = None

    if "Attack Type" in test_data.columns:

        actual_labels = (
            test_data[
                "Attack Type"
            ].values
        )


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    print(
        "\nRunning predictions..."
    )


    results = predictor.predict(
        test_data
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)


    for i, result in enumerate(
        results
    ):

        print(
            f"\nFlow {i + 1}"
        )

        if actual_labels is not None:

            print(
                f"Actual Attack Type : "
                f"{actual_labels[i]}"
            )


        print(
            f"Binary Prediction   : "
            f"{result['binary_prediction']}"
        )

        print(
            f"Binary Confidence   : "
            f"{result['binary_confidence']:.4f}"
        )

        print(
            f"Multiclass Prediction: "
            f"{result['multiclass_prediction']}"
        )

        print(
            f"Multiclass Confidence: "
            f"{result['multiclass_confidence']:.4f}"
        )

        print(
            f"Anomaly Score       : "
            f"{result['anomaly_score']:.6f}"
        )

        print(
            f"Is Anomaly          : "
            f"{result['is_anomaly']}"
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)


    binary_attacks = sum(
        result[
            "binary_prediction"
        ] != "Normal Traffic"
        for result in results
    )


    anomalies = sum(
        result[
            "is_anomaly"
        ]
        for result in results
    )


    print(
        f"\nTotal flows tested : "
        f"{len(results)}"
    )

    print(
        f"Binary RF attacks  : "
        f"{binary_attacks}"
    )

    print(
        f"IF anomalies       : "
        f"{anomalies}"
    )


    print("\n")
    print(
        "Prediction layer test completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()