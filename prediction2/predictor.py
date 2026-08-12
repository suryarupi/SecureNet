import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


BINARY_RF_PATH = os.path.join(
    MODELS_DIR,
    "binary_random_forest.pkl"
)

MULTICLASS_RF_PATH = os.path.join(
    MODELS_DIR,
    "multiclass_random_forest.pkl"
)

ISOLATION_FOREST_PATH = os.path.join(
    MODELS_DIR,
    "isolation_forest_improved.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "anomaly_threshold.pkl"
)


# ============================================================
# PREDICTOR CLASS
# ============================================================

class NetworkPredictor:

    def __init__(self):

        print("\nLoading prediction models...")

        # ----------------------------------------------------
        # Binary Random Forest
        # ----------------------------------------------------

        self.binary_model = joblib.load(
            BINARY_RF_PATH
        )

        print(
            "✓ Binary Random Forest loaded"
        )


        # ----------------------------------------------------
        # Multiclass Random Forest
        # ----------------------------------------------------

        self.multiclass_model = joblib.load(
            MULTICLASS_RF_PATH
        )

        print(
            "✓ Multiclass Random Forest loaded"
        )


        # ----------------------------------------------------
        # Improved Isolation Forest
        # ----------------------------------------------------

        isolation_artifact = joblib.load(
            ISOLATION_FOREST_PATH
        )

        # Your improved IF was saved as a dictionary:
        #
        # {
        #     "model": ...,
        #     "features": ...,
        #     "medians": ...,
        #     "threshold": ...
        # }

        self.isolation_model = (
            isolation_artifact["model"]
        )

        self.isolation_features = (
            isolation_artifact["features"]
        )

        self.isolation_medians = (
            isolation_artifact.get(
                "medians",
                {}
            )
        )

        # Prefer threshold stored inside
        # improved model artifact.

        if "threshold" in isolation_artifact:

            self.anomaly_threshold = (
                isolation_artifact["threshold"]
            )

        else:

            threshold_artifact = (
                joblib.load(
                    THRESHOLD_PATH
                )
            )

            self.anomaly_threshold = (
                threshold_artifact["threshold"]
            )


        print(
            "✓ Improved Isolation Forest loaded"
        )

        print(
            f"✓ Anomaly threshold: "
            f"{self.anomaly_threshold:.6f}"
        )


        # ----------------------------------------------------
        # Model feature information
        # ----------------------------------------------------

        self.binary_features = (
            self._get_model_features(
                self.binary_model
            )
        )

        self.multiclass_features = (
            self._get_model_features(
                self.multiclass_model
            )
        )


        print(
            "\nModels loaded successfully."
        )


        # ----------------------------------------------------
        # Print classes
        # ----------------------------------------------------

        print(
            "\nBinary RF classes:"
        )

        print(
            self.binary_model.classes_
        )


        print(
            "\nMulticlass RF classes:"
        )

        print(
            self.multiclass_model.classes_
        )


    # ========================================================
    # GET MODEL FEATURES
    # ========================================================

    def _get_model_features(
        self,
        model
    ):

        if hasattr(
            model,
            "feature_names_in_"
        ):

            return list(
                model.feature_names_in_
            )

        raise ValueError(
            "Model does not contain "
            "feature_names_in_. "
            "Please provide the exact "
            "training feature list."
        )


    # ========================================================
    # CLEAN INPUT
    # ========================================================

    def _clean_input(
        self,
        data
    ):

        X = data.copy()


        # ----------------------------------------------------
        # Remove target columns if present
        # ----------------------------------------------------

        columns_to_remove = [
            "Attack Type",
            "Binary_Label"
        ]

        X = X.drop(
            columns=[
                col
                for col in columns_to_remove
                if col in X.columns
            ],
            errors="ignore"
        )


        # ----------------------------------------------------
        # Replace infinity
        # ----------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )


        # ----------------------------------------------------
        # Convert to numeric
        # ----------------------------------------------------

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )


        return X


    # ========================================================
    # PREPARE FEATURES FOR RANDOM FOREST
    # ========================================================

    def _prepare_rf_features(
        self,
        data,
        features
    ):

        X = self._clean_input(
            data
        )


        # ----------------------------------------------------
        # Check missing feature columns
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in features
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Input is missing "
                f"{len(missing_features)} "
                "required RF features.\n"
                f"Missing features: "
                f"{missing_features[:10]}"
            )


        # ----------------------------------------------------
        # EXACT feature order
        # ----------------------------------------------------

        X = X.reindex(
            columns=features
        )


        # ----------------------------------------------------
        # Fill missing values
        # ----------------------------------------------------

        X = X.fillna(
            X.median()
        )

        X = X.fillna(0)


        return X


    # ========================================================
    # PREPARE FEATURES FOR ISOLATION FOREST
    # ========================================================

    def _prepare_if_features(
        self,
        data
    ):

        X = self._clean_input(
            data
        )


        # ----------------------------------------------------
        # Check missing feature columns
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in self.isolation_features
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Input is missing "
                f"{len(missing_features)} "
                "required Isolation Forest "
                "features.\n"
                f"Missing features: "
                f"{missing_features[:10]}"
            )


        # ----------------------------------------------------
        # EXACT feature order
        # ----------------------------------------------------

        X = X.reindex(
            columns=self.isolation_features
        )


        # ----------------------------------------------------
        # Apply training medians
        # ----------------------------------------------------

        if self.isolation_medians:

            median_series = pd.Series(
                self.isolation_medians
            )

            X = X.fillna(
                median_series
            )


        X = X.fillna(0)


        return X


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        data
    ):

        # Make sure input is a DataFrame.

        if isinstance(
            data,
            pd.Series
        ):

            data = data.to_frame().T


        if not isinstance(
            data,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a "
                "pandas DataFrame."
            )


        if len(data) == 0:

            raise ValueError(
                "Input DataFrame is empty."
            )


        # ====================================================
        # BINARY RANDOM FOREST
        # ====================================================

        X_binary = (
            self._prepare_rf_features(
                data,
                self.binary_features
            )
        )


        binary_predictions = (
            self.binary_model.predict(
                X_binary
            )
        )


        binary_probabilities = (
            self.binary_model.predict_proba(
                X_binary
            )
        )


        binary_confidence = (
            np.max(
                binary_probabilities,
                axis=1
            )
        )


        # ====================================================
        # MULTICLASS RANDOM FOREST
        # ====================================================

        X_multiclass = (
            self._prepare_rf_features(
                data,
                self.multiclass_features
            )
        )


        multiclass_predictions = (
            self.multiclass_model.predict(
                X_multiclass
            )
        )


        multiclass_probabilities = (
            self.multiclass_model.predict_proba(
                X_multiclass
            )
        )


        multiclass_confidence = (
            np.max(
                multiclass_probabilities,
                axis=1
            )
        )


        # ====================================================
        # ISOLATION FOREST
        # ====================================================

        X_isolation = (
            self._prepare_if_features(
                data
            )
        )


        anomaly_scores = (
            self.isolation_model
            .decision_function(
                X_isolation
            )
        )


        # IMPORTANT:
        #
        # score < calibrated threshold
        # means anomaly.

        anomaly_predictions = (
            anomaly_scores
            <
            self.anomaly_threshold
        )


        # ====================================================
        # BUILD RESULT
        # ====================================================

        results = []


        for i in range(
            len(data)
        ):

            result = {

                # --------------------------------------------
                # Binary RF
                # --------------------------------------------

                "binary_prediction":
                    binary_predictions[i],

                "binary_confidence":
                    float(
                        binary_confidence[i]
                    ),


                # --------------------------------------------
                # Multiclass RF
                # --------------------------------------------

                "multiclass_prediction":
                    multiclass_predictions[i],

                "multiclass_confidence":
                    float(
                        multiclass_confidence[i]
                    ),


                # --------------------------------------------
                # Isolation Forest
                # --------------------------------------------

                "anomaly_score":
                    float(
                        anomaly_scores[i]
                    ),

                "is_anomaly":
                    bool(
                        anomaly_predictions[i]
                    )
            }


            results.append(
                result
            )


        return results


    # ========================================================
    # SINGLE FLOW PREDICTION
    # ========================================================

    def predict_one(
        self,
        flow
    ):

        if isinstance(
            flow,
            dict
        ):

            flow = pd.DataFrame(
                [flow]
            )


        results = self.predict(
            flow
        )


        return results[0]