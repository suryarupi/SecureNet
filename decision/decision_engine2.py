from typing import Dict, Any


class DecisionEngine:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        normal_binary_label=0,
        attack_binary_label=1
    ):
        """
        Decision Engine for SECURENET.

        Binary Random Forest:
            0 = Normal Traffic
            1 = Attack

        Multiclass Random Forest:
            Returns the specific traffic/attack class.

        Isolation Forest:
            is_anomaly = True  -> anomalous
            is_anomaly = False -> normal-like
        """

        self.normal_binary_label = (
            normal_binary_label
        )

        self.attack_binary_label = (
            attack_binary_label
        )


    # ========================================================
    # MAIN DECISION
    # ========================================================

    def decide(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required_fields = [
            "binary_prediction",
            "binary_confidence",
            "multiclass_prediction",
            "multiclass_confidence",
            "anomaly_score",
            "is_anomaly"
        ]


        missing_fields = [
            field
            for field in required_fields
            if field not in prediction
        ]


        if missing_fields:

            raise ValueError(
                "Prediction result is missing "
                f"required fields: {missing_fields}"
            )


        # ----------------------------------------------------
        # Extract predictions
        # ----------------------------------------------------

        binary_prediction = (
            prediction["binary_prediction"]
        )

        binary_confidence = float(
            prediction["binary_confidence"]
        )


        multiclass_prediction = (
            prediction["multiclass_prediction"]
        )

        multiclass_confidence = float(
            prediction["multiclass_confidence"]
        )


        anomaly_score = float(
            prediction["anomaly_score"]
        )


        is_anomaly = bool(
            prediction["is_anomaly"]
        )


        # ====================================================
        # CASE 1 — KNOWN ATTACK
        # ====================================================

        if binary_prediction == self.attack_binary_label:

            status = "Known Attack"

            attack_type = (
                multiclass_prediction
            )

            final_confidence = (
                multiclass_confidence
            )

            reason = (
                "Binary Random Forest classified "
                "the traffic as an attack."
            )


        # ====================================================
        # CASE 2 — POTENTIAL UNKNOWN ATTACK
        # ====================================================

        elif (
            binary_prediction
            == self.normal_binary_label
            and is_anomaly
        ):

            status = (
                "Potential Unknown Attack"
            )

            attack_type = (
                "Unknown / Unclassified"
            )

            final_confidence = (
                binary_confidence
            )

            reason = (
                "Binary Random Forest classified "
                "the traffic as normal, but Isolation "
                "Forest detected anomalous behavior."
            )


        # ====================================================
        # CASE 3 — NORMAL TRAFFIC
        # ====================================================

        elif (
            binary_prediction
            == self.normal_binary_label
            and not is_anomaly
        ):

            status = "Normal Traffic"

            attack_type = None

            final_confidence = (
                binary_confidence
            )

            reason = (
                "Binary Random Forest classified "
                "the traffic as normal and Isolation "
                "Forest detected no significant anomaly."
            )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        else:

            status = "Undetermined"

            attack_type = (
                multiclass_prediction
            )

            final_confidence = 0.0

            reason = (
                "The prediction combination did not "
                "match any expected decision rule."
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = {

            # Final decision
            "status":
                status,

            "attack_type":
                attack_type,

            "final_confidence":
                float(
                    final_confidence
                ),


            # Binary RF
            "binary_prediction":
                binary_prediction,

            "binary_confidence":
                binary_confidence,


            # Multiclass RF
            "multiclass_prediction":
                multiclass_prediction,

            "multiclass_confidence":
                multiclass_confidence,


            # Isolation Forest
            "anomaly_score":
                anomaly_score,

            "is_anomaly":
                is_anomaly,


            # Explanation
            "reason":
                reason
        }


        return result


    # ========================================================
    # BATCH DECISION
    # ========================================================

    def decide_batch(
        self,
        predictions
    ):

        results = []

        for prediction in predictions:

            results.append(
                self.decide(
                    prediction
                )
            )

        return results