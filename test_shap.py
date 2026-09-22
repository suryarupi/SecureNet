import pandas as pd

from explainability.shap_explainer import (
    explain_binary,
    explain_multiclass
)

DATA_PATH = "data/raw/cicids2017_cleaned.csv"

print("=" * 60)
print("SecureNet SHAP Integration Test")
print("=" * 60)

# Read CSV in chunks until an attack flow is found
attack_sample = None

for chunk in pd.read_csv(DATA_PATH, chunksize=10000):
    attacks = chunk[chunk["Attack Type"] != "Normal Traffic"]

    if not attacks.empty:
        attack_sample = attacks.iloc[[0]]
        break

if attack_sample is None:
    raise RuntimeError("No attack sample found in dataset.")

actual_attack = attack_sample["Attack Type"].iloc[0]

print(f"\nActual attack type: {actual_attack}")

print("\n" + "=" * 60)
print("BINARY XGBOOST SHAP")
print("=" * 60)

binary_result = explain_binary(
    attack_sample,
    top_n=10
)

print(binary_result.to_string(index=False))

print("\n" + "=" * 60)
print("MULTICLASS XGBOOST SHAP")
print("=" * 60)

multiclass_result = explain_multiclass(
    attack_sample,
    top_n=10
)

print(multiclass_result.to_string(index=False))

print("\n" + "=" * 60)
print("SHAP TEST COMPLETED")
print("=" * 60)