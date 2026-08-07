import pandas as pd

# Load dataset
df = pd.read_csv("./data/raw/cicids2017_cleaned.csv")

# Create binary target
df["Binary_Label"] = (df["Attack Type"] != "Normal Traffic").astype(int)

# Check the mapping
print(df[["Attack Type", "Binary_Label"]].head())

# Check class distribution
print("\nBinary Label Distribution:")
print(df["Binary_Label"].value_counts())

# Save processed dataset
df.to_csv("./data/processed/cicids2017_binary.csv", index=False)

print("\nBinary dataset saved successfully!")