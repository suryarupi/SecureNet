import pandas as pd
df = pd.read_csv("./data/raw/cicids2017_cleaned.csv")
print(df.head())
print(df["Attack Type"].unique())