import pandas as pd

df = pd.read_csv("trades_enriched.csv")

print("=== SAMPLE CAMPO METADATA ===")
print(df["metadata"].iloc[0])
print("\n---")
print(df["metadata"].iloc[1])