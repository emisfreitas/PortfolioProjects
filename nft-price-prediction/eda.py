import pandas as pd

df = pd.read_csv("trades_raw.csv")

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUNAS ===")
print(df.columns.tolist())

print("\n=== PRIMEIRAS LINHAS ===")
print(df.head(3))

print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

print("\n=== ESTATISTICAS DE PREÇO ===")
print(df["price_eth"].describe())