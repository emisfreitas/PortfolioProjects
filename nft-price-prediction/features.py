import pandas as pd
import numpy as np
import ast

df = pd.read_csv("trades_enriched.csv")
df["timestamp"] = pd.to_datetime(df["block_timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Remove linhas sem preço do ETH
df = df.dropna(subset=["eth_usd"])
print(f"Shape apos limpeza: {df.shape}")
print(f"Range de datas: {df['timestamp'].min()} até {df['timestamp'].max()}")


# Remove linhas sem preço do ETH
df = df.dropna(subset=["eth_usd"])

# Mantém só dados de 2025 para reduzir data drift
df = df[df["timestamp"].dt.year >= 2025].copy()
df = df.reset_index(drop=True)
print(f"Shape após filtro 2025: {df.shape}")

print(f"Shape apos limpeza: {df.shape}")


# Log no preço
df["log_price"] = np.log1p(df["price_eth"])

# Features de tempo
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year

# Features de histórico global (# toda a coleção ordenada por tempo)
df["price_lag_1"] = df["log_price"].shift(1)
df["price_lag_2"] = df["log_price"].shift(2)
df["price_lag_3"] = df["log_price"].shift(3)
df["price_rolling_mean_5"] = df["log_price"].rolling(5, min_periods=2).mean()
df["price_rolling_mean_10"] = df["log_price"].rolling(10, min_periods=3).mean()
df["price_rolling_std_5"] = df["log_price"].rolling(5, min_periods=2).std()

# Features de mercado
df["marketplace_encoded"] = df["marketplace"].astype("category").cat.codes
df["log_eth_usd"] = np.log1p(df["eth_usd"])
df["eth_usd_rolling_7"] = df["log_eth_usd"].rolling(7, min_periods=2).mean()

# Rarity score
def parse_traits(metadata_str):
    try:
        meta = ast.literal_eval(str(metadata_str))
        attrs = meta.get("attributes", [])
        return {a["traitType"]: a["value"] for a in attrs if "traitType" in a and "value" in a}
    except:
        return {}

print("Calculando rarity score...")
df["traits"] = df["metadata"].apply(parse_traits)

trait_counts = {}
for traits in df["traits"]:
    for trait_type, value in traits.items():
        if trait_type not in trait_counts:
            trait_counts[trait_type] = {}
        trait_counts[trait_type][value] = trait_counts[trait_type].get(value, 0) + 1

total = len(df)

def calc_rarity(traits):
    if not traits:
        return 0
    scores = []
    for trait_type, value in traits.items():
        count = trait_counts.get(trait_type, {}).get(value, 1)
        scores.append(1 / (count / total))
    return float(np.mean(scores))

df["rarity_score"] = df["traits"].apply(calc_rarity)
print(f"Rarity score — min: {df['rarity_score'].min():.2f}, max: {df['rarity_score'].max():.2f}")

# Remove NaN dos lags
df = df.dropna(subset=[
    "price_lag_1", "price_lag_2", "price_lag_3",
    "price_rolling_mean_5", "eth_usd_rolling_7"
])

features = [
    "log_price",
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_rolling_mean_5",
    "price_rolling_mean_10",
    "price_rolling_std_5",
    "hour",
    "day_of_week",
    "month",
    "year",
    "marketplace_encoded",
    "log_eth_usd",
    "eth_usd_rolling_7",
    "rarity_score"
]

df_final = df[features].copy()

print(f"Shape final: {df_final.shape}")
print("\nEstatísticas de preço (ETH):")
print(df["price_eth"].describe())

df_final.to_csv("trades_processed_v4.csv", index=False)
print("\nSalvo em trades_processed_v4.csv")