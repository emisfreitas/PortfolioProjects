import pandas as pd
import yfinance as yf

df_raw = pd.read_csv("trades_raw.csv")
df_raw["timestamp"] = pd.to_datetime(df_raw["block_timestamp"])
df_raw = df_raw[df_raw["price_eth"] > 0.1]
p99 = df_raw["price_eth"].quantile(0.99)
df_raw = df_raw[df_raw["price_eth"] <= p99]
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

print(f"Range de datas: {df_raw['timestamp'].min()} até {df_raw['timestamp'].max()}")

# Busca ETH com range correto
print("Buscando histórico de preço do ETH...")
eth = yf.download("ETH-USD", start="2021-01-01", end="2026-01-01", interval="1d", progress=False)
eth = eth[["Close"]].reset_index()
eth.columns = ["date", "eth_usd"]
eth["date"] = pd.to_datetime(eth["date"]).dt.date
print(f"{len(eth)} dias de histórico carregados")

df_raw["date"] = df_raw["timestamp"].dt.date
df_enriched = df_raw.merge(eth, on="date", how="left")

print(f"Nulos em eth_usd: {df_enriched['eth_usd'].isnull().sum()}")
print(df_enriched[["timestamp", "price_eth", "eth_usd"]].head())

df_enriched.to_csv("trades_enriched.csv", index=False)
print("Salvo em trades_enriched.csv")