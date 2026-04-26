import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Config visual
plt.style.use("dark_background")
sns.set_palette("husl")
BLUE = "#4F8EF7"
GREEN = "#2ECC71"
RED = "#E74C3C"
GRAY = "#95A5A6"

# Carrega dados e modelo
df_features = pd.read_csv("trades_processed_v4.csv")
df_raw = pd.read_csv("trades_enriched.csv")
df_raw["timestamp"] = pd.to_datetime(df_raw["block_timestamp"])
df_raw = df_raw.sort_values("timestamp")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

X = df_features.drop(columns=["log_price"])
y = df_features["log_price"]
split = int(len(df_features) * 0.8)
X_test = X.iloc[split:]
y_test = y.iloc[split:]
y_pred = model.predict(X_test)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("BAYC NFT — Modelo Preditivo de Preços", fontsize=16, fontweight="bold", y=1.01)

# ─ Gráfico 1: Preço ao longo do tempo ─
ax1 = axes[0, 0]
df_plot = df_raw[df_raw["price_eth"] > 0.1].copy()
df_daily = df_plot.set_index("timestamp")["price_eth"].resample("W").median()
ax1.plot(df_daily.index, df_daily.values, color=BLUE, linewidth=1.5)
ax1.fill_between(df_daily.index, df_daily.values, alpha=0.15, color=BLUE)
ax1.set_title("Preço mediano semanal (ETH)", fontsize=12)
ax1.set_ylabel("ETH")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
ax1.tick_params(axis="x", rotation=30)
ax1.grid(alpha=0.2)

# ─ Gráfico 2: Previsão vs Real ─
ax2 = axes[0, 1]
y_test_eth = np.expm1(y_test.values)
y_pred_eth = np.expm1(y_pred)
ax2.scatter(y_test_eth, y_pred_eth, alpha=0.4, s=15, color=BLUE)
max_val = max(y_test_eth.max(), y_pred_eth.max())
ax2.plot([0, max_val], [0, max_val], color=GREEN, linewidth=1.5, linestyle="--", label="Previsão perfeita")
ax2.set_title("Previsão vs Real (ETH)", fontsize=12)
ax2.set_xlabel("Preço real (ETH)")
ax2.set_ylabel("Preço previsto (ETH)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)
r2 = r2_score(y_test_eth, y_pred_eth)
rmse = np.sqrt(mean_squared_error(y_test_eth, y_pred_eth))
ax2.text(0.05, 0.92, f"R² = {r2:.3f}\nRMSE = {rmse:.2f} ETH",
         transform=ax2.transAxes, fontsize=9, color=GRAY,
         bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.7))

# ─ Gráfico 3: Importância das features ─
ax3 = axes[1, 0]
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
colors = [GREEN if v > 0.1 else BLUE for v in feat_imp.values]
ax3.barh(feat_imp.index, feat_imp.values, color=colors)
ax3.set_title("Importância das features", fontsize=12)
ax3.set_xlabel("Importância")
ax3.grid(alpha=0.2, axis="x")

# ─ Gráfico 4: Distribuição dos erros ─
ax4 = axes[1, 1]
erros = y_pred_eth - y_test_eth
ax4.hist(erros, bins=40, color=BLUE, alpha=0.7, edgecolor="none")
ax4.axvline(0, color=GREEN, linewidth=1.5, linestyle="--", label="Erro zero")
ax4.axvline(erros.mean(), color=RED, linewidth=1.5, linestyle="--", label=f"Média: {erros.mean():.2f} ETH")
ax4.set_title("Distribuição dos erros (ETH)", fontsize=12)
ax4.set_xlabel("Erro (previsto - real)")
ax4.set_ylabel("Frequência")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("resultados.png", dpi=150, bbox_inches="tight")
print("Gráfico salvo em resultados.png")
plt.show()