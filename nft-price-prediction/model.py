import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv("trades_processed_v4.csv")

X = df.drop(columns=["log_price"])
y = df["log_price"]

# Split temporal 80/20
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Treino: {len(X_train)} amostras")
print(f"Teste:  {len(X_test)} amostras")

# Cross-validation temporal para avaliar estabilidade
tscv = TimeSeriesSplit(n_splits=5)
cv_r2s, cv_rmses = [], []

for train_idx, val_idx in tscv.split(X_train):
    m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
    m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    pred = m.predict(X_train.iloc[val_idx])
    cv_r2s.append(r2_score(y_train.iloc[val_idx], pred))
    cv_rmses.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))

print(f"\nCross-validation (treino):")
print(f"  R² médio:   {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")
print(f"  RMSE médio: {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")

# Modelo final
print("\nTreinando modelo final...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rmse_eth = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

print(f"\n=== RESULTADOS FINAIS ===")
print(f"R²:         {r2:.4f}")
print(f"RMSE (log): {rmse:.4f}")
print(f"RMSE (ETH): {rmse_eth:.4f}")

print("\n=== IMPORTANCIA DAS FEATURES ===")
for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Simulação de previsão
print("\n=== SIMULACAO ===")
ultima = X_test.iloc[[-1]]
preco_previsto = np.expm1(model.predict(ultima)[0])
preco_real = np.expm1(y_test.iloc[-1])
print(f"Preço real:     {preco_real:.4f} ETH")
print(f"Preço previsto: {preco_previsto:.4f} ETH")
print(f"Erro:           {abs(preco_real - preco_previsto):.4f} ETH")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModelo salvo em model.pkl")