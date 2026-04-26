import requests
import pandas as pd
import time
from config import BASE_URL, HEADERS, CHAIN

CONTRACT = "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"

def get_all_trades(contract, max_pages=50):
    """Coleta trades da coleção inteira de forma controlada."""
    url = f"{BASE_URL}/nft/{contract}/trades"
    params = {"chain": CHAIN, "limit": 100}

    all_trades = []
    page = 0

    while page < max_pages:
        r = requests.get(url, headers=HEADERS, params=params).json()
        results = r.get("result", [])
        all_trades.extend(results)
        page += 1

        print(f"Página {page} — {len(all_trades)} trades coletados")

        cursor = r.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
        time.sleep(0.3)  # evita rate limit

    return pd.DataFrame(all_trades)

if __name__ == "__main__":
    # max_pages=50 = até 5.000 trades — gasta ~500 CUs, seguro para o plano gratuito
    print("Coletando trades da coleção BAYC...")
    trades = get_all_trades(CONTRACT, max_pages=50)
    trades["price_eth"] = trades["price"].astype(float) / 1e18
    trades.to_csv("trades_raw.csv", index=False)
    print(f"\nTotal: {len(trades)} trades salvos em trades_raw.csv")