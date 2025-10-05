# generate_dataset.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

OUT_DIR = "dataset"
OUT_FILE = os.path.join(OUT_DIR, "stock_data.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# create a list of tickers (mix of US and Indian style tickers with .NS)
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA",
    "TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS"
]

# target number of rows
TARGET_ROWS = 12000

# create a sequence of dates backwards from today
days_needed = int(np.ceil(TARGET_ROWS / len(TICKERS))) + 10
end_date = datetime.now().date()
dates = [end_date - timedelta(days=i) for i in range(days_needed)]
dates = sorted(dates)

rows = []
# Generate per-ticker base price and volatility
for ticker in TICKERS:
    base = random.uniform(50, 3500) if not ticker.endswith(".NS") else random.uniform(100, 3500)
    vol = random.uniform(0.5, 3.5)  # percent daily movement typical
    price = base
    for d in dates:
        # simulate weekends/holidays: include all days but we can random-skip some
        if random.random() < 0.02:  # small chance market closed
            continue
        # random walk
        daily_pct = np.random.normal(loc=0, scale=vol/100.0)
        open_p = price
        close_p = max(0.1, price * (1 + daily_pct))
        high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.01)))
        low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(abs(np.random.normal(1_000_000, 300_000))) if not ticker.endswith(".NS") else int(abs(np.random.normal(500_000, 200_000)))
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "symbol": ticker,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": int(volume)
        })
        price = close_p
        # stop if enough rows
        if len(rows) >= TARGET_ROWS:
            break
    if len(rows) >= TARGET_ROWS:
        break

df = pd.DataFrame(rows)
# shuffle so it's not perfectly grouped
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUT_FILE, index=False)
print(f"Generated {len(df)} rows to {OUT_FILE}")
