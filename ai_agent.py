# ai_agent.py
import os
import pandas as pd
import numpy as np
from config import Config
from openai import OpenAI
from sklearn.linear_model import LinearRegression

DATASET_PATH = os.path.join("dataset", "stock_data.csv")

# Initialize Gemini client if provider is set
gemini_client = None
if Config.AI_PROVIDER.lower() == "gemini" and Config.GEMINI_API_KEY:
    gemini_client = OpenAI(
        api_key=Config.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def load_symbol_df(symbol):
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df = df[df["symbol"] == symbol].sort_values("date")
    return df if not df.empty else None

def analyze_with_ml(symbol):
    """
    Returns dictionary with latest_price, trend, volatility, moving averages, and signal.
    """
    df = load_symbol_df(symbol)
    if df is None:
        return {"error": "No dataset data available for symbol."}

    df = df.reset_index(drop=True)
    df["pct"] = df["close"].pct_change().fillna(0)
    vol = df["pct"].std() * np.sqrt(252)
    df["ma20"] = df["close"].rolling(window=20, min_periods=5).mean()
    df["ma50"] = df["close"].rolling(window=50, min_periods=10).mean()

    latest = df.iloc[-1]
    latest_price = float(latest["close"])
    ma20 = float(latest["ma20"]) if not pd.isna(latest["ma20"]) else None
    ma50 = float(latest["ma50"]) if not pd.isna(latest["ma50"]) else None

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["close"].values
    slope = float(LinearRegression().fit(X, y).coef_[0])
    trend = "Upward" if slope > 0 else "Downward" if slope < 0 else "Flat"

    # Simple Buy/Hold/Sell signal
    signal = "Hold"
    if ma20 and ma50:
        signal = "Buy" if ma20 > ma50 else "Sell" if ma20 < ma50 else "Hold"
    elif ma20:
        signal = "Buy" if latest_price > ma20 else "Sell"

    return {
        "symbol": symbol,
        "latest_price": latest_price,
        "trend": trend,
        "slope": slope,
        "volatility_annual": float(vol),
        "ma20": ma20,
        "ma50": ma50,
        "signal": signal,
        "data_points": len(df)
    }

def analyze_with_gemini(symbol):
    """
    Uses Google Gemini API to generate a concise analysis.
    """
    if not gemini_client:
        return {"error": "Gemini not configured or API key missing."}

    ml_summary = analyze_with_ml(symbol)
    if "error" in ml_summary:
        return ml_summary

    prompt = f"""
You are an expert stock market analyst. Analyze the stock {symbol} using the numeric summary below.
Do not give financial advice, just provide a short description (3-5 sentences) and a 1-line interpretation tag.

Summary:
- latest_price: {ml_summary['latest_price']}
- trend: {ml_summary['trend']}
- volatility_annual: {ml_summary['volatility_annual']}
- ma20: {ml_summary['ma20']}
- ma50: {ml_summary['ma50']}
- signal: {ml_summary['signal']}
- data_points: {ml_summary['data_points']}
"""

    try:
        response = gemini_client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": "You are a helpful stock analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )

        text = response.choices[0].message.content

        return {
            "ml_summary": ml_summary,
            "gemini_text": text
        }

    except Exception as e:
        return {"error": f"Exception calling Gemini: {e}"}
