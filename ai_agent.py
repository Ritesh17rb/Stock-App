import os
import pandas as pd
import numpy as np
from config import Config
from openai import OpenAI
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# Initialize Gemini client if provider is set
gemini_client = None
if Config.AI_PROVIDER.lower() == "gemini" and Config.GEMINI_API_KEY:
    try:
        gemini_client = OpenAI(
            api_key=Config.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        print("Gemini client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")


def load_stock_data(symbol, data_path="stock_data"):
    """
    Load stock data from your NIFTY-50 CSV files
    """
    try:
        file_path = os.path.join(data_path, f"{symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            return df
        else:
            # Fallback to yfinance if local data not available
            return None
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return None


def enhanced_analyze_with_ml(symbol):
    """
    Enhanced ML analysis with technical indicators and predictions
    """
    df = load_stock_data(symbol)

    if df is None or len(df) < 30:
        return {"error": f"Insufficient data for {symbol} or symbol not found"}

    try:
        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Get latest data
        latest = df.iloc[-1]
        latest_price = float(latest['Close'])

        # Trend analysis
        trend, slope = calculate_trend(df)

        # Volatility
        volatility = calculate_volatility(df)

        # Moving averages
        ma20 = float(df['MA_20'].iloc[-1]) if 'MA_20' in df.columns else None
        ma50 = float(df['MA_50'].iloc[-1]) if 'MA_50' in df.columns else None

        # RSI
        rsi = float(df['RSI_14'].iloc[-1]) if 'RSI_14' in df.columns else None

        # Generate trading signals
        signal, confidence = generate_trading_signal(df)

        # Price prediction (simple)
        prediction = predict_next_price(df)

        return {
            "symbol": symbol,
            "latest_price": latest_price,
            "trend": trend,
            "slope": slope,
            "volatility_annual": volatility,
            "ma20": ma20,
            "ma50": ma50,
            "rsi": rsi,
            "signal": signal,
            "confidence": confidence,
            "prediction": prediction,
            "data_points": len(df),
            "last_date": df['Date'].iloc[-1].strftime('%Y-%m-%d'),
            "analysis_type": "Enhanced Technical Analysis"
        }

    except Exception as e:
        return {"error": f"Analysis error for {symbol}: {str(e)}"}


def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    data = df.copy()

    # Moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # RSI
    data['RSI_14'] = calculate_rsi(data['Close'])

    # MACD
    data['MACD'] = calculate_macd(data['Close'])

    # Bollinger Bands
    data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])

    # Volume indicators
    if 'Volume' in data.columns:
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']

    # Price changes
    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Range'] = (data['High'] - data['Low']) / data['Close']

    return data.dropna()


def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, slow=26, fast=12):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return ema_fast - ema_slow


def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


def calculate_trend(df):
    """Calculate price trend using linear regression"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    slope = float(LinearRegression().fit(X, y).coef_[0])

    if slope > 0.01:
        return "Strong Uptrend", slope
    elif slope > 0:
        return "Weak Uptrend", slope
    elif slope < -0.01:
        return "Strong Downtrend", slope
    elif slope < 0:
        return "Weak Downtrend", slope
    else:
        return "Sideways", slope


def calculate_volatility(df, window=252):
    """Calculate annualized volatility"""
    returns = df['Close'].pct_change().dropna()
    if len(returns) < window:
        window = len(returns)
    return returns.tail(window).std() * np.sqrt(252)


def generate_trading_signal(df):
    """Generate trading signal based on multiple indicators"""
    if len(df) < 20:
        return "Insufficient Data", "low"

    latest = df.iloc[-1]
    signals = []
    confidence_factors = []

    # Moving Average Signal
    if pd.notna(latest.get('MA_20')) and pd.notna(latest.get('MA_50')):
        if latest['MA_20'] > latest['MA_50']:
            signals.append("BUY")
            confidence_factors.append(0.3)
        else:
            signals.append("SELL")
            confidence_factors.append(0.3)

    # RSI Signal
    if pd.notna(latest.get('RSI_14')):
        rsi = latest['RSI_14']
        if rsi < 30:
            signals.append("BUY")
            confidence_factors.append(0.4)
        elif rsi > 70:
            signals.append("SELL")
            confidence_factors.append(0.4)
        else:
            signals.append("HOLD")
            confidence_factors.append(0.2)

    # Price vs MA Signal
    if pd.notna(latest.get('MA_20')):
        if latest['Close'] > latest['MA_20']:
            signals.append("BUY")
            confidence_factors.append(0.3)
        else:
            signals.append("SELL")
            confidence_factors.append(0.3)

    # Determine final signal
    if not signals:
        return "HOLD", "medium"

    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    hold_count = signals.count("HOLD")

    if buy_count > sell_count and buy_count > hold_count:
        confidence = "high" if np.mean(confidence_factors) > 0.3 else "medium"
        return "BUY", confidence
    elif sell_count > buy_count and sell_count > hold_count:
        confidence = "high" if np.mean(confidence_factors) > 0.3 else "medium"
        return "SELL", confidence
    else:
        return "HOLD", "medium"


def predict_next_price(df, days=1):
    """Simple price prediction using recent trend"""
    if len(df) < 10:
        return {"error": "Insufficient data for prediction"}

    recent_prices = df['Close'].tail(10).values
    X = np.arange(len(recent_prices)).reshape(-1, 1)
    y = recent_prices

    model = LinearRegression()
    model.fit(X, y)

    next_price = model.predict([[len(recent_prices)]])[0]
    price_change = ((next_price - recent_prices[-1]) / recent_prices[-1]) * 100

    return {
        "predicted_price": round(float(next_price), 2),
        "predicted_change_percent": round(float(price_change), 2),
        "prediction_horizon": f"{days} day",
        "method": "Linear Regression"
    }


def interactive_gemini_analysis(symbol, user_question=None, analysis_type="comprehensive"):
    """
    Enhanced Gemini analysis with interactive capabilities
    """
    if not gemini_client:
        return {"error": "Gemini not configured or API key missing."}

    # Get enhanced ML analysis
    ml_analysis = enhanced_analyze_with_ml(symbol)
    if "error" in ml_analysis:
        return ml_analysis

    # Build context-aware prompt
    if user_question:
        prompt = f"""
        As a professional stock analyst, answer the user's specific question about {symbol} stock.

        USER QUESTION: {user_question}

        TECHNICAL ANALYSIS CONTEXT:
        - Current Price: ${ml_analysis['latest_price']}
        - Trend: {ml_analysis['trend']}
        - Volatility: {ml_analysis['volatility_annual']:.4f}
        - RSI: {ml_analysis.get('rsi', 'N/A')}
        - Moving Average (20): {ml_analysis['ma20']}
        - Moving Average (50): {ml_analysis['ma50']}
        - Trading Signal: {ml_analysis['signal']} (Confidence: {ml_analysis['confidence']})
        - Predicted Price: {ml_analysis['prediction'].get('predicted_price', 'N/A')}
        - Data Points: {ml_analysis['data_points']}

        Please provide a helpful, educational response that:
        1. Directly addresses the user's question
        2. References the technical context where relevant
        3. Avoids financial advice but offers insights
        4. Is clear and concise (2-3 paragraphs max)
        """
    else:
        prompt = f"""
        As a professional stock analyst, provide a comprehensive analysis of {symbol} stock.

        TECHNICAL ANALYSIS DATA:
        - Current Price: ${ml_analysis['latest_price']}
        - Trend: {ml_analysis['trend']}
        - Volatility: {ml_analysis['volatility_annual']:.4f}
        - RSI: {ml_analysis.get('rsi', 'N/A')}
        - Moving Average (20): {ml_analysis['ma20']}
        - Moving Average (50): {ml_analysis['ma50']}
        - Trading Signal: {ml_analysis['signal']} (Confidence: {ml_analysis['confidence']})
        - Predicted Price: {ml_analysis['prediction'].get('predicted_price', 'N/A')}
        - Data Points: {ml_analysis['data_points']}

        Please provide a structured analysis covering:
        1. Current technical position and momentum
        2. Key support and resistance levels
        3. Risk assessment based on volatility and trend
        4. Short-term outlook (1-4 weeks)
        5. Important factors to monitor

        Keep the analysis educational and avoid specific financial advice.
        Format your response in clear paragraphs with emojis for better readability.
        """

    try:
        response = gemini_client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional stock market analyst. Provide balanced, data-driven insights. Always clarify that your analysis is for educational purposes only and not financial advice. Use clear, engaging language with occasional emojis for better readability."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )

        analysis_text = response.choices[0].message.content

        return {
            "symbol": symbol,
            "analysis": analysis_text,
            "ml_context": ml_analysis,
            "user_question": user_question,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_type": "Interactive Gemini Analysis"
        }

    except Exception as e:
        return {"error": f"Gemini analysis failed: {str(e)}"}


def analyze_portfolio_with_gemini(portfolio_data):
    """
    Analyze entire portfolio using Gemini
    """
    if not gemini_client:
        return {"error": "Gemini not configured"}

    # Prepare portfolio summary
    portfolio_summary = "PORTFOLIO HOLDINGS:\n"
    total_value = 0
    total_investment = 0

    for position in portfolio_data.get('positions', []):
        current_value = position.get('current_value', position.get('qty', 0) * position.get('avg_price', 0))
        investment = position.get('qty', 0) * position.get('avg_price', 0)
        pnl = current_value - investment

        portfolio_summary += f"- {position['symbol']}: {position.get('qty', 0)} shares | Avg Cost: ${position.get('avg_price', 0):.2f} | Current: ${position.get('current_price', 0):.2f} | P&L: ${pnl:.2f}\n"

        total_value += current_value
        total_investment += investment

    portfolio_summary += f"\nPORTFOLIO SUMMARY:\n"
    portfolio_summary += f"Total Investment: ${total_investment:.2f}\n"
    portfolio_summary += f"Current Value: ${total_value:.2f}\n"
    portfolio_summary += f"Total P&L: ${total_value - total_investment:.2f}\n"
    portfolio_summary += f"Return: {((total_value - total_investment) / total_investment * 100):.2f}%"

    prompt = f"""
    Analyze this investment portfolio and provide educational insights:

    {portfolio_summary}

    Please provide:
    1. Diversification assessment across sectors
    2. Risk analysis based on holdings
    3. General observations about portfolio composition
    4. Suggestions for factors to consider (not specific advice)
    5. Overall portfolio health assessment

    Keep the analysis educational and avoid specific buy/sell recommendations.
    """

    try:
        response = gemini_client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": "You are a portfolio management expert providing educational insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )

        return {
            "portfolio_analysis": response.choices[0].message.content,
            "portfolio_summary": {
                "total_investment": round(total_investment, 2),
                "current_value": round(total_value, 2),
                "total_pnl": round(total_value - total_investment, 2),
                "return_percent": round(((total_value - total_investment) / total_investment * 100), 2)
            },
            "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }

    except Exception as e:
        return {"error": f"Portfolio analysis failed: {str(e)}"}


# Backward compatibility functions
def analyze_with_ml(symbol):
    """Original ML analysis function for backward compatibility"""
    return enhanced_analyze_with_ml(symbol)


def analyze_with_gemini(symbol):
    """Original Gemini analysis function for backward compatibility"""
    result = interactive_gemini_analysis(symbol)
    if "error" in result:
        return result
    return {
        "ml_summary": result["ml_context"],
        "gemini_text": result["analysis"]
    }