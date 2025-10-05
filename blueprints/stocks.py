from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
import yfinance as yf
from models import get_portfolio, upsert_portfolio
from ai_agent import (
    enhanced_analyze_with_ml,
    interactive_gemini_analysis,
    analyze_portfolio_with_gemini,
    analyze_with_ml,  # for backward compatibility
    analyze_with_gemini  # for backward compatibility
)
import pandas as pd
from config import Config
import requests
import json
from datetime import datetime

stocks = Blueprint("stocks", __name__, url_prefix="")

# Available NIFTY-50 symbols for autocomplete
NIFTY50_SYMBOLS = [
    'RELIANCE', 'TCS', 'HDFC', 'INFY', 'HUL', 'ICICI', 'KOTAK', 'SBIN', 'BHARTI', 'LT',
    'ASIAN', 'HCL', 'MARUTI', 'SBI', 'AXIS', 'SUN', 'TITAN', 'WIPRO', 'ULTRACEM', 'NESTLE',
    'POWERGRID', 'M&M', 'BAJAJ', 'TECHM', 'HDFC', 'BRITANNIA', 'GRASIM', 'JSW', 'HAL', 'ADANIPORTS'
]


def fetch_finance_news():
    """
    Fetch top finance/stock news using NewsAPI including images.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "stock market OR investing OR NSE OR BSE OR Nifty",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 6,
        "apiKey": Config.NEWS_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            # Ensure each article has an image URL
            for a in articles:
                if not a.get("urlToImage"):
                    a["urlToImage"] = "/static/default_news.png"
                # Format date
                if a.get('publishedAt'):
                    a['publishedAt'] = a['publishedAt'][:10]
            return articles
    except Exception as e:
        print("News fetch error:", e)
    return []


def get_stock_data(symbol):
    """Get comprehensive stock data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)

        # Get historical data
        hist = ticker.history(period="1mo")

        # Get current price and info
        info = ticker.info
        current_price = info.get('regularMarketPrice',
                                 info.get('currentPrice',
                                          float(hist['Close'].iloc[-1]) if not hist.empty else 0))

        # Get additional data
        company_name = info.get('longName', symbol)
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 'N/A')

        # Prepare chart data
        chart_data = []
        if not hist.empty:
            chart_data = [{
                'date': date.strftime('%Y-%m-%d'),
                'price': float(price)
            } for date, price in zip(hist.index, hist['Close'])]

        return {
            'symbol': symbol,
            'company_name': company_name,
            'current_price': current_price,
            'sector': sector,
            'market_cap': market_cap,
            'chart_data': chart_data,
            'history': hist
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


@stocks.route("/")
def dashboard():
    news = fetch_finance_news()

    # Get popular stocks for dashboard
    popular_stocks = []
    for symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFC']:
        stock_data = get_stock_data(symbol + '.NS')
        if stock_data:
            popular_stocks.append(stock_data)

    return render_template("index.html", news=news, popular_stocks=popular_stocks)


@stocks.route("/stocks")
def list_stocks():
    q = (request.args.get("q") or "RELIANCE").upper()

    # Add .NS for NSE stocks if not present
    if not q.endswith('.NS') and q in NIFTY50_SYMBOLS:
        q += '.NS'

    news = fetch_finance_news()
    stock_data = get_stock_data(q)

    if not stock_data:
        flash(f"Error fetching data for {q}. Please check the symbol.", "danger")
        return render_template("stocks.html", symbol=q, news=news)

    # Enhanced ML Analysis
    ml_analysis = enhanced_analyze_with_ml(q.replace('.NS', ''))

    # Gemini Analysis
    gemini_analysis = None
    if Config.AI_PROVIDER.lower() == "gemini":
        try:
            gemini_result = interactive_gemini_analysis(q.replace('.NS', ''))
            gemini_analysis = gemini_result
        except Exception as e:
            gemini_analysis = {"error": f"Gemini error: {str(e)}"}

    return render_template(
        "stocks.html",
        symbol=q,
        stock_data=stock_data,
        ml=ml_analysis,
        gemini=gemini_analysis,
        news=news,
        nifty_symbols=NIFTY50_SYMBOLS
    )


@stocks.route("/api/ask-gemini", methods=["POST"])
def api_ask_gemini():
    """API endpoint for interactive Gemini questions"""
    if "user_id" not in session:
        return jsonify({"error": "Please login to use this feature"}), 401

    data = request.get_json()
    symbol = data.get("symbol", "").upper().replace('.NS', '')
    question = data.get("question", "")

    if not symbol or not question:
        return jsonify({"error": "Symbol and question are required"}), 400

    try:
        result = interactive_gemini_analysis(symbol, question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stocks.route("/api/stock-analysis", methods=["POST"])
def api_stock_analysis():
    """API endpoint for quick stock analysis"""
    data = request.get_json()
    symbol = data.get("symbol", "").upper().replace('.NS', '')

    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    try:
        # Quick ML analysis
        ml_analysis = enhanced_analyze_with_ml(symbol)

        # Quick Gemini insight if available
        gemini_insight = None
        if Config.AI_PROVIDER.lower() == "gemini":
            gemini_result = interactive_gemini_analysis(symbol, "Give a brief 2-sentence overview")
            if "error" not in gemini_result:
                gemini_insight = gemini_result.get("analysis", "")

        return jsonify({
            "symbol": symbol,
            "ml_analysis": ml_analysis,
            "gemini_insight": gemini_insight
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stocks.route("/portfolio")
def view_portfolio():
    if "user_id" not in session:
        flash("Please login to view your portfolio", "warning")
        return redirect(url_for("auth.login"))

    portfolio = get_portfolio(session["user_id"]) or {"user_id": session["user_id"], "positions": []}
    positions = portfolio.get("positions", [])

    total_investment = 0
    total_current_value = 0

    # Enhance portfolio data with current prices and calculations
    for position in positions:
        try:
            symbol_with_ns = position["symbol"] + '.NS'
            ticker = yf.Ticker(symbol_with_ns)
            hist = ticker.history(period="1d")

            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else position["avg_price"]
            position["current_price"] = current_price
            position["current_value"] = position["qty"] * current_price
            position["investment"] = position["qty"] * position["avg_price"]
            position["pnl"] = position["current_value"] - position["investment"]
            position["pnl_percent"] = (position["pnl"] / position["investment"]) * 100 if position[
                                                                                              "investment"] > 0 else 0

            total_investment += position["investment"]
            total_current_value += position["current_value"]

        except Exception as e:
            print(f"Error updating position {position['symbol']}: {e}")
            position["current_price"] = position["avg_price"]
            position["current_value"] = position["qty"] * position["avg_price"]
            position["investment"] = position["qty"] * position["avg_price"]
            position["pnl"] = 0
            position["pnl_percent"] = 0

    total_pnl = total_current_value - total_investment
    total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

    return render_template(
        "portfolio.html",
        positions=positions,
        total_investment=round(total_investment, 2),
        total_current_value=round(total_current_value, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_percent=round(total_pnl_percent, 2)
    )


@stocks.route("/portfolio/add", methods=["POST"])
def add_to_portfolio():
    if "user_id" not in session:
        flash("Please login to manage portfolio", "warning")
        return redirect(url_for("auth.login"))

    symbol = (request.form.get("symbol") or "").upper().replace('.NS', '')
    if not symbol:
        flash("Stock symbol is required", "danger")
        return redirect(url_for("stocks.view_portfolio"))

    try:
        qty = float(request.form.get("qty") or 0)
        avg_price = float(request.form.get("avg_price") or 0)

        if qty <= 0 or avg_price <= 0:
            flash("Quantity and average price must be positive numbers", "danger")
            return redirect(url_for("stocks.view_portfolio"))

    except ValueError:
        flash("Quantity and average price must be valid numbers", "danger")
        return redirect(url_for("stocks.view_portfolio"))

    portfolio = get_portfolio(session["user_id"]) or {"user_id": session["user_id"], "positions": []}
    positions = portfolio.get("positions", [])

    # Check if position already exists
    found = False
    for position in positions:
        if position["symbol"] == symbol:
            total_qty = position["qty"] + qty
            if total_qty > 0:
                # Calculate new average price
                position["avg_price"] = ((position["avg_price"] * position["qty"]) + (avg_price * qty)) / total_qty
            position["qty"] = total_qty
            found = True
            break

    if not found:
        positions.append({
            "symbol": symbol,
            "qty": qty,
            "avg_price": avg_price,
            "added_date": datetime.now().strftime("%Y-%m-%d")
        })

    upsert_portfolio(session["user_id"], {"user_id": session["user_id"], "positions": positions})
    flash(f"Position for {symbol} added/updated successfully!", "success")
    return redirect(url_for("stocks.view_portfolio"))


@stocks.route("/portfolio/remove/<symbol>", methods=["POST"])
def remove_from_portfolio(symbol):
    if "user_id" not in session:
        flash("Please login to manage portfolio", "warning")
        return redirect(url_for("auth.login"))

    portfolio = get_portfolio(session["user_id"])
    if portfolio:
        positions = portfolio.get("positions", [])
        positions = [p for p in positions if p["symbol"] != symbol]

        upsert_portfolio(session["user_id"], {"user_id": session["user_id"], "positions": positions})
        flash(f"Position for {symbol} removed successfully!", "success")

    return redirect(url_for("stocks.view_portfolio"))


@stocks.route("/portfolio/analyze")
def analyze_portfolio():
    """AI-powered portfolio analysis"""
    if "user_id" not in session:
        flash("Please login to analyze portfolio", "warning")
        return redirect(url_for("auth.login"))

    portfolio = get_portfolio(session["user_id"]) or {"positions": []}

    # Enhance portfolio data
    positions = portfolio.get("positions", [])
    for position in positions:
        try:
            symbol_with_ns = position["symbol"] + '.NS'
            ticker = yf.Ticker(symbol_with_ns)
            hist = ticker.history(period="1d")
            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else position["avg_price"]
            position["current_price"] = current_price
            position["current_value"] = position["qty"] * current_price
        except Exception:
            position["current_price"] = position["avg_price"]
            position["current_value"] = position["qty"] * position["avg_price"]

    # AI Portfolio Analysis
    portfolio_analysis = None
    if Config.AI_PROVIDER.lower() == "gemini":
        portfolio_analysis = analyze_portfolio_with_gemini(portfolio)

    total_investment = sum(p["qty"] * p["avg_price"] for p in positions)
    total_value = sum(p["current_value"] for p in positions)
    total_pnl = total_value - total_investment

    return render_template(
        "portfolio_analysis.html",
        positions=positions,
        total_investment=round(total_investment, 2),
        total_value=round(total_value, 2),
        total_pnl=round(total_pnl, 2),
        portfolio_analysis=portfolio_analysis
    )


@stocks.route("/analyze", methods=["GET", "POST"])
def analyze():
    """Enhanced analysis page with interactive features"""
    result = None
    symbol = None
    gemini_reply = None

    if request.method == "POST":
        symbol = (request.form.get("symbol") or "").upper().replace('.NS', '')
        question = request.form.get("question")

        if not symbol:
            flash("Stock symbol is required", "danger")
            return redirect(url_for("stocks.analyze"))

        # ML Analysis
        ml = enhanced_analyze_with_ml(symbol)
        result = {"ml": ml}

        # Gemini Analysis
        if Config.AI_PROVIDER.lower() == "gemini":
            try:
                if question:
                    # Interactive question
                    gemini_result = interactive_gemini_analysis(symbol, question)
                    gemini_reply = gemini_result
                else:
                    # Standard analysis
                    gemini_result = interactive_gemini_analysis(symbol)
                    result["gemini"] = gemini_result
            except Exception as e:
                result["gemini"] = {"error": f"Gemini error: {str(e)}"}

    return render_template(
        "analysis.html",
        result=result,
        symbol=symbol,
        gemini_reply=gemini_reply,
        nifty_symbols=NIFTY50_SYMBOLS
    )


@stocks.route("/market-overview")
def market_overview():
    """Market overview with multiple stocks"""
    news = fetch_finance_news()

    # Get data for major NIFTY-50 stocks
    major_stocks = []
    for symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 'SBIN']:
        stock_data = get_stock_data(symbol + '.NS')
        if stock_data:
            # Add quick analysis
            analysis = enhanced_analyze_with_ml(symbol)
            stock_data['analysis'] = analysis
            major_stocks.append(stock_data)

    return render_template(
        "market_overview.html",
        major_stocks=major_stocks,
        news=news
    )


@stocks.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully", "success")
    return redirect(url_for("stocks.dashboard"))