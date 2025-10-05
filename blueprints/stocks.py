# blueprints/stocks.py
from flask import Blueprint, render_template, request, session, redirect, url_for, flash
import yfinance as yf
from models import get_portfolio, upsert_portfolio
from ai_agent import analyze_with_ml, analyze_with_gemini, gemini_client
import pandas as pd
from config import Config
import requests

stocks = Blueprint("stocks", __name__, url_prefix="")

def fetch_finance_news():
    """
    Fetch top finance/stock news using NewsAPI including images.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "finance OR stock OR market OR investment",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": Config.NEWS_API_KEY
    }
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            # Ensure each article has an image URL
            for a in articles:
                if not a.get("urlToImage"):
                    a["urlToImage"] = "/static/default_news.png"  # fallback image
            return articles
    except Exception as e:
        print("News fetch error:", e)
    return []

@stocks.route("/")
def dashboard():
    news = fetch_finance_news()
    return render_template("index.html", news=news)

@stocks.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully", "success")
    return redirect(url_for("stocks.dashboard"))

@stocks.route("/stocks")
def list_stocks():
    q = (request.args.get("q") or "AAPL").upper()
    news = fetch_finance_news()  # finance news on stocks page
    try:
        ticker = yf.Ticker(q)
        hist = ticker.history(period="10d")
        current = None
        try:
            current = ticker.info.get("regularMarketPrice")
        except Exception:
            if not hist.empty:
                current = float(hist["Close"].iloc[-1])
        chart_data = hist[["Close"]].reset_index().to_dict(orient="records") if not hist.empty else []
    except Exception as e:
        flash("Error fetching stock: " + str(e), "danger")
        hist = pd.DataFrame()
        current = None
        chart_data = []

    ml_analysis = analyze_with_ml(q)
    gemini_analysis = None
    if Config.AI_PROVIDER.lower() == "gemini":
        try:
            gem_result = analyze_with_gemini(q)
            if isinstance(gem_result, dict) and "error" in gem_result:
                gemini_analysis = gem_result["error"]
            else:
                gemini_analysis = gem_result
        except Exception as e:
            gemini_analysis = f"Gemini error: {str(e)}"

    if isinstance(gemini_analysis, dict):
        gemini_pretty = []
        for key, value in gemini_analysis.items():
            gemini_pretty.append({"key": key.replace("_", " ").title(), "value": value})
    else:
        gemini_pretty = gemini_analysis  # string or error

    return render_template(
        "stocks.html",
        symbol=q,
        price=current,
        chart=chart_data,
        ml=ml_analysis,
        gemini=gemini_pretty,
        news=news
    )

@stocks.route("/portfolio")
def view_portfolio():
    if "user_id" not in session:
        flash("Login to view portfolio", "warning")
        return redirect(url_for("auth.login"))

    pf = get_portfolio(session["user_id"]) or {"user_id": session["user_id"], "positions": []}
    positions = pf.get("positions", [])
    for pos in positions:
        try:
            t = yf.Ticker(pos["symbol"])
            h = t.history(period="1d")
            try:
                pos["current_price"] = t.info.get("regularMarketPrice")
            except Exception:
                pos["current_price"] = float(h["Close"].iloc[-1]) if not h.empty else None
        except Exception:
            pos["current_price"] = None

    return render_template("portfolio.html", positions=positions)

@stocks.route("/portfolio/add", methods=["POST"])
def add_to_portfolio():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    symbol = (request.form.get("symbol") or "").upper()
    if not symbol:
        flash("Symbol required", "danger")
        return redirect(url_for("stocks.view_portfolio"))

    try:
        qty = float(request.form.get("qty") or 0)
        avg_price = float(request.form.get("avg_price") or 0)
    except ValueError:
        flash("Quantity and avg_price must be numbers", "danger")
        return redirect(url_for("stocks.view_portfolio"))

    doc = get_portfolio(session["user_id"]) or {"user_id": session["user_id"], "positions": []}
    positions = doc.get("positions", [])
    found = False
    for p in positions:
        if p["symbol"] == symbol:
            total_qty = p["qty"] + qty
            if total_qty > 0:
                p["avg_price"] = ((p["avg_price"] * p["qty"]) + (avg_price * qty)) / total_qty
            p["qty"] = total_qty
            found = True
            break

    if not found:
        positions.append({"symbol": symbol, "qty": qty, "avg_price": avg_price})

    upsert_portfolio(session["user_id"], {"user_id": session["user_id"], "positions": positions})
    flash("Position added/updated", "success")
    return redirect(url_for("stocks.view_portfolio"))

@stocks.route("/analyze", methods=["GET","POST"])
def analyze():
    result = None
    symbol = None

    if request.method == "POST":
        symbol = (request.form.get("symbol") or "").upper()
        if not symbol:
            flash("Symbol required", "danger")
            return redirect(url_for("stocks.analyze"))

        ml = analyze_with_ml(symbol)
        result = {"ml": ml}

        if Config.AI_PROVIDER.lower() == "gemini":
            try:
                gem = analyze_with_gemini(symbol)
                if isinstance(gem, dict) and "error" in gem:
                    result["gemini"] = gem["error"]
                else:
                    result["gemini"] = gem
            except Exception as e:
                result["gemini"] = f"Gemini error: {str(e)}"

    return render_template("analysis.html", result=result, symbol=symbol)

@stocks.route("/analyze/ask", methods=["POST"])
def ask_gemini():
    symbol = (request.form.get("symbol") or "").upper()
    question = request.form.get("question")

    gemini_reply = None
    if Config.AI_PROVIDER.lower() == "gemini" and question:
        try:
            ml_summary = analyze_with_ml(symbol)
            prompt = f"""
You are a helpful stock analyst. Based on the following stock summary, answer the user's question.
Stock Summary:
- Latest Price: {ml_summary['latest_price']}
- Trend: {ml_summary['trend']}
- Volatility (annual): {ml_summary['volatility_annual']}
- MA20: {ml_summary['ma20']}
- MA50: {ml_summary['ma50']}
- Signal: {ml_summary['signal']}
User question: {question}
"""
            response = gemini_client.chat.completions.create(
                model="gemini-2.5-flash-lite",
                messages=[
                    {"role": "system", "content": "You are a helpful stock analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            gemini_reply = response.choices[0].message.content
        except Exception as e:
            gemini_reply = f"Error asking Gemini: {e}"

    ml = analyze_with_ml(symbol)
    gemini_data = analyze_with_gemini(symbol) if Config.AI_PROVIDER.lower() == "gemini" else None

    return render_template(
        "analysis.html",
        symbol=symbol,
        result={"ml": ml, "gemini": gemini_data},
        gemini_reply=gemini_reply
    )
