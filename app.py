from flask import Flask, render_template, request, jsonify
import random
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

API_KEY = "P397VJ8HS1072MSB"

# ── Sector → Alpha Vantage BSE symbol mapping ──────────────────────────────
SECTOR_SYMBOLS = {
    "it":          "INFY.NS",
    "banking":     "HDFCBANK.NS",
    "pharma":      "SUNPHARMA.NS",
    "auto":        "TATAMOTORS.NS",
    "energy":      "RELIANCE.NS",
    "fmcg":        "HUL.NS",
    "realestate":  "DLF.NS",
    "infra":       "LT.NS",
    "metals":      "TATASTEEL.NS",
}

# Fallback price ranges (₹) when API fails
FALLBACK_RANGES = {
    "it":         (1400, 1800),
    "banking":    (1500, 1800),
    "pharma":     (900,  1300),
    "auto":       (800,  1100),
    "energy":     (1200, 1600),
    "fmcg":       (2200, 2600),
    "realestate": (600,  900),
    "infra":      (3000, 3600),
    "metals":     (900,  1300),
}

# Risk labels per sector
SECTOR_RISK = {
    "it":         "High",
    "banking":    "Medium-High",
    "pharma":     "Low-Medium",
    "auto":       "High",
    "energy":     "Medium",
    "fmcg":       "Low",
    "realestate": "High",
    "infra":      "Medium",
    "metals":     "High",
}



def get_real_market_data(sectors):
    """Fetch real live NSE prices using yfinance. Falls back to range if unavailable."""
    result = {}
    for sec in sectors:
        ticker = SECTOR_SYMBOLS.get(sec)
        if not ticker:
            continue
        try:
            data = yf.Ticker(ticker)
            info = data.fast_info          # lightweight — single network call
            price = info.last_price        # real-time / last traded price
            if price and price > 0:
                result[sec] = round(float(price), 2)
            else:
                raise ValueError("No price")
        except Exception:
            # Fallback: try downloading last 5 days and take most recent close
            try:
                hist = yf.download(ticker, period="5d", progress=False)
                if not hist.empty:
                    result[sec] = round(float(hist["Close"].iloc[-1]), 2)
                else:
                    raise ValueError("Empty history")
            except Exception:
                fb = FALLBACK_RANGES.get(sec, (500, 2000))
                result[sec] = round(random.uniform(*fb), 2)
    return result


def calculate_portfolio_value(allocations, market_data):
    """Weighted average of sector prices by allocation percentages."""
    total_alloc = sum(allocations.values())
    if total_alloc == 0:
        return 0.0
    total_value = sum(
        (alloc / total_alloc) * market_data.get(sec, 0)
        for sec, alloc in allocations.items()
    )
    return round(total_value, 2)


def simulate_portfolio(allocations, scenario):
    years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]

    sectors = list(allocations.keys())
    predictions = predict_returns_ml(sectors)

    avg_return = sum(predictions.values()) / len(predictions)

    no_intervention = []
    ai_recommendation = []

    value_no = 100
    value_ai = 100

    for i in range(5):
        # Without AI (random-ish)
        value_no *= (1 + np.random.uniform(-0.1, 0.08))

        # With AI (based on prediction)
        value_ai *= (1 + (avg_return / 100))

        no_intervention.append(round(value_no, 2))
        ai_recommendation.append(round(value_ai, 2))

    return years, no_intervention, ai_recommendation


def get_risk(allocations):
    """Aggregate risk from sector weights."""
    risk_score = 0
    weight_map = {"High": 3, "Medium-High": 2.5, "Medium": 2, "Low-Medium": 1.5, "Low": 1}
    total = sum(allocations.values()) or 1

    for sec, alloc in allocations.items():
        label = SECTOR_RISK.get(sec, "Medium")
        risk_score += (alloc / total) * weight_map.get(label, 2)

    if risk_score >= 2.5:
        return "High Risk"
    elif risk_score >= 1.8:
        return "Medium Risk"
    else:
        return "Low Risk"


def generate_explanation(allocations, market_data):
    if not allocations:
        return "No sectors selected."

    top_sec = max(allocations, key=allocations.get)
    top_pct = allocations[top_sec]
    risk = get_risk(allocations)

    lines = [
        f"Highest allocation: {top_sec.upper()} at {top_pct}%.",
        f"Overall portfolio risk: {risk}.",
    ]

    if top_pct > 50:
        lines.append(f"Concentration in {top_sec} may increase volatility.")
        lines.append("Consider spreading across 3–4 sectors for balance.")
    elif len(allocations) >= 4:
        lines.append("Well-diversified across multiple sectors.")
        lines.append("This reduces sector-specific risk effectively.")
    else:
        lines.append("Moderate diversification. Adding 1–2 sectors could help.")

    return "\n".join(lines)


def get_opportunity(allocations, scenario):
    sectors = list(allocations.keys())
    if scenario == "crash":
        defensive = [s for s in sectors if s in ("pharma", "fmcg", "energy")]
        if defensive:
            return f"Opportunity: {', '.join(s.upper() for s in defensive)} tend to be defensive in downturns."
        return "Opportunity: Consider adding defensive sectors (Pharma, FMCG) to cushion a crash."
    elif scenario == "growth":
        growth = [s for s in sectors if s in ("it", "auto", "realestate", "infra")]
        if growth:
            return f"Opportunity: {', '.join(s.upper() for s in growth)} can outperform in bull runs."
        return "Opportunity: Growth sectors (IT, Auto, Infra) can maximise bull-market returns."
    else:
        return "Opportunity: Diversified portfolios capture steady compounding across cycles."


def get_ai_recommendation(allocations, scenario, market_data):
    sectors = list(allocations.keys())

    predictions = predict_returns_ml(sectors)

    # Sort sectors by predicted return
    sorted_sectors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    total_score = sum(max(p, 0) for _, p in sorted_sectors) or 1

    recommendation = {}
    for sec, score in sorted_sectors:
        weight = max(score, 0) / total_score * 100
        recommendation[sec] = round(weight)

    # Fix total = 100
    diff = 100 - sum(recommendation.values())
    if diff != 0:
        recommendation[sorted_sectors[0][0]] += diff

    return recommendation

def predict_returns_ml(sectors):
    predictions = {}

    for sec in sectors:
        symbol = SECTOR_SYMBOLS.get(sec)

        try:
            # Convert BSE → NSE for yfinance
            ticker = symbol.replace(".BSE", ".NS")

            data = yf.download(ticker, period="2y")

            if data.empty:
                predictions[sec] = 5
                continue

            data["Return"] = data["Close"].pct_change()
            data = data.dropna()

            # Features = previous return
            X = data["Return"].shift(1).dropna().values.reshape(-1, 1)
            y = data["Return"][1:].values

            model = LinearRegression()
            model.fit(X, y)

            # Predict next return
            pred = model.predict([[data["Return"].iloc[-1]]])[0]

            predictions[sec] = float(pred * 100)

        except Exception as e:
            predictions[sec] = 5  # fallback

    return predictions

@app.route("/")
def home():
    return render_template("select.html")


@app.route("/dashboard")
def dashboard():
    sectors = request.args.get("sectors", "")
    return render_template("index.html", sectors=sectors)

@app.route("/live-data", methods=["POST"])
def live_data():
    data = request.json
    sectors = data.get("sectors", ["it", "banking", "pharma"])

    market_data = get_real_market_data(sectors)

    return jsonify({
        "real_data": market_data
    })

@app.route("/compare")
def compare():
    sectors = request.args.get("sectors", "")
    return render_template("compare.html", sectors=sectors)

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json
    sector_keys = data.get("sectors", ["it", "banking", "pharma"])
    allocations = {k: int(data.get(k, 0)) for k in sector_keys}
    scenario = data.get("scenario", "normal")

    market_data = get_real_market_data(list(allocations.keys()))

    years, no_intervention, ai_recommendation = simulate_portfolio(allocations, scenario)
    explanation = generate_explanation(allocations, market_data)
    risk = get_risk(allocations)
    opportunity = get_opportunity(allocations, scenario)
    portfolio_value = calculate_portfolio_value(allocations, market_data)
    recommendation = get_ai_recommendation(allocations, scenario, market_data)
    return jsonify({
        "years": years,
        "no_intervention": no_intervention,
        "ai_recommendation": ai_recommendation,
        "explanation": explanation,
        "risk": risk,
        "opportunity": opportunity,
        "recommendation": recommendation,
        "portfolio_value": portfolio_value,
        "real_data": market_data,
    })


if __name__ == "__main__":
    app.run(debug=True)