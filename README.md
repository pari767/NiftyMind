# 🧠 NiftyMind AI
### Intelligent Portfolio Simulator for Indian Equity Markets

> Simulate, compare, and optimise NSE sector portfolios in seconds — powered by live market data and a machine-learning return predictor.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![yfinance](https://img.shields.io/badge/yfinance-NSE%20Live-green)](https://github.com/ranaroussi/yfinance)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org)

---

## 🎯 What It Does

NiftyMind lets you:
- **Pick sectors** from 9 NSE categories (IT, Banking, Pharma, Auto, Energy, FMCG, Real Estate, Infra, Metals)
- **Allocate capital** across sectors using interactive sliders
- **Simulate 5-year growth** — comparing a naive random-walk vs. an AI-optimised path
- **Compare two portfolios (A vs B)** side-by-side with a scenario selector (Normal / Bull / Crash)
- **Get an AI reallocation recommendation** based on a linear regression model trained on 2 years of live NSE data
- **Understand your risk** — Low / Medium / High scoring based on weighted sector volatility

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip
- Internet connection (for live NSE data via yfinance)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/niftymind-ai.git
cd niftymind-ai
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Mac/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
flask
yfinance
numpy
pandas
scikit-learn
requests
```

### 4. Run the App
```bash
python app.py
```

The app starts at **http://127.0.0.1:5000**

### 5. Open in Browser
Navigate to `http://127.0.0.1:5000` and start simulating!

---

## 📁 Project Structure

```
niftymind-ai/
├── app.py              # Flask backend — all ML, data, and routing logic
├── templates/
│   └── index.html      # Full-stack UI (embedded in app.py as a string)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

> **Note:** The HTML/CSS/JS frontend is rendered via Flask's `render_template_string` and is contained within `app.py` in the current version. A future refactor will separate it into `templates/index.html`.

---

## 🧩 Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main UI |
| `/simulate` | POST | Runs simulation for a single portfolio |
| `/compare` | POST | Compares Portfolio A vs Portfolio B |

### Example `/simulate` Payload
```json
{
  "sectors": ["it", "banking", "pharma"],
  "it": 50,
  "banking": 30,
  "pharma": 20,
  "scenario": "growth"
}
```

### Example Response
```json
{
  "portfolio_value": 1452.30,
  "risk": "High Risk",
  "years": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
  "no_intervention": [98.2, 94.1, 101.3, 97.8, 103.2],
  "ai_recommendation": [107.5, 116.3, 126.0, 136.8, 148.5],
  "explanation": "Highest allocation: IT at 50%...",
  "opportunity": "Opportunity: IT can outperform in bull runs.",
  "ai_reallocation": {"it": 45, "banking": 35, "pharma": 20}
}
```

---

## 🤖 How the ML Works

1. **Data**: Downloads 2 years of daily OHLCV data per sector from NSE via yfinance
2. **Feature**: Computes daily % returns; creates a 1-day lagged feature (`Return[t-1]`)
3. **Model**: Fits a `scikit-learn LinearRegression` — retrained fresh on every request
4. **Output**: Predicted next-period return % per sector
5. **Allocation**: Sectors are ranked by predicted return; weights assigned proportionally

---

## 🛡️ Resilience & Fallbacks

| Failure | Recovery |
|---------|----------|
| yfinance live price fails | Downloads last 5-day history |
| yfinance download fails | Uses static fallback price range |
| ML training fails | Returns 5% default prediction |
| Zero allocation | Returns ₹0 safely (no crash) |

---

## 🌐 Deployment

**Render / Railway (Free Tier):**
```bash
# Procfile
web: python app.py
```
Set environment variable: `PORT=10000`

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## ⚠️ Disclaimer

NiftyMind is a **simulation and educational tool**. It does not constitute financial advice. All projections are based on historical data and linear extrapolation. Past performance does not guarantee future returns. Please consult a SEBI-registered Investment Adviser before making investment decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built for the AI Agents Hackathon 2025 · NiftyMind Team*
