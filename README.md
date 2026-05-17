# NiftyMind 📈

An AI-powered Indian stock market portfolio simulator built with Flask and Machine Learning.

## 🌐 Live Demo
👉 [https://niftymind.onrender.com](https://niftymind.onrender.com)

## ✨ Features
- 📊 Live NSE market data using yFinance
- 🤖 AI portfolio recommendations using Linear Regression
- ⚡ Shock events simulation (war, recession, rate hike, pandemic etc.)
- ⚖️ Portfolio A vs B comparison
- 📈 5-year trajectory simulation
- 9 sectors: IT, Banking, Pharma, Auto, Energy, FMCG, Real Estate, Infra, Metals
  
## 🛠️ Tech Stack
- Python, Flask
- yFinance, scikit-learn, NumPy, Pandas
- Chart.js
- Deployed on Render

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## 📁 Project Structure
```
NiftyMind/
├── app.py
├── requirements.txt
├── Procfile
└── templates/
    ├── select.html
    ├── index.html
    └── compare.html
```
