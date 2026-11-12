# FraudLens – AI Based Financial Fraud Detection Using Machine Learning

A full-stack final-year project. A Python backend trains a fraud model and serves a JSON API.
A web dashboard calls that API to screen transactions.

## Features
- Live dashboard: streaming transactions scored in real time, fraud-by-hour chart
- Scanner: score one transaction and see which factors raised the risk
- Batch upload: score many transactions from a CSV
- Model page: accuracy, precision, recall, F1, ROC-AUC, ROC curve, confusion matrix, feature weights
- Decisions: under 50% approve, 50–80% review, over 80% block

## Structure
```
fraudlens/
├── backend/main.py        FastAPI server, model training, API
├── frontend/index.html    Dashboard (served by the backend)
├── requirements.txt       Python dependencies
├── render.yaml            One-click deployment on Render
├── .gitignore
└── README.md
```

## Tech stack
Python, FastAPI, scikit-learn (class-weighted Logistic Regression), NumPy, HTML, CSS, JavaScript, Git, GitHub, Render.

## Run locally
```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```
Open http://localhost:8000 (API docs at http://localhost:8000/docs).

## Put it on GitHub
```
git init
git add .
git commit -m "FraudLens: initial version"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fraudlens.git
git push -u origin main
```
Create the empty repository `fraudlens` on github.com first. If git is blocked on your laptop,
open the repository in the browser, choose **Add file → Upload files**, and drag in everything
except `venv`.

## Deploy as a public website (free)
1. Sign in at render.com with your GitHub account.
2. Choose **New → Blueprint** and select the `fraudlens` repository. Render reads `render.yaml`.
3. Click **Apply**. After the build finishes you get a public link like `https://fraudlens.onrender.com`.

The free plan sleeps after inactivity, so open the link a minute before your presentation.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | Service status |
| GET | /api/metrics | Model metrics and chart data |
| POST | /api/predict | Score one transaction |
| POST | /api/batch | Score many transactions |
| GET | /api/simulate | Random scored transaction for the live feed |

Example body for `/api/predict`:
```
{"amount": 42000, "hour": 3, "distance_km": 380, "tx_last_24h": 11, "online": 1, "new_merchant": 1}
```

## Data note
The model trains on synthetic transactions generated at startup, so the metrics describe this demo.
To use real data, replace the training block in `backend/main.py` with the Kaggle ULB `creditcard.csv`.

## Future work
XGBoost and LightGBM, Isolation Forest for anomaly detection, SHAP explanations,
human-review feedback loop, PostgreSQL storage, UPI and loan-fraud models, bank integration.

## Team
Aman Tyagi, Aryan Choudhary, Yash Malhotra, Abhijeet Kumar. Guide: Mr. Abdul Qadir.
