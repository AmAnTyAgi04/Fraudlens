"""FraudLens backend.

Trains a class-weighted Logistic Regression on synthetic card transactions at startup,
serves a JSON API under /api, and serves the dashboard from ../frontend.
To use real data, replace the training block with creditcard.csv from Kaggle.
"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["Amount", "Night-time payment", "Distance from home",
            "Transactions in 24h", "Online payment", "New merchant"]


class Txn(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    distance_km: float = Field(ge=0)
    tx_last_24h: int = Field(ge=0)
    online: int = Field(ge=0, le=1)
    new_merchant: int = Field(ge=0, le=1)


class Batch(BaseModel):
    rows: list[Txn] = Field(max_length=5000)


def vec(t):
    return [np.log1p(t["amount"]), float(t["hour"] <= 5), t["distance_km"],
            t["tx_last_24h"], t["online"], t["new_merchant"]]


def make_row(rng, fraud):
    if fraud:
        hour = int(rng.integers(0, 6)) if rng.random() < .55 else int(rng.integers(0, 24))
        return dict(amount=round(float(rng.lognormal(9.2, .8))), hour=hour,
                    distance_km=round(float(rng.exponential(150))), tx_last_24h=int(rng.integers(5, 14)),
                    online=int(rng.random() < .85), new_merchant=int(rng.random() < .8))
    return dict(amount=round(float(rng.lognormal(7.4, .9))), hour=int(rng.integers(8, 23)),
                distance_km=round(abs(float(rng.normal(0, 20)))), tx_last_24h=int(rng.integers(0, 5)),
                online=int(rng.random() < .35), new_merchant=int(rng.random() < .2))


# ---- training ----
rng = np.random.default_rng(7)
rows, labels = [], []
for _ in range(6000):
    f = rng.random() < .04
    rows.append(make_row(rng, f))
    labels.append(int(f))
X = np.array([vec(r) for r in rows])
y = np.array(labels)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=7, stratify=y)
scaler = StandardScaler().fit(Xtr)
model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(Xtr), ytr)
ptest = model.predict_proba(scaler.transform(Xte))[:, 1]
pred = (ptest >= .5).astype(int)
fpr, tpr, _ = roc_curve(yte, ptest)
idx = np.linspace(0, len(fpr) - 1, 40).astype(int)
by_hour = [0] * 24
for r, l in zip(rows, labels):
    if l:
        by_hour[r["hour"]] += 1
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
METRICS = {
    "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
    "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "auc": roc_auc_score(yte, ptest),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "roc": [[float(fpr[i]), float(tpr[i])] for i in idx],
    "weights": [{"name": n, "weight": float(w)} for n, w in zip(FEATURES, model.coef_[0])],
    "fraud_by_hour": by_hour, "train_size": len(Xtr), "test_size": len(Xte),
}


def decide(p):
    return "approve" if p < .5 else "review" if p < .8 else "block"


def score(t):
    x = scaler.transform([vec(t)])
    p = float(model.predict_proba(x)[0, 1])
    contrib = x[0] * model.coef_[0]
    factors = sorted(({"name": n, "contribution": round(float(c), 2)} for n, c in zip(FEATURES, contrib) if c > .05),
                     key=lambda d: -d["contribution"])[:4]
    return {"fraud_probability": round(p, 4), "decision": decide(p), "factors": factors}


# ---- API ----
app = FastAPI(title="FraudLens API")
sim_rng = np.random.default_rng()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(t: Txn):
    return score(t.dict())


@app.post("/api/batch")
def batch(b: Batch):
    out = [{**r.dict(), **score(r.dict())} for r in b.rows]
    return {"count": len(out), "flagged": sum(o["decision"] != "approve" for o in out), "results": out}


@app.get("/api/simulate")
def simulate():
    t = make_row(sim_rng, sim_rng.random() < .06)
    return {"txn": t, **score(t)}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="frontend")
