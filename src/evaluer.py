
import os
import json
import joblib
import yaml
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["validation"]

test = pd.read_csv("data/prepare/test.csv")

X = test.drop(columns=["fraude"])
y = test["fraude"]

modele = joblib.load("models/model.pkl")
predictions = modele.predict(X)

metriques = {
    "f1": round(f1_score(y, predictions, zero_division=0), 4),
    "precision": round(precision_score(y, predictions, zero_division=0), 4),
    "recall": round(recall_score(y, predictions, zero_division=0), 4)
}

os.makedirs("reports", exist_ok=True)

with open("reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2)

print("évaluation :", metriques)
