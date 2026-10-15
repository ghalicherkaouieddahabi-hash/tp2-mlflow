
import os
import json
import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp2-registre")

os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns=["fraude"])
y = df["fraude"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

for n_estimators, max_depth in [(200, 8), (300, 12), (500, 16)]:
    with mlflow.start_run(
        run_name=f"rf-{n_estimators}-{max_depth}"
    ):
        modele = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        modele.fit(X_train, y_train)

        pred = modele.predict(X_test)
        proba = modele.predict_proba(X_test)[:, 1]

        metriques = {
            "f1": f1_score(y_test, pred, zero_division=0),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        })
        mlflow.log_metrics(metriques)
        mlflow.set_tag("auteur", "GHALI CHERKAOUI")
        mlflow.set_tag("jeu_de_donnees", "transactions v1")

        ConfusionMatrixDisplay.from_predictions(y_test, pred)
        plt.savefig("reports/confusion.png", dpi=110, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("reports/confusion.png")

       
mlflow.sklearn.log_model(
    sk_model=modele,
    name="modele",
    input_example=X_test.head(3),
    registered_model_name="fraude",
    serialization_format="pickle",
)


print(f"F1 = {metriques['f1']:.4f}")
