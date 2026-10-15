
import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["entrainement"]

train = pd.read_csv("data/prepare/train.csv")

X = train.drop(columns=["fraude"])
y = train["fraude"]

modele = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)

modele.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(modele, "models/model.pkl")

print("entraînement : modèle écrit dans models/model.pkl")
