
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = FastAPI(title="API de détection de fraude")

class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    montant: float
    heure: int
    frequence: int
    score: float

@app.get("/")
def accueil():
    return {"message": "API de détection de fraude opérationnelle"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        model = mlflow.sklearn.load_model("models:/fraude@production")

        donnees = pd.DataFrame([{
            "montant": transaction.montant,
            "heure": transaction.heure,
            "frequence": transaction.frequence,
            "score": transaction.score,
        }])

        prediction = int(model.predict(donnees)[0])

        return {
            "prediction": prediction,
            "resultat": "Fraude" if prediction == 1 else "Transaction normale"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
