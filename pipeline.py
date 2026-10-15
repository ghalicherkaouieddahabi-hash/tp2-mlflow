
import sys
import subprocess
import yaml
import mlflow

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp2-pipeline")


def lancer(etape):
    subprocess.run(
        [sys.executable, f"src/{etape}.py"],
        check=True
    )


with mlflow.start_run(run_name="pipeline") as parent:
    with mlflow.start_run(
        run_name="preparer", nested=True
    ):
        lancer("preparer")
        mlflow.log_params(params["preparation"])

    with mlflow.start_run(
        run_name="entrainer", nested=True
    ):
        lancer("entrainer")
        mlflow.log_params(params["entrainement"])

    with mlflow.start_run(
        run_name="evaluer", nested=True
    ):
        lancer("evaluer")

        import json

        with open("reports/metrics.json", "r", encoding="utf-8") as f:
            metriques = json.load(f)

        mlflow.log_metrics(metriques)

        seuil = params["validation"]["seuil_f1"]
        mlflow.log_param("seuil_f1", seuil)

        if metriques["f1"] >= seuil:
            mlflow.set_tag("statut", "accepte")
            print(
                f"Modèle accepté : F1={metriques['f1']} >= {seuil}"
            )
        else:
            mlflow.set_tag("statut", "refuse")
            print(
                f"Modèle refusé : F1={metriques['f1']} < {seuil}"
            )
            raise SystemExit(1)
