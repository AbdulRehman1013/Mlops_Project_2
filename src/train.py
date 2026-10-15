import os, argparse
from pathlib import Path
import mlflow, mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
REG_MODEL = "MLOps_Production_Model"
CLS_MODEL = "MLOps_Classification_Model"
EXP_NAME = "MLOps_MultiPipeline_Project"

# stage -> (alias to load, alias to set)
STAGES = {
    "dev":      (None, "Challenger"),
    "pre-prod": ("Challenger", "Challenger-pre-test"),
    "prod":     ("Challenger-post-test", "Champion"),
}

def setup_mlflow():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    art_root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "/var/lib/jenkins/mlflow/artifacts"))
    art_root.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(EXP_NAME) is None:
        mlflow.create_experiment(EXP_NAME, artifact_location=art_root.as_uri())
    mlflow.set_experiment(EXP_NAME)

def train_models():
    df = pd.read_csv(ROOT / "data" / "disease_classification.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(Xtr, ytr)
    rf_c = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    lr_acc = accuracy_score(yte, lr.predict(Xte))
    rf_acc = accuracy_score(yte, rf_c.predict(Xte))
    mlflow.log_metric("logreg_accuracy", lr_acc)
    mlflow.log_metric("rf_classifier_accuracy", rf_acc)
    cls = lr if lr_acc >= rf_acc else rf_c
    mlflow.log_metric("classification_accuracy", max(lr_acc, rf_acc))

    dr = pd.read_csv(ROOT / "data" / "boston_regression.csv")
    dr = dr.sample(n=min(2000, len(dr)), random_state=42)
    Xr, yr = dr.drop(columns=["MedHouseVal"]), dr["MedHouseVal"]
    Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(Xrtr, yrtr)
    mlflow.log_metric("regression_mse", mean_squared_error(yrte, reg.predict(Xrte)))
    return {CLS_MODEL: (cls, "classification_model"), REG_MODEL: (reg, "regression_model")}

def train_and_register(stage):
    setup_mlflow()
    client = MlflowClient()
    load_alias, set_alias = STAGES[stage]

    with mlflow.start_run():
        mlflow.set_tag("pipeline_stage", stage)
        if stage == "dev":
            models = train_models()
        else:
            models = {}
            for name, art in [(CLS_MODEL, "classification_model"), (REG_MODEL, "regression_model")]:
                print(f"{stage}: loading {name}@{load_alias}")
                models[name] = (mlflow.sklearn.load_model(f"models:/{name}@{load_alias}"), art)

        for name, (model, art) in models.items():
            info = mlflow.sklearn.log_model(
                model, name=art, registered_model_name=name, pip_requirements=[]
            )
            version = info.registered_model_version
            client.set_registered_model_alias(name, set_alias, version)
            print(f"{name} v{version} -> alias '{set_alias}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="dev", choices=list(STAGES))
    train_and_register(p.parse_args().stage)
