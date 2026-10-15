import os, sys
from pathlib import Path
import mlflow, mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
REG_MODEL = "MLOps_Production_Model"
CLS_MODEL = "MLOps_Classification_Model"
MIN_ACC = float(os.getenv("MIN_ACCURACY", "0.5"))
MIN_R2 = float(os.getenv("MIN_R2", "0.3"))

def main(alias):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    cls = mlflow.sklearn.load_model(f"models:/{CLS_MODEL}@{alias}")
    df = pd.read_csv(ROOT / "data" / "disease_classification.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    acc = accuracy_score(yte, cls.predict(Xte))

    reg = mlflow.sklearn.load_model(f"models:/{REG_MODEL}@{alias}")
    dr = pd.read_csv(ROOT / "data" / "boston_regression.csv")
    dr = dr.sample(n=min(2000, len(dr)), random_state=42)
    Xr, yr = dr.drop(columns=["MedHouseVal"]), dr["MedHouseVal"]
    _, Xrte, _, yrte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    pred = reg.predict(Xrte)
    r2, mse = r2_score(yrte, pred), mean_squared_error(yrte, pred)

    print(f"[{alias}] accuracy={acc:.3f} (min {MIN_ACC}), r2={r2:.3f} (min {MIN_R2}), mse={mse:.3f}")
    if acc < MIN_ACC or r2 < MIN_R2:
        print("Evaluation FAILED")
        sys.exit(1)
    print("Evaluation PASSED")

    if alias == "Challenger-pre-test":
        for name in (CLS_MODEL, REG_MODEL):
            v = client.get_model_version_by_alias(name, alias).version
            client.set_registered_model_alias(name, "Challenger-post-test", v)
            print(f"{name} v{v} -> alias 'Challenger-post-test'")

if __name__ == "__main__":
    main(sys.argv[1])
