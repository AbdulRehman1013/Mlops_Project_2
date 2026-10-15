import os, sys, shutil
from pathlib import Path
import mlflow, mlflow.sklearn

MODELS = ["MLOps_Classification_Model", "MLOps_Production_Model"]

def main(alias, env_name):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    base = Path("/var/lib/jenkins/mlflow/deployed") / env_name
    for name in MODELS:
        model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
        target = base / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(model, str(target), pip_requirements=[],
                                  serialization_format="cloudpickle")
        print(f"Deployed {name}@{alias} -> {target}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
