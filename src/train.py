import os
import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

def train_and_register(pipeline_stage="dev"):
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_artifact_uri("file:///var/lib/jenkins/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MLOps_MultiPipeline_Project")
    
    client = mlflow.tracking.MlflowClient()
    model_name = "MLOps_Production_Model"
    
    with mlflow.start_run() as run:
        if pipeline_stage == "pre-prod":
            print("Pre-prod: Loading model with alias 'Challenger'")
            try:
                model_uri = f"models:/{model_name}@Challenger"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                print("Successfully loaded 'Challenger' model for validation.")
            except Exception as e:
                print(f"Warning: Could not load Challenger model, creating fallback: {e}")
                
        elif pipeline_stage == "prod":
            print("Prod: Loading model with alias 'Challenger-post-test'")
            try:
                model_uri = f"models:/{model_name}@Challenger-post-test"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                print("Successfully loaded 'Challenger-post-test' model.")
            except Exception as e:
                print(f"Warning: Could not load Challenger-post-test model: {e}")

        disease_data_path = ROOT_DIR / "data" / "disease_classification.csv"
        df_class = pd.read_csv(disease_data_path)
        X_c = df_class.drop(columns=['target'])
        y_c = df_class['target']
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xc_train, yc_train)
        clf_acc = accuracy_score(yc_test, clf.predict(Xc_test))
        mlflow.log_metric("classification_accuracy", clf_acc)
        mlflow.sklearn.log_model(clf, "classification_model", pip_requirements=[])
        
        boston_data_path = ROOT_DIR / "data" / "boston_regression.csv"
        df_reg = pd.read_csv(boston_data_path).sample(n=2000, random_state=42)
        X_r = df_reg.drop(columns=['MedHouseVal'])
        y_r = df_reg['MedHouseVal']
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
        
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(Xr_train, yr_train)
        reg_mse = mean_squared_error(yr_test, reg.predict(Xr_test))
        mlflow.log_metric("regression_mse", reg_mse)
        
        mlflow.sklearn.log_model(
            sk_model=reg,
            artifact_path="regression_model",
            registered_model_name=model_name,
            pip_requirements=[]
        )
        
        versions = client.get_latest_versions(model_name, stages=["None"])
        latest_version = versions[-1].version if versions else "1"
        
        if pipeline_stage == "dev":
            client.set_registered_model_alias(model_name, "Challenger", latest_version)
            print(f"Dev Pipeline: Registered version {latest_version} with alias 'Challenger'")
        elif pipeline_stage == "pre-prod":
            client.set_registered_model_alias(model_name, "Challenger-pre-test", latest_version)
            print(f"Pre-prod Pipeline: Registered version {latest_version} with alias 'Challenger-pre-test'")
        elif pipeline_stage == "prod":
            client.set_registered_model_alias(model_name, "Champion", latest_version)
            print(f"Prod Pipeline: Promoted version {latest_version} to 'Champion'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="dev", choices=["dev", "pre-prod", "prod"])
    args = parser.parse_args()
    train_and_register(pipeline_stage=args.stage)
