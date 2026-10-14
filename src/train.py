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
    # Current script se root directory automatically find karna (chahe src se chale ya root se)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    # MLflow Tracking URI set karna
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MLOps_MultiPipeline_Project")
    
    with mlflow.start_run() as run:
        # --- Task 1: Classification Model (Disease Dataset) ---
        disease_data_path = ROOT_DIR / "data" / "disease_classification.csv"
        df_class = pd.read_csv(disease_data_path)
        X_c = df_class.drop(columns=['target'])
        y_c = df_class['target']
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xc_train, yc_train)
        clf_acc = accuracy_score(yc_test, clf.predict(Xc_test))
        
        mlflow.log_metric("classification_accuracy", clf_acc)
        mlflow.sklearn.log_model(clf, "classification_model")
        
        # --- Task 2: Regression Model (Boston / Housing Dataset) ---
        boston_data_path = ROOT_DIR / "data" / "boston_regression.csv"
        df_reg = pd.read_csv(boston_data_path).sample(n=2000, random_state=42)
        X_r = df_reg.drop(columns=['MedHouseVal'])
        y_r = df_reg['MedHouseVal']
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
        
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(Xr_train, yr_train)
        reg_mse = mean_squared_error(yr_test, reg.predict(Xr_test))
        
        mlflow.log_metric("regression_mse", reg_mse)
        
        # Primary regression model ko MLflow Registry mein log aur register karna
        model_name = "MLOps_Production_Model"
        mlflow.sklearn.log_model(
            sk_model=reg,
            artifact_path="regression_model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        # Pipeline stage ke mutabiq Model Registry Aliases assign karna
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version
        
        if pipeline_stage == "dev":
            client.set_registered_model_alias(model_name, "Challenger", latest_version)
            print(f"Dev Pipeline: Registered model version {latest_version} with alias 'Challenger'")
        elif pipeline_stage == "pre-prod":
            client.set_registered_model_alias(model_name, "Challenger-pre-test", latest_version)
            print(f"Pre-prod Pipeline: Registered model version {latest_version} with alias 'Challenger-pre-test'")
        elif pipeline_stage == "prod":
            client.set_registered_model_alias(model_name, "Champion", latest_version)
            print(f"Prod Pipeline: Promoted model version {latest_version} to 'Champion'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="dev", choices=["dev", "pre-prod", "prod"])
    args = parser.parse_args()
    train_and_register(pipeline_stage=args.stage)