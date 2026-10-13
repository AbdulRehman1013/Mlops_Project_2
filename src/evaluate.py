import sys
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate_model(alias="Challenger"):
    print(f"Evaluating model with alias: {alias}")
    
    # Load model from MLflow registry using its alias
    model_uri = f"models:/MLOps_Production_Model@{alias}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Quick test check
    df_reg = pd.read_csv("data/boston_regression.csv").sample(n=500, random_state=42)
    X = df_reg.drop(columns=['MedHouseVal'])
    y = df_reg['MedHouseVal']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Evaluation MSE for {alias}: {mse}")
    
    # Threshold check
    if mse > 2.0:
        print("Model evaluation FAILED performance threshold.")
        sys.exit(1)
    else:
        print("Model evaluation PASSED successfully.")
        sys.exit(0)

if __name__ == "__main__":
    alias_arg = sys.argv[1] if len(sys.argv) > 1 else "Challenger"
    evaluate_model(alias=alias_arg)