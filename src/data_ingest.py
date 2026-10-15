import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

def ingest_data():
    os.makedirs("data", exist_ok=True)
    
    # 1. Classification Dataset (Disease / Breast Cancer)
    cancer = load_breast_cancer()
    df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df_cancer['target'] = cancer.target
    df_cancer.to_csv("data/disease_classification.csv", index=False)
    print("Classification dataset ingested successfully.")
    
    # 2. Regression Dataset (Boston / Housing alternative)
    housing = fetch_california_housing(as_frame=True)
    df_housing = housing.frame
    df_housing.to_csv("data/boston_regression.csv", index=False)
    print("Regression dataset ingested successfully.")

if __name__ == "__main__":
    ingest_data()