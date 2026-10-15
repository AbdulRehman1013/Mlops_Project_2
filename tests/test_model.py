import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

def test_data_existence():
    assert os.path.exists("data/disease_classification.csv")
    assert os.path.exists("data/boston_regression.csv")

def test_dataset_shape():
    df = pd.read_csv("data/disease_classification.csv")
    assert df.shape[0] > 0
    assert df.shape[1] > 0