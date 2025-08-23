import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and process data
data = pd.read_csv("data/census.csv")

cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

train, test = train_test_split(data, test_size=0.20, random_state=42)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)

model = train_model(X_train, y_train)

# TODO: implement the first test. Change the function name and input as needed
def test_apply_labels():
    """
    # test that label values are 0 or 1
    """
    # Your code here
    assert set(np.unique(y_train)).issubset({0, 1})


# TODO: implement the second test. Change the function name and input as needed
def test_train_model():
    """
    # test that the trained model is a RandomForestClassifier
    """
    # Your code here
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # test that compute_model_metrics returns floats
    """
    # Your code here
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    for metric in [precision, recall, fbeta]:
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0
