import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd

model = LogisticRegression()
model.fit([[0], [1]], [0, 1])

with mlflow.start_run():
    # test both
    mlflow.sklearn.log_model(model, artifact_path="model_artifact")
    mlflow.sklearn.log_model(model, name="model_name")
