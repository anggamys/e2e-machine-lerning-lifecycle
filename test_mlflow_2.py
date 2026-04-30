import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("test_exp")
model = LogisticRegression()
model.fit([[0], [1]], [0, 1])

with mlflow.start_run():
    mlflow.sklearn.log_model(model, name="my_cool_model")
