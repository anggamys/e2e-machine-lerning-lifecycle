import warnings
import mlflow
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("pijak-dicoding")

data = pd.read_csv("./twitter-dataset-cleaned/data_clean.csv")
data = data.dropna(subset=["cleaned_text"])

X = data["cleaned_text"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Ensure y_train and y_test are Series (1D) for sklearn and MLflow
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Suppress pickle warning from sklearn
warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution",
)

# Suppress DataConversionWarning from sklearn
warnings.filterwarnings("ignore", category=DataConversionWarning)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs"],
    "class_weight": ["balanced", None],
    "max_iter": [500, 1000],
}

with mlflow.start_run():
    mlflow_sklearn.autolog()

    grid = GridSearchCV(
        LogisticRegression(), param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train_vec, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy_manual", float(acc))
    mlflow.log_params(grid.best_params_)

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    report = classification_report(y_test, y_pred, output_dict=False)
    print("Classification Report:\n", report)

    mlflow.log_text(str(report), "classification_report.txt")
