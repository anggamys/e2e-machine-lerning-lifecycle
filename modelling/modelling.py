import mlflow
import warnings
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("./twitter-dataset-cleaned/data_clean.csv")
data = data.dropna(subset=["cleaned_text"])

X = data["cleaned_text"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert y_train to DataFrame to avoid MLflow warning
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Suppress pickle warning from sklearn
warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution",
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


with mlflow.start_run():
    mlflow_sklearn.autolog()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy_manual", float(acc))

    print("Accuracy:", acc)
    report = classification_report(y_test, y_pred, output_dict=False)
    print("Classification Report:\n", report)

    mlflow.log_text(str(report), "classification_report.txt")
