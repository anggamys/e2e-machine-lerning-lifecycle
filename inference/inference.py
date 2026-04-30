from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
import cloudpickle
import os

app = FastAPI(title="Sentiment Analysis API")

# Initialize Prometheus Instrumentator
Instrumentator().instrument(app).expose(app)

# Custom Prometheus Metrics
PREDICTION_COUNTER = Counter(
    "sentiment_predictions_total",
    "Total number of sentiment predictions made",
    ["sentiment"],
)

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "downloaded_model/model")
try:
    pkl_path = os.path.join(MODEL_PATH, "model.pkl")
    print(f"Loading model from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        model = cloudpickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: int


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Model is a pipeline with TfidfVectorizer and LogisticRegression
        prediction = model.predict([request.text])
        sentiment_result = int(prediction[0])

        # Increment Prometheus counter
        PREDICTION_COUNTER.labels(sentiment=str(sentiment_result)).inc()

        return PredictResponse(sentiment=sentiment_result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ok"}
