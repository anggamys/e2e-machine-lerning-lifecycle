from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import os

app = FastAPI(title="Sentiment Analysis API")

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "downloaded_model/model")
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = mlflow.sklearn.load_model(MODEL_PATH)
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
        return PredictResponse(sentiment=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ok"}
