FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies for API
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the inference code
COPY inference/inference.py /app/inference.py

# The downloaded model will be copied here by GitHub Actions
COPY downloaded_model /app/downloaded_model

ENV MODEL_PATH=/app/downloaded_model/model

EXPOSE 8000

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
