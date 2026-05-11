# download_models.py
from google.cloud import storage
import os
os.makedirs("models", exist_ok=True)

client = storage.Client()
bucket = client.bucket("fraud-detection-models")
bucket.blob("models/artifacts/best_model.pkl").download_to_filename("models/best_model.pkl")
bucket.blob("models/artifacts/best_parameters.pkl").download_to_filename("models/best_parameters.pkl")
bucket.blob("models/artifacts/scaler.pkl").download_to_filename("models/scaler.pkl")