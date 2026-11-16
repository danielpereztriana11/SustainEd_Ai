import os

from dotenv import load_dotenv

# Charge les variables du fichier .env (voir .env.example)
# Si GOOGLE_APPLICATION_CREDENTIALS est défini dans .env, il est aussi exporté ici
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "europe-west2")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

RAW_DATA_BLOB = "data/sustained_dataset.csv"
RAW_DATA_URI = f"gs://{BUCKET_NAME}/{RAW_DATA_BLOB}"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Dataset Hugging Face utilisé comme source de questions candidates
HF_DATASET_ID = "nessa01macias/sustainability_qa"

# Modèle Gemini (sur Vertex AI) qui rédige les brouillons du dataset, à relire avant l'entraînement
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "global")

# Endpoint Vertex AI sur lequel le modèle fine-tuné est déployé
PROJECT_NUMBER = os.getenv("GCP_PROJECT_NUMBER")
ENDPOINT_ID = os.getenv("GCP_ENDPOINT_ID")
ENDPOINT_URL = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}"
    f"/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
)
