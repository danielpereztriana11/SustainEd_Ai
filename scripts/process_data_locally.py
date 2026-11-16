import os

from kfp.dsl import Metrics

from src.constants import RAW_DATA_URI
from src.pipeline_components.data_transformation_component import prepare_sustained_data

# Exécute la fonction du composant en local (sans Vertex AI) pour vérifier le résultat
OUTPUT_DIR = "data"
LOCAL_DATASET = os.path.join(OUTPUT_DIR, "sustained_dataset.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Le CSV local est utilisé s'il existe, sinon celui du bucket
raw_data_uri = LOCAL_DATASET if os.path.exists(LOCAL_DATASET) else RAW_DATA_URI
metrics = Metrics(uri=os.path.join(OUTPUT_DIR, "data_metrics"))

prepare_sustained_data.python_func(
    raw_data_uri=raw_data_uri,
    train_output_path=os.path.join(OUTPUT_DIR, "train_sustained.csv"),
    validation_output_path=os.path.join(OUTPUT_DIR, "validation_sustained.csv"),
    test_output_path=os.path.join(OUTPUT_DIR, "test_sustained.csv"),
    metrics=metrics,
)

print(f"✅ Fichiers train, validation et test créés dans le dossier '{OUTPUT_DIR}/' à partir de {raw_data_uri}")
print(metrics.metadata)
