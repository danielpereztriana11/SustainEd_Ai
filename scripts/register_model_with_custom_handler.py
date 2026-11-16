"""Upload src/handler.py next to a fine-tuned model in GCS, then register the model in Vertex AI.

Usage (from the root of the repository):
    uv run python -m scripts.register_model_with_custom_handler \\
        --model-uri gs://<bucket>/pipeline_root/<project_number>/<run_id>/fine-tuning-component_<id>/model \\
        --model-name sustained-phi3

Add ``--parent-model <model_id>`` to register the model as a new version of an existing model.
"""

import argparse
from pathlib import Path

from google.cloud import aiplatform, storage

from src.constants import PROJECT_ID, REGION

HANDLER_PATH = Path(__file__).resolve().parents[1] / "src" / "handler.py"

# Conteneur Hugging Face pré-construit : versions de transformers / peft identiques à l'entraînement
SERVING_CONTAINER_IMAGE_URI = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311"
)


def upload_handler(model_uri: str) -> str:
    """Upload the local handler.py file into the model artifact directory in GCS."""
    bucket_name, prefix = model_uri.removeprefix("gs://").rstrip("/").split("/", 1)
    blob = storage.Client(project=PROJECT_ID).bucket(bucket_name).blob(f"{prefix}/handler.py")
    blob.upload_from_filename(str(HANDLER_PATH))
    return f"gs://{bucket_name}/{blob.name}"


def register_model(
    model_uri: str,
    model_name: str,
    description: str,
    parent_model: str | None = None,
) -> aiplatform.Model:
    """Register the model artifacts in the Vertex AI Model Registry."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    return aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_uri.rstrip("/"),
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
        # Port attendu par le conteneur, sur lequel Vertex AI envoie les requêtes
        serving_container_ports=[8080],
        parent_model=parent_model,
        description=description,
        labels={"base_model": "phi-3-mini-4k-instruct", "adapter": "lora"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", required=True, help="GCS URI of the fine-tuned model artifact")
    parser.add_argument("--model-name", required=True, help="Display name in the Model Registry")
    parser.add_argument(
        "--description",
        default="Phi-3-mini-4k-instruct fine-tuned with LoRA to teach sustainability in a four-part format",
    )
    parser.add_argument(
        "--parent-model",
        default=None,
        help="ID or resource name of an existing model, to register a new version of it",
    )
    args = parser.parse_args()

    if not args.model_uri.startswith("gs://"):
        parser.error("--model-uri must be a gs:// URI")

    print(f"⏳ Envoi de {HANDLER_PATH.name} vers {args.model_uri}...")
    print(f"✅ Handler disponible : {upload_handler(args.model_uri)}")

    print(f"⏳ Enregistrement du modèle '{args.model_name}' dans le Model Registry...")
    model = register_model(args.model_uri, args.model_name, args.description, args.parent_model)
    print(f"✅ Modèle enregistré : {model.resource_name} (version {model.version_id})")
    print(
        f"https://console.cloud.google.com/vertex-ai/models/locations/{REGION}"
        f"/models/{model.name}/versions/{model.version_id}?project={PROJECT_ID}"
    )


if __name__ == "__main__":
    main()
