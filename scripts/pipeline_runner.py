from google.cloud import aiplatform
from kfp import compiler

from src.constants import BUCKET_NAME, PIPELINE_ROOT, PROJECT_ID, RAW_DATA_URI, REGION
from src.pipelines.model_training_pipeline import sustained_pipeline

PIPELINE_PACKAGE_PATH = "sustained_pipeline.yaml"

# ==========================================
# 1. Compilation de la pipeline
# ==========================================
print("⏳ Compilation de la pipeline...")
# Cette étape transforme le code Python en un fichier YAML compréhensible par Vertex AI
compiler.Compiler().compile(
    pipeline_func=sustained_pipeline,
    package_path=PIPELINE_PACKAGE_PATH,
)

# ==========================================
# 2. Lancement sur Vertex AI
# ==========================================
print(f"⏳ Initialisation de Vertex AI sur le projet {PROJECT_ID} ({REGION})...")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

job = aiplatform.PipelineJob(
    display_name="sustained_ai_pipeline",
    template_path=PIPELINE_PACKAGE_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"raw_data_uri": RAW_DATA_URI},
)

print("🚀 Soumission du job à Vertex AI...")
job.submit()
print("\n✅ Soumission réussie ! Cliquez sur le lien généré au-dessus pour voir la pipeline en direct.")
