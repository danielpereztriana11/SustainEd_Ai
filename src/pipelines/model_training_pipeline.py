from kfp import dsl

from src.pipeline_components.data_transformation_component import prepare_sustained_data
from src.pipeline_components.evaluation_component import evaluation_component
from src.pipeline_components.fine_tuning_component import fine_tuning_component
from src.pipeline_components.inference_component import (
    base_inference_component,
    inference_component,
)


def use_gpu(task: dsl.PipelineTask) -> dsl.PipelineTask:
    # Machine avec un GPU T4 (région avec un quota GPU)
    return (
        task.set_accelerator_type("NVIDIA_TESLA_T4")
        .set_accelerator_limit(1)
        .set_cpu_limit("16")
        .set_memory_limit("50G")
    )


@dsl.pipeline(
    name="sustained-ai-training-pipeline",
    description=(
        "Préparation des données, fine-tuning LoRA de Phi-3, inférence et évaluation "
        "du modèle fine-tuné et du modèle de base."
    ),
)
def sustained_pipeline(raw_data_uri: str):
    # 1. Préparation des données
    data_task = prepare_sustained_data(raw_data_uri=raw_data_uri)

    # 2. Fine-tuning de Phi-3 avec LoRA sur le jeu d'entraînement
    fine_tuning_task = use_gpu(
        fine_tuning_component(
            train_data=data_task.outputs["train_output_path"],
            validation_data=data_task.outputs["validation_output_path"],
            num_train_epochs=2,
        )
    )

    # 3. Prédictions du modèle fine-tuné sur le jeu de test, puis évaluation
    inference_task = use_gpu(
        inference_component(
            model=fine_tuning_task.outputs["model"],
            test_data=data_task.outputs["test_output_path"],
        )
    ).set_display_name("inference-fine-tuned")
    evaluation_component(predictions=inference_task.outputs["predictions"]).set_display_name(
        "evaluation-fine-tuned"
    )

    # 4. Mêmes étapes avec le modèle de base, pour mesurer ce que le fine-tuning apporte.
    # Lancé après l'inférence du modèle fine-tuné : un seul GPU T4 utilisé à la fois (quota)
    base_inference_task = (
        use_gpu(base_inference_component(test_data=data_task.outputs["test_output_path"]))
        .set_display_name("inference-base")
        .after(inference_task)
    )
    evaluation_component(predictions=base_inference_task.outputs["predictions"]).set_display_name(
        "evaluation-base"
    )
