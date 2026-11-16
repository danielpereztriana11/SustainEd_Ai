from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(
    # Image avec torch + drivers CUDA pour l'entraînement sur GPU
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        # Versions compatibles avec les images de serving pré-construites de Vertex AI
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.2",
        "accelerate==1.1.1",
        "bitsandbytes==0.47.0",
        "datasets==3.1.0",
        "pandas==2.2.3",
        "tensorboard==2.18.0",
    ],
)
def fine_tuning_component(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 5,
    learning_rate: float = 3e-4,
    per_device_train_batch_size: int = 4,
    max_seq_length: int = 512,
    seed: int = 42,
):
    # Les imports DOIVENT être dans la fonction pour KFP
    import json
    import logging

    import pandas as pd
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    logging.getLogger().setLevel(logging.INFO)

    hyperparams = {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "max_seq_length": max_seq_length,
        "seed": seed,
    }
    logging.info(f"Hyper-paramètres : {hyperparams}")

    # A. Chargement des datasets (messages sérialisés en JSON par le composant de transformation)
    def load_messages(path: str) -> HFDataset:
        df = pd.read_csv(path)
        return HFDataset.from_dict({"messages": [json.loads(m) for m in df["messages"]]})

    train_split = load_messages(train_data.path)
    eval_split = load_messages(validation_data.path)
    logging.info(f"{len(train_split)} lignes (train), {len(eval_split)} lignes (validation).")

    # B. Chargement du modèle quantizé en 4 bits + adaptateurs LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="cuda:0"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
    )
    peft_model = get_peft_model(base_model, lora_config)
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    logging.info(f"Paramètres entraînables : {trainable_params:,} / {all_params:,}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    # C. Entraînement (logs TensorBoard écrits dans l'artifact de métriques)
    sft_config = SFTConfig(
        output_dir="/tmp/phi3-mini-sustained-adapter",
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        auto_find_batch_size=True,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=max_seq_length,
        packing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=["tensorboard"],
        logging_dir=metrics.path,
        seed=seed,
    )
    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_split,
        eval_dataset=eval_split,
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    # D. Sauvegarde des adaptateurs LoRA + tokenizer dans l'artifact du modèle
    logging.info(f"Sauvegarde du modèle dans : {model.path}")
    trainer.save_model(model.path)
    tokenizer.save_pretrained(model.path)
    model.metadata["base_model"] = model_id

    # E. Log des métriques et hyper-paramètres dans Kubeflow
    metrics.log_metric("train_loss", train_result.training_loss)
    metrics.log_metric("eval_loss", eval_result["eval_loss"])
    metrics.log_metric("trainable_params", trainable_params)
    metrics.log_metric("trainable_params_pct", 100 * trainable_params / all_params)
    for name, value in hyperparams.items():
        metrics.log_metric(name, value)

    logging.info("Succès ! Composant terminé.")
