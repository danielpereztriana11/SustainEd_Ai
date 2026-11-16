from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

# Les composants KFP doivent être autonomes : les deux composants ci-dessous ne peuvent pas partager
# de fonctions. Ils ne diffèrent que par le modèle chargé (fine-tuné ou modèle de base).
BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel"
PACKAGES = [
    "transformers==4.46.3",
    "peft==0.13.2",
    "accelerate==1.1.1",
    "pandas==2.2.3",
]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def inference_component(
    model: Input[Model],
    test_data: Input[Dataset],
    predictions: Output[Dataset],
    max_new_tokens: int = 256,
):
    # Les imports (et fonctions utilitaires) DOIVENT être dans la fonction pour KFP
    import json
    import logging
    import re
    from typing import Any

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.getLogger().setLevel(logging.INFO)

    def build_prompt(tokenizer: AutoTokenizer, instruction: str) -> str:
        """Build a prompt from the user's message applying the chat template."""
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model given a prompt."""
        # Le chat template contient déjà les tokens spéciaux
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        # Phi-3 termine sa réponse par <|end|> : on arrête la génération dessus
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
        with torch.no_grad():
            outputs = model.generate(**tokenized, eos_token_id=eos_token_ids, **kwargs)
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    def extract_response(model_output: str) -> str:
        """Extract the actual response from the model output."""
        match = re.search(r"<\|assistant\|>(.*?)(?:<\|end\|>|<\|endoftext\|>|$)", model_output, re.DOTALL)
        return match.group(1).strip() if match else model_output.strip()

    # A. Chargement du modèle fine-tuné (transformers charge les adaptateurs LoRA via peft)
    logging.info(f"Chargement du modèle depuis : {model.path}")
    tokenizer = AutoTokenizer.from_pretrained(model.path)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model.path, torch_dtype=torch.float16, device_map="cuda:0"
    )
    loaded_model.eval()

    # B. Chargement du dataset de test
    df = pd.read_csv(test_data.path, keep_default_na=False)
    messages = df["messages"].apply(json.loads)

    # C. Génération des prédictions
    rows = []
    for i, conversation in enumerate(messages):
        instruction = conversation[0]["content"]
        reference = conversation[1]["content"]
        prompt = build_prompt(tokenizer, instruction)
        output = generate_response(
            loaded_model, tokenizer, prompt, max_new_tokens=max_new_tokens, do_sample=False
        )
        rows.append(
            {
                "user_input": instruction,
                "reference": reference,
                "extracted_response": extract_response(output),
                "topic": df["topic"][i],
                "learner_level": df["learner_level"][i],
                "business_sector": df["business_sector"][i],
            }
        )
        if i % 10 == 0:
            logging.info(f"[{i + 1}/{len(messages)}] {instruction} -> {rows[-1]['extracted_response']}")

    # D. Sauvegarde des prédictions
    pd.DataFrame(rows).to_csv(predictions.path, index=False)
    logging.info(f"{len(rows)} prédictions sauvegardées dans : {predictions.path}")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def base_inference_component(
    test_data: Input[Dataset],
    predictions: Output[Dataset],
    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens: int = 256,
):
    # Les imports (et fonctions utilitaires) DOIVENT être dans la fonction pour KFP
    import json
    import logging
    import re
    from typing import Any

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.getLogger().setLevel(logging.INFO)

    def build_prompt(tokenizer: AutoTokenizer, instruction: str) -> str:
        """Build a prompt from the user's message applying the chat template."""
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model given a prompt."""
        # Le chat template contient déjà les tokens spéciaux
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        # Phi-3 termine sa réponse par <|end|> : on arrête la génération dessus
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
        with torch.no_grad():
            outputs = model.generate(**tokenized, eos_token_id=eos_token_ids, **kwargs)
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    def extract_response(model_output: str) -> str:
        """Extract the actual response from the model output."""
        match = re.search(r"<\|assistant\|>(.*?)(?:<\|end\|>|<\|endoftext\|>|$)", model_output, re.DOTALL)
        return match.group(1).strip() if match else model_output.strip()

    # A. Chargement du modèle de base, sans adaptateurs : sert de point de comparaison
    logging.info(f"Chargement du modèle de base : {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda:0"
    )
    loaded_model.eval()

    # B. Chargement du dataset de test
    df = pd.read_csv(test_data.path, keep_default_na=False)
    messages = df["messages"].apply(json.loads)

    # C. Génération des prédictions
    rows = []
    for i, conversation in enumerate(messages):
        instruction = conversation[0]["content"]
        reference = conversation[1]["content"]
        prompt = build_prompt(tokenizer, instruction)
        output = generate_response(
            loaded_model, tokenizer, prompt, max_new_tokens=max_new_tokens, do_sample=False
        )
        rows.append(
            {
                "user_input": instruction,
                "reference": reference,
                "extracted_response": extract_response(output),
                "topic": df["topic"][i],
                "learner_level": df["learner_level"][i],
                "business_sector": df["business_sector"][i],
            }
        )
        if i % 10 == 0:
            logging.info(f"[{i + 1}/{len(messages)}] {instruction} -> {rows[-1]['extracted_response']}")

    # D. Sauvegarde des prédictions
    pd.DataFrame(rows).to_csv(predictions.path, index=False)
    logging.info(f"{len(rows)} prédictions sauvegardées dans : {predictions.path}")
