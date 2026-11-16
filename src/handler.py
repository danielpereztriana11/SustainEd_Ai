"""Custom handler for the pre-built Hugging Face inference container on Vertex AI.

The container downloads the model artifacts (LoRA adapters, tokenizer and this file) from GCS
to a local directory, then calls ``EndpointHandler(model_dir)`` once and ``handler(request)``
for each prediction request. The file is uploaded next to the model by
``scripts/register_model_with_custom_handler.py``, so it must stay self-contained (no ``src`` imports).

Request format:
    {
        "instances": [{"prompt": "<|user|>\\nI run a small coffee shop. What are Scope 3 emissions? Explain for a beginner.<|end|>\\n<|assistant|>\\n"}],
        "parameters": {"max_new_tokens": 256, "do_sample": false}
    }

Response format:
    {"predictions": [{"generated_text": "<prompt + completion>", "num_generated_tokens": 120}]}
"""

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# En local : dossier où le modèle a été téléchargé depuis GCS (cf. notebooks/inference.ipynb)
MODEL_DIR = os.getenv("MODEL_DIR", "phi3-mini-sustained-adapter")

# Mêmes paramètres que l'inférence de la pipeline, surchargés par "parameters" dans la requête
DEFAULT_PARAMETERS = {"max_new_tokens": 256, "do_sample": False}


class EndpointHandler:
    """Handler for processing inference requests using a Hugging Face model."""

    def __init__(self, model_dir: str = MODEL_DIR, **kwargs: Any) -> None:
        """Load tokenizer and model from the specified directory."""
        if torch.cuda.is_available():
            device_map, dtype = "cuda:0", torch.float16  # le T4 ne gère pas bfloat16
        elif torch.backends.mps.is_available():
            device_map, dtype = "mps", torch.bfloat16
        else:
            device_map, dtype = "cpu", torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # transformers détecte adapter_config.json : modèle de base + adaptateurs LoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device_map
        )
        self.model.eval()

        # Phi-3 termine sa réponse par <|end|> : on arrête la génération dessus
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        end_token_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
        if end_token_id not in (None, self.tokenizer.unk_token_id):
            self.eos_token_ids.append(end_token_id)

    def __call__(self, data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Generate a completion for each instance of a Vertex AI prediction request."""
        parameters = {**DEFAULT_PARAMETERS, **(data.get("parameters") or {})}

        predictions = []
        for instance in data["instances"]:
            prompt = instance["prompt"] if isinstance(instance, dict) else instance
            # Le prompt contient déjà le chat template et donc ses tokens spéciaux
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
                self.model.device
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    eos_token_id=self.eos_token_ids,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    **parameters,
                )
            predictions.append(
                {
                    "generated_text": self.tokenizer.decode(outputs[0], skip_special_tokens=False),
                    "num_generated_tokens": int(outputs.shape[1] - inputs["input_ids"].shape[1]),
                }
            )
        return {"predictions": predictions}
