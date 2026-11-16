"""Chainlit app to chat with the fine-tuned SustainEd model deployed on a Vertex AI endpoint.

The learner picks a level and a business profile in the chat settings, then asks a question.
Each message is traced in Langfuse: a span with the user's question and the answer shown to them,
and inside it a generation with the request sent to the endpoint and the model's raw response.

Run from the root of the repository:
    uv run chainlit run src/app/main.py
"""

import re
import subprocess
import sys
import shutil
from pathlib import Path

import chainlit as cl
import requests
from chainlit.input_widget import Select
from langfuse import Langfuse, observe, propagate_attributes
from opentelemetry import context as otel_context
from transformers import AutoTokenizer

# Chainlit charge ce fichier directement : on rend le package src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.constants import BASE_MODEL_ID, ENDPOINT_ID, ENDPOINT_URL  # noqa: E402
from src.taxonomy import LEVELS, SECTORS, build_instruction, parse_sections  # noqa: E402

# Décodage glouton comme dans l'évaluation de la pipeline ; 256 tokens suffisent pour les 4 parties
GENERATION_PARAMETERS = {"max_new_tokens": 256, "do_sample": False}

DEFAULT_SETTINGS = {"level": next(iter(LEVELS)), "sector": next(iter(SECTORS))}

# Le chat template est celui de Phi-3, identique pour le modèle de base et le modèle fine-tuné
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# Lit LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY et LANGFUSE_HOST (chargés depuis .env par src.constants).
# Chainlit émet ses propres traces OpenTelemetry : on les bloque pour ne garder que les nôtres
langfuse = Langfuse(blocked_instrumentation_scopes=["chainlit"])


def get_access_token() -> str:
    """Get an access token for the gcloud user account (use a service account in production)."""
    gcloud = shutil.which("gcloud") or "gcloud"  # sous Windows, trouve gcloud.cmd
    return subprocess.check_output([gcloud, "auth", "print-access-token"], text=True).strip()


def build_prompt(question: str, level: str, sector: str) -> str:
    """Add the profile and the level to the question, then apply the chat template."""
    messages = [{"role": "user", "content": build_instruction(question, level, sector)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_response(model_output: str) -> str:
    """Extract the model's answer from the generated text (same logic as inference_component)."""
    match = re.search(r"<\|assistant\|>(.*?)(?:<\|end\|>|<\|endoftext\|>|$)", model_output, re.DOTALL)
    return match.group(1).strip() if match else model_output.strip()


def format_answer(answer: str) -> str:
    """Show the four sections as separate paragraphs with bold labels."""
    found = parse_sections(answer)
    if not found:
        return answer
    return "\n\n".join(f"**{label}:** {text}" for label, text in found.items())


def call_endpoint(prompt: str) -> str:
    """Send a templated prompt to the Vertex AI endpoint and return the raw generated text.

    Traced as a Langfuse generation: the request sent, the raw output, the model and its version,
    the token counts and the latency (measured by Langfuse from the start and end of the call).
    """
    payload = {"instances": [{"prompt": prompt}], "parameters": GENERATION_PARAMETERS}
    with langfuse.start_as_current_observation(
        name="vertex-endpoint-predict",
        as_type="generation",
        input=payload,
        model_parameters=GENERATION_PARAMETERS,
        metadata={"endpoint_id": ENDPOINT_ID},
    ) as generation:
        try:
            response = requests.post(
                ENDPOINT_URL,
                json=payload,
                headers={"Authorization": f"Bearer {get_access_token()}"},
                timeout=120,
            )
            response.raise_for_status()
        except (requests.RequestException, subprocess.CalledProcessError) as exc:
            generation.update(level="ERROR", status_message=str(exc))
            raise

        body = response.json()
        prediction = body["predictions"][0]
        generation.update(
            output=prediction["generated_text"],
            model=body.get("modelDisplayName"),
            version=body.get("modelVersionId"),
            usage_details={
                "input": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                "output": prediction.get("num_generated_tokens", 0),
            },
            metadata={
                "endpoint_id": ENDPOINT_ID,
                "deployed_model_id": body.get("deployedModelId"),
                "model_resource_name": body.get("model"),
            },
        )
        return prediction["generated_text"]


@observe(name="ask-sustained", capture_input=False, capture_output=False)
def ask_sustained(question: str, level: str, sector: str, session_id: str | None = None) -> str:
    """Answer the learner's question, traced as a Langfuse span around the endpoint call."""
    # Regroupe les messages d'une même conversation Chainlit dans une session Langfuse
    with propagate_attributes(session_id=session_id):
        langfuse.update_current_span(
            input=build_instruction(question, level, sector),
            metadata={"learner_level": level, "business_sector": sector, "question": question},
        )
        answer = extract_response(call_endpoint(build_prompt(question, level, sector)))
        langfuse.update_current_span(output=answer)
        return answer


def ask_sustained_in_new_trace(
    question: str, level: str, sector: str, session_id: str | None = None
) -> str:
    """Call ask_sustained in an empty OpenTelemetry context, so that each message gets its own trace.

    Chainlit runs all the messages of a chat inside the same OpenTelemetry context: without this,
    they would all end up in a single trace. The chat itself is grouped with the session ID.
    """
    token = otel_context.attach(otel_context.Context())
    try:
        return ask_sustained(question, level, sector, session_id=session_id)
    finally:
        otel_context.detach(token)


@cl.on_chat_start
async def on_chat_start() -> None:
    # Le niveau et le profil se choisissent dans les réglages du chat (icône à côté du champ de saisie)
    settings = await cl.ChatSettings(
        [
            Select(id="level", label="Explanation level", values=list(LEVELS), initial_index=0),
            Select(id="sector", label="Your business", values=list(SECTORS), initial_index=0),
        ]
    ).send()
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = {**DEFAULT_SETTINGS, **(cl.user_session.get("settings") or {})}
    try:
        # requests est bloquant : on l'exécute dans un thread pour ne pas figer l'interface
        answer = await cl.make_async(ask_sustained_in_new_trace)(
            message.content,
            settings["level"],
            settings["sector"],
            session_id=cl.context.session.id,
        )
        answer = format_answer(answer)
    except (requests.RequestException, subprocess.CalledProcessError) as exc:
        answer = f"⚠️ Endpoint call failed: {exc}"
    await cl.Message(content=answer).send()
