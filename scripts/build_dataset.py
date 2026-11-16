"""Draft the SustainEd training dataset with Gemini on Vertex AI.

For each concept, Gemini writes five examples (one per learner level and business profile) in the
four-part teaching format. The result is a CSV with the record fields of the project brief.
Every record starts with reviewer_status = "draft": read and correct them before training.

Usage (from the root of the repository):
    uv run python -m scripts.build_dataset --limit-concepts 10    # pilot of about 50 examples
    uv run python -m scripts.build_dataset                        # 120 concepts, 600 examples
    uv run python -m scripts.build_dataset --upload               # also copy the CSV to the bucket

The drafts are cached in data/drafts/, so a run that stops can be restarted where it stopped.
If data/hf_candidates.csv exists (see scripts/download_hf_dataset.py), --hf-per-theme questions
of each theme come from the Hugging Face dataset instead of the list in src/taxonomy.py.
"""

import argparse
import json
import os
import random
import time
from itertools import zip_longest

import pandas as pd

from src.constants import (
    BUCKET_NAME,
    GEMINI_LOCATION,
    GEMINI_MODEL,
    HF_DATASET_ID,
    PROJECT_ID,
    RAW_DATA_BLOB,
    RAW_DATA_URI,
)
from src.taxonomy import (
    LEVELS,
    SECTORS,
    THEMES,
    build_instruction,
    format_target_response,
    variants_for_concept,
)

OUTPUT_DIR = "data"
DRAFTS_DIR = os.path.join(OUTPUT_DIR, "drafts")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sustained_dataset.csv")
HF_CANDIDATES_PATH = os.path.join(OUTPUT_DIR, "hf_candidates.csv")

VARIANTS_PER_CONCEPT = 5
FIELDS = ["question", "definition", "example", "action", "check"]

PROMPT = """\
You write training examples for SustainEd AI, a tutor that explains sustainability to employees \
and small-business owners. It gives educational explanations, never legal or regulatory advice.

Theme: {topic}
Concept: {concept}
{seed}
Write {n} examples about this concept, one for each learner below, in this order:
{learners}

Each example has five fields:
- question: a short question this learner would ask about the concept. Do not mention the level or \
the business in it: they are added automatically. Change the angle from one example to the next \
(what it is, why it matters, how it differs from a related idea, where to start).
- definition: one or two sentences that define the concept and answer the question.
- example: one or two sentences with a concrete case in that learner's business.
- action: one sentence with a practical next step that this business can take this month.
- check: one short question that tests whether the learner understood. It must end with "?" and \
be answerable from the definition and the example.

Level:
- beginner: everyday words, short sentences, no jargon or acronyms without an explanation.
- intermediate: the standard terms of the field, each briefly explained.
- advanced: precise technical vocabulary, the relevant standard or method named (for example the \
GHG Protocol categories), and one nuance or limit.

Accuracy:
- State only well-established facts, consistent with the GHG Protocol, the US EPA guidance for \
small businesses, the GRI Standards and the European Commission's VSME material.
- No invented statistics, dates, thresholds or deadlines. Describe regulations in general terms.
- No markdown, no labels, no lists inside a field.

Two examples of the expected quality:

Learner: level = beginner, business = coffee shop
question: What are Scope 3 emissions?
definition: Scope 3 emissions are the indirect emissions in your value chain that are not covered \
by Scope 2. They come from what you buy, sell and throw away, not from your own equipment or \
your electricity.
example: For your coffee shop, they include growing and shipping the coffee beans and making the \
cups you buy.
action: List your five biggest purchases and ask each supplier what they know about their emissions.
check: Is the electricity you buy for the shop normally Scope 2 or Scope 3?

Learner: level = advanced, business = IT consulting firm
question: How do location-based and market-based Scope 2 figures differ?
definition: The GHG Protocol Scope 2 Guidance asks for two figures: the location-based method \
applies the average emission factor of the grid where the electricity is used, while the \
market-based method applies the factors of the contracts you chose, such as a supplier tariff or \
energy attribute certificates. The market-based figure depends on the quality of those instruments.
example: Your consulting firm's office on a renewable tariff backed by certificates may report \
near-zero market-based emissions, while its location-based figure still reflects the local grid mix.
action: Ask your electricity supplier for the contractual emission factor and the certificates \
behind your tariff, then report both figures side by side.
check: Which Scope 2 method reflects the average grid mix where the electricity is used?

Return a JSON array of {n} objects with exactly these keys: {fields}.
"""


def select_concepts(hf_per_theme: int, seed: int) -> list[dict]:
    """List the concepts to draft, interleaved by theme so that any prefix covers all themes."""
    hf_candidates = None
    if hf_per_theme > 0 and os.path.exists(HF_CANDIDATES_PATH):
        hf_candidates = pd.read_csv(HF_CANDIDATES_PATH, keep_default_na=False)

    rng = random.Random(seed)
    per_theme = []
    for theme_index, (topic, concept_names) in enumerate(THEMES.items(), start=1):
        concepts = [{"topic": topic, "concept": name, "hf_row": None} for name in concept_names]
        if hf_candidates is not None:
            rows = hf_candidates[hf_candidates["topic"] == topic].to_dict("records")
            picked = rng.sample(rows, min(hf_per_theme, len(rows)))
            # Les questions Hugging Face remplacent les derniers concepts de la liste
            concepts = concepts[: len(concepts) - len(picked)] + [
                {
                    "topic": topic,
                    "concept": row["question"],
                    "hf_row": int(row["hf_row"]),
                    "hf_answer": row["answer"],
                }
                for row in picked
            ]
        for concept_index, concept in enumerate(concepts):
            concept["concept_id"] = f"t{theme_index}-c{concept_index + 1:02d}"
            concept["variants"] = variants_for_concept((theme_index - 1) * 20 + concept_index)
        rng.shuffle(concepts)
        per_theme.append(concepts)

    interleaved = zip_longest(*per_theme)
    return [concept for group in interleaved for concept in group if concept is not None]


def assign_splits(concepts: list[dict], validation_size: float, test_size: float) -> None:
    """Assign a split to each concept: all the variants of a concept stay on the same side."""
    n_test = max(1, round(len(concepts) * test_size))
    n_validation = max(1, round(len(concepts) * validation_size))
    for i, concept in enumerate(concepts):
        if i < n_test:
            concept["dataset_split"] = "test"
        elif i < n_test + n_validation:
            concept["dataset_split"] = "validation"
        else:
            concept["dataset_split"] = "train"


def build_prompt(concept: dict) -> str:
    seed = ""
    if concept["hf_row"] is not None:
        seed = (
            "This concept is a question taken from an existing Q&A dataset. Its answer there was: "
            f"\"{concept['hf_answer']}\". That answer is unverified: use only what is generally true.\n"
        )
    learners = "\n".join(
        f"{i}. level = {level}, business = {sector}"
        for i, (level, sector) in enumerate(concept["variants"], start=1)
    )
    return PROMPT.format(
        topic=concept["topic"],
        concept=concept["concept"],
        seed=seed,
        n=len(concept["variants"]),
        learners=learners,
        fields=", ".join(FIELDS),
    )


def validate_examples(examples: object, n_expected: int) -> list[dict]:
    """Check the shape of Gemini's answer and return the cleaned examples."""
    if not isinstance(examples, list) or len(examples) != n_expected:
        raise ValueError(f"{n_expected} exemples attendus")
    cleaned = []
    for example in examples:
        values = {field: " ".join(str(example.get(field, "")).split()) for field in FIELDS}
        if not all(values.values()):
            raise ValueError("champ vide")
        if not values["question"].endswith("?") or not values["check"].endswith("?"):
            raise ValueError("question ou check sans point d'interrogation")
        cleaned.append(values)
    return cleaned


def draft_concept(client, concept: dict, max_attempts: int = 4) -> list[dict]:
    """Ask Gemini for the examples of one concept, with a cache on disk and a few retries."""
    from google.genai import types

    cache_path = os.path.join(DRAFTS_DIR, f"{concept['concept_id']}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as file:
            cached = json.load(file)
        # Le cache n'est valable que pour le même concept (la sélection Hugging Face peut changer)
        if cached["concept"] == concept["concept"]:
            return cached["examples"]

    prompt = build_prompt(concept)
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7, response_mime_type="application/json"
                ),
            )
            examples = validate_examples(json.loads(response.text), len(concept["variants"]))
            break
        except Exception as exc:  # erreurs de quota, de réseau ou de format : on réessaie
            if attempt == max_attempts:
                raise
            print(f"   ↻ {concept['concept_id']} : {exc} (essai {attempt}/{max_attempts})")
            time.sleep(5 * attempt)

    with open(cache_path, "w") as file:
        json.dump({"concept": concept["concept"], "examples": examples}, file, indent=2)
    return examples


def to_records(concept: dict, examples: list[dict]) -> list[dict]:
    """Turn the examples of one concept into dataset records with their provenance."""
    if concept["hf_row"] is None:
        sources = f"Drafted by {GEMINI_MODEL}; to verify against GHG Protocol, US EPA and EC VSME material"
        license_status = "original draft"
    else:
        sources = (
            f"Drafted by {GEMINI_MODEL} from the question of row {concept['hf_row']} of "
            f"{HF_DATASET_ID}; to verify against GHG Protocol, US EPA and EC VSME material"
        )
        license_status = "original draft; seed question from a dataset with no stated license"

    records = []
    for i, ((level, sector), example) in enumerate(zip(concept["variants"], examples), start=1):
        records.append(
            {
                "id": f"{concept['concept_id']}-v{i}",
                "topic": concept["topic"],
                "concept": concept["concept"],
                "learner_level": level,
                "business_sector": sector,
                "instruction": build_instruction(example["question"], level, sector),
                "target_response": format_target_response(
                    example["definition"], example["example"], example["action"], example["check"]
                ),
                "factual_sources": sources,
                "license_status": license_status,
                "reviewer_status": "draft",
                "dataset_split": concept["dataset_split"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit-concepts", type=int, default=None, help="Draft only the first N concepts")
    parser.add_argument("--hf-per-theme", type=int, default=5, help="Hugging Face questions per theme")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true", help=f"Copy the CSV to {RAW_DATA_URI}")
    parser.add_argument("--overwrite", action="store_true", help="Replace a CSV that was already reviewed")
    args = parser.parse_args()

    # Le CSV est relu et corrigé à la main : on ne l'écrase pas par accident
    if os.path.exists(OUTPUT_PATH) and not args.overwrite:
        statuses = pd.read_csv(OUTPUT_PATH, keep_default_na=False)["reviewer_status"]
        if (statuses != "draft").any():
            parser.error(f"{OUTPUT_PATH} contient des exemples déjà relus : ajouter --overwrite pour le remplacer")

    from google import genai

    os.makedirs(DRAFTS_DIR, exist_ok=True)

    concepts = select_concepts(args.hf_per_theme, args.seed)[: args.limit_concepts]
    assign_splits(concepts, args.validation_size, args.test_size)
    n_hf = sum(concept["hf_row"] is not None for concept in concepts)
    print(f"⏳ {len(concepts)} concepts à rédiger avec {GEMINI_MODEL}, dont {n_hf} issus de Hugging Face...")

    # Gemini est appelé via Vertex AI, avec les mêmes identifiants gcloud que la pipeline
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GEMINI_LOCATION)

    records = []
    for i, concept in enumerate(concepts, start=1):
        records.extend(to_records(concept, draft_concept(client, concept)))
        if i % 10 == 0 or i == len(concepts):
            print(f"   {i}/{len(concepts)} concepts")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ {len(df)} exemples dans {OUTPUT_PATH}")
    print(df["dataset_split"].value_counts().to_string())
    print(f"   Niveaux : {', '.join(LEVELS)} | Profils : {', '.join(SECTORS)}")

    if args.upload:
        from google.cloud import storage

        storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME).blob(RAW_DATA_BLOB).upload_from_filename(
            OUTPUT_PATH
        )
        print(f"✅ Envoyé vers {RAW_DATA_URI}")


if __name__ == "__main__":
    main()
