"""Download the Hugging Face sustainability Q&A dataset and keep the questions usable as seeds.

Usage (from the root of the repository):
    uv run python -m scripts.download_hf_dataset

Writes:
    data/hf_raw.csv         the full dataset (question, answer)
    data/hf_candidates.csv  general questions tagged with one of the six themes

Most questions of this dataset are about one specific paper or report ("What does the author..."),
so they are filtered out. The dataset card states no license: the candidates are only used as
question seeds by scripts/build_dataset.py, and their origin is recorded in each record.
"""

import os
import re

import pandas as pd
from datasets import load_dataset

from src.constants import HF_DATASET_ID
from src.taxonomy import THEME_KEYWORDS

OUTPUT_DIR = "data"

# Questions liées à un document précis : inutilisables hors de leur contexte
DOCUMENT_SPECIFIC = re.compile(
    r"\b(author|paper|article|study|studies|document|text|passage|chapter|section|figure|table"
    r"|survey|respondent|interview|journal|book|according to|et al|mentioned|discussed|wrote"
    r"|published|the report|this report|the company|this company|company's|researchers?)\b"
    r"|\b(19|20)\d{2}\b",
    flags=re.IGNORECASE,
)
GENERAL_QUESTION = re.compile(r"^(what|how|why|which|when|who can|can|is|are|does|do)\b", re.IGNORECASE)


def find_theme(question: str) -> str | None:
    """Return the first theme with a keyword in the question, most specific themes first."""
    text = question.lower()
    # Les thèmes larges (fondamentaux) passent en dernier pour ne pas tout capter
    for theme in reversed(list(THEME_KEYWORDS)):
        if any(keyword in text for keyword in THEME_KEYWORDS[theme]):
            return theme
    return None


def select_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep short, general, well-formed questions and tag them with a theme."""
    df = df.copy()
    df["hf_row"] = df.index
    for column in ["question", "answer"]:
        df[column] = df[column].fillna("").astype(str).map(lambda text: " ".join(text.split()))

    keep = (
        df["question"].str.endswith("?")
        & df["question"].str.len().between(15, 140)
        & df["question"].map(lambda q: bool(GENERAL_QUESTION.match(q)))
        & ~df["question"].map(lambda q: bool(DOCUMENT_SPECIFIC.search(q)))
        & df["answer"].str.len().between(30, 800)
        # Réponses contenant un texte à trous laissé par la génération du dataset
        & ~df["answer"].str.contains(r"[\[\]]", regex=True)
    )
    df = df[keep]
    df = df[~df["question"].str.lower().duplicated()]
    df["topic"] = df["question"].map(find_theme)
    return df.dropna(subset=["topic"])[["hf_row", "topic", "question", "answer"]]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"⏳ Téléchargement de {HF_DATASET_ID}...")
    raw = load_dataset(HF_DATASET_ID, split="train").to_pandas()
    raw.to_csv(os.path.join(OUTPUT_DIR, "hf_raw.csv"), index=False)
    print(f"✅ {len(raw)} lignes dans {OUTPUT_DIR}/hf_raw.csv")

    candidates = select_candidates(raw)
    candidates.to_csv(os.path.join(OUTPUT_DIR, "hf_candidates.csv"), index=False)
    print(f"✅ {len(candidates)} questions générales dans {OUTPUT_DIR}/hf_candidates.csv")
    print(candidates["topic"].value_counts().to_string())


if __name__ == "__main__":
    main()
