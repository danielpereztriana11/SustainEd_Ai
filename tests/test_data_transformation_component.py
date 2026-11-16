import json

import pandas as pd
from kfp.dsl import Metrics

from src.pipeline_components.data_transformation_component import prepare_sustained_data

ANSWER = "Definition: A.\nExample: B.\nAction: C.\nCheck: D?"


def make_raw_data(n_concepts: int = 10, variants: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"c{c}-v{v}",
                "topic": "circular economy",
                "concept": f"concept {c}",
                "learner_level": "beginner",
                "business_sector": "coffee shop",
                "instruction": f"I run a small coffee shop.  What is concept {c}, part {v}? ",
                "target_response": ANSWER,
                "reviewer_status": "reviewed" if c < 5 else "draft",
            }
            for c in range(n_concepts)
            for v in range(variants)
        ]
    )


def run_component(tmp_path, raw: pd.DataFrame, **kwargs) -> tuple[dict[str, pd.DataFrame], Metrics]:
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    paths = {name: tmp_path / f"{name}.csv" for name in ["train", "validation", "test"]}
    metrics = Metrics(uri=str(tmp_path / "metrics"))
    prepare_sustained_data.python_func(
        raw_data_uri=str(raw_path),
        train_output_path=str(paths["train"]),
        validation_output_path=str(paths["validation"]),
        test_output_path=str(paths["test"]),
        metrics=metrics,
        max_tokens=0,  # pas de téléchargement du tokenizer pendant les tests
        **kwargs,
    )
    return {name: pd.read_csv(path) for name, path in paths.items()}, metrics


def test_formats_and_splits_by_concept(tmp_path):
    splits, metrics = run_component(tmp_path, make_raw_data())

    assert [len(splits[name]) for name in ["train", "validation", "test"]] == [16, 2, 2]
    assert list(splits["train"].columns) == ["messages", "topic", "learner_level", "business_sector"]
    assert metrics.metadata["train_rows"] == 16

    messages = json.loads(splits["train"]["messages"][0])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    # Les espaces en trop sont supprimés
    assert messages[0]["content"].startswith("I run a small coffee shop. What is concept")
    assert messages[1]["content"] == ANSWER

    # Toutes les variantes d'un concept sont dans le même split
    def concepts(df: pd.DataFrame) -> set[str]:
        return {json.loads(m)[0]["content"].split(",")[0] for m in df["messages"]}

    assert not concepts(splits["train"]) & concepts(splits["test"])
    assert not concepts(splits["train"]) & concepts(splits["validation"])


def test_uses_dataset_split_column(tmp_path):
    raw = make_raw_data()
    raw["dataset_split"] = ["test"] * 4 + ["validation"] * 6 + ["train"] * 10

    splits, _ = run_component(tmp_path, raw)

    assert [len(splits[name]) for name in ["train", "validation", "test"]] == [10, 6, 4]


def test_drops_duplicates_and_malformed_answers(tmp_path):
    raw = make_raw_data()
    raw.loc[0, "target_response"] = "Scope 3 emissions are indirect emissions."  # sans les 4 parties
    raw.loc[2, "target_response"] = "Example: B.\nDefinition: A.\nAction: C.\nCheck: D?"  # désordre
    raw.loc[4, "instruction"] = raw.loc[5, "instruction"].upper()  # doublon

    splits, metrics = run_component(tmp_path, raw)

    assert sum(len(df) for df in splits.values()) == 17
    assert metrics.metadata["raw_rows"] == 20
    assert metrics.metadata["well_formed_rows"] == 18
    assert metrics.metadata["deduplicated_rows"] == 17


def test_require_reviewed_keeps_only_reviewed_rows(tmp_path):
    splits, _ = run_component(tmp_path, make_raw_data(), require_reviewed=True)

    assert sum(len(df) for df in splits.values()) == 10
