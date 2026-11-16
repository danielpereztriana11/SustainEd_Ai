import pandas as pd
from kfp.dsl import Dataset, Metrics

from src.pipeline_components.evaluation_component import evaluation_component

REFERENCE = (
    "Definition: Scope 3 emissions are indirect emissions in your value chain.\n"
    "Example: For your coffee shop, they include growing the coffee beans you buy.\n"
    "Action: List your five biggest purchases.\n"
    "Check: Is purchased electricity Scope 2 or Scope 3?"
)


def test_evaluation_component_scores_predictions(tmp_path):
    predictions = Dataset(uri=str(tmp_path / "predictions.csv"))
    pd.DataFrame(
        {
            "user_input": ["I run a small coffee shop. What are Scope 3 emissions? Explain for a beginner."] * 3,
            "reference": [REFERENCE] * 3,
            "extracted_response": [
                REFERENCE,
                "Scope 3 emissions are all the other indirect emissions of a company.",
                "",
            ],
            "topic": ["climate and Scope 1/2/3"] * 3,
            "learner_level": ["beginner", "beginner", "advanced"],
            "business_sector": ["coffee shop"] * 3,
        }
    ).to_csv(predictions.path, index=False)
    evaluation_results = Dataset(uri=str(tmp_path / "evaluation_results.csv"))
    metrics = Metrics(uri=str(tmp_path / "metrics"))

    evaluation_component.python_func(
        predictions=predictions, evaluation_results=evaluation_results, metrics=metrics
    )

    results = pd.read_csv(evaluation_results.path)
    assert len(results) == 3
    assert list(results["format_adherence"]) == [1, 0, 0]
    assert list(results["has_definition"]) == [1, 0, 0]
    assert list(results["check_is_question"]) == [1, 0, 0]
    assert list(results["sector_in_example"]) == [1, 0, 0]
    assert results.loc[0, "rouge_score"] > results.loc[1, "rouge_score"] > results.loc[2, "rouge_score"]
    assert metrics.metadata["format_adherence_mean"] == 1 / 3
    assert "reading_grade_beginner_mean" in metrics.metadata
