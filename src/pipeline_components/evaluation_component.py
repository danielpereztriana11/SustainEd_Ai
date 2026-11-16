from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output


@dsl.component(
    # ragas a besoin de git, absent des images Python standard
    base_image="cicirello/pyaction:3.11",
    packages_to_install=[
        "ragas==0.4.3",
        "rouge-score==0.1.2",
        "sacrebleu==2.6.0",
        # ragas 0.4.3 importe un module supprimé dans langchain-community 0.4
        "langchain-community==0.3.31",
        "pandas==2.2.3",
    ],
)
def evaluation_component(
    predictions: Input[Dataset],
    evaluation_results: Output[Dataset],
    metrics: Output[Metrics],
):
    # Les imports (et fonctions utilitaires) DOIVENT être dans la fonction pour KFP
    import logging
    import re

    import pandas as pd
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, RougeScore
    from ragas.metrics.base import SingleTurnMetric

    logging.getLogger().setLevel(logging.INFO)

    sections = ["Definition", "Example", "Action", "Check"]

    def compute_text_metrics(
        user_input: str,
        response: str,
        reference: str,
        metric_definitions: list[SingleTurnMetric],
    ) -> dict[str, float]:
        """Compare a prediction with its reference answer (ROUGE and BLEU)."""
        if not response.strip():
            return {metric.name: 0.0 for metric in metric_definitions}
        sample = SingleTurnSample(user_input=user_input, response=response, reference=reference)
        return {metric.name: float(metric.single_turn_score(sample)) for metric in metric_definitions}

    def parse_sections(response: str) -> dict[str, str]:
        """Return the text of each labelled section found in a response."""
        pattern = r"^\s*\**(" + "|".join(sections) + r")\**\s*:\**\s*"
        found, current = {}, None
        for line in response.splitlines():
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                current = match.group(1).capitalize()
                found[current] = line[match.end() :].strip()
            elif current and line.strip():
                found[current] = f"{found[current]} {line.strip()}".strip()
        return found

    def count_syllables(word: str) -> int:
        """Rough syllable count: groups of vowels, minus a silent final e."""
        groups = re.findall(r"[aeiouy]+", word.lower())
        silent_e = word.lower().endswith("e") and not word.lower().endswith("le") and len(groups) > 1
        return max(1, len(groups) - int(silent_e))

    def reading_grade(text: str) -> float:
        """Flesch-Kincaid grade level: higher means harder to read."""
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        n_sentences = max(1, len(re.findall(r"[.!?]+", text)))
        if not words:
            return 0.0
        n_syllables = sum(count_syllables(word) for word in words)
        return 0.39 * len(words) / n_sentences + 11.8 * n_syllables / len(words) - 15.59

    def compute_format_metrics(response: str, business_sector: str) -> dict[str, float]:
        """Check the four-part teaching format, the business example and the reading level."""
        found = parse_sections(response)
        has_section = {f"has_{label.lower()}": float(bool(found.get(label))) for label in sections}
        # Les quatre parties doivent être présentes, non vides et dans l'ordre
        in_order = list(found) == sections and all(found.values())
        # Mots du profil (ex. "coffee", "shop") retrouvés dans la partie Example
        sector_words = [w for w in re.findall(r"[a-z]+", business_sector.lower()) if len(w) > 3]
        example = found.get("Example", "").lower()
        return {
            **has_section,
            "format_adherence": float(in_order),
            "check_is_question": float(found.get("Check", "").rstrip().endswith("?")),
            "sector_in_example": float(any(word in example for word in sector_words)),
            "word_count": float(len(response.split())),
            "reading_grade": reading_grade(response),
        }

    # Métriques sans LLM : comparaison avec la réponse de référence + respect du format pédagogique
    metric_definitions = [RougeScore(), BleuScore()]

    df = pd.read_csv(predictions.path, keep_default_na=False)
    logging.info(f"{len(df)} prédictions chargées depuis : {predictions.path}")

    per_sample_metrics = pd.DataFrame(
        [
            {
                **compute_text_metrics(
                    row.user_input, row.extracted_response, row.reference, metric_definitions
                ),
                **compute_format_metrics(row.extracted_response, row.business_sector),
            }
            for row in df.itertuples()
        ]
    )

    pd.concat([df, per_sample_metrics], axis=1).to_csv(evaluation_results.path, index=False)
    logging.info(f"Résultats par exemple sauvegardés dans : {evaluation_results.path}")

    aggregated = {f"{name}_mean": float(per_sample_metrics[name].mean()) for name in per_sample_metrics}
    # Adaptation au niveau : une réponse "advanced" doit être plus difficile à lire qu'une "beginner"
    for level, grades in per_sample_metrics["reading_grade"].groupby(df["learner_level"]):
        if level:
            aggregated[f"reading_grade_{level}_mean"] = float(grades.mean())

    for name, value in aggregated.items():
        logging.info(f"{name} = {value:.4f}")
        metrics.log_metric(name, value)
