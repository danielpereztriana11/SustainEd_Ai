from kfp import dsl
from kfp.dsl import Metrics, Output, OutputPath


@dsl.component(
    base_image="python:3.11-slim",  # Même version de Python qu'en local
        packages_to_install=[
        "pandas==3.1.0",
        "gcsfs==2026.8.2",  # Permet à pandas de lire les URI gs://
        "transformers==4.46.3",  # Uniquement pour le tokenizer du modèle de base
        "jinja2==3.1.6",  # Nécessaire au chat template du tokenizer
    ],
)
def prepare_sustained_data(
    raw_data_uri: str,
    train_output_path: OutputPath("CSV"),
    validation_output_path: OutputPath("CSV"),
    test_output_path: OutputPath("CSV"),
    metrics: Output[Metrics],
    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    max_tokens: int = 512,
    require_reviewed: bool = False,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
):
    # Les imports DOIVENT être dans la fonction pour KFP
    import json
    import logging
    import random
    import re

    import pandas as pd

    logging.getLogger().setLevel(logging.INFO)
    logging.info("Démarrage de la préparation des données SustainEd...")

    sections = ["Definition", "Example", "Action", "Check"]
    metadata_columns = ["topic", "learner_level", "business_sector"]

    # A. Lecture depuis GCS (pandas lit directement les URI gs://, ou un chemin local)
    logging.info(f"Lecture des données depuis : {raw_data_uri}")
    df = pd.read_csv(raw_data_uri, keep_default_na=False, dtype=str)
    n_raw = len(df)
    logging.info(f"{n_raw} lignes chargées.")

    # B. Nettoyage : espaces normalisés, lignes vides supprimées
    def normalize(text: str) -> str:
        lines = [" ".join(line.split()) for line in str(text).splitlines()]
        return "\n".join(line for line in lines if line)

    df["instruction"] = df["instruction"].map(normalize)
    df["target_response"] = df["target_response"].map(normalize)
    df = df[(df["instruction"] != "") & (df["target_response"] != "")]

    # Seuls les exemples relus sont gardés si require_reviewed est activé
    if require_reviewed:
        df = df[df["reviewer_status"].str.strip().str.lower() == "reviewed"]
        logging.info(f"{len(df)} lignes avec reviewer_status = reviewed.")

    # Chaque réponse doit contenir les quatre parties, dans l'ordre
    format_pattern = re.compile(
        r"\A" + r".+?".join(rf"^{label}:" for label in sections) + r".+\Z",
        flags=re.DOTALL | re.MULTILINE,
    )
    df = df[df["target_response"].map(lambda text: bool(format_pattern.match(text)))]
    n_well_formed = len(df)

    # Déduplication sur la question, sans tenir compte de la casse ni de la ponctuation
    dedup_key = df["instruction"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df[~dedup_key.duplicated()].reset_index(drop=True)
    n_deduplicated = len(df)
    logging.info(f"{n_well_formed} lignes bien formées, {n_deduplicated} après déduplication.")

    # C. Formatage conversationnel pour Phi-3
    df["messages"] = [
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        for instruction, response in zip(df["instruction"], df["target_response"])
    ]

    # Les conversations trop longues pour l'entraînement sont écartées (tokenizer du modèle de base)
    if max_tokens > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        n_tokens = df["messages"].map(lambda m: len(tokenizer.apply_chat_template(m, tokenize=True)))
        logging.info(f"Tokens par conversation : moyenne {n_tokens.mean():.0f}, max {n_tokens.max()}.")
        df = df[n_tokens <= max_tokens].reset_index(drop=True)
    n_kept = len(df)

    # D. Séparation train / validation / test
    split_names = ["train", "validation", "test"]
    if "dataset_split" in df.columns and df["dataset_split"].isin(split_names).all():
        # Le split préparé avec le dataset (par concept) est conservé
        logging.info("Utilisation de la colonne dataset_split.")
        split = df["dataset_split"]
    else:
        # Sinon, split par concept : toutes les variantes d'un concept restent du même côté,
        # pour qu'aucune paraphrase d'une question d'entraînement ne se retrouve dans le test
        logging.info(f"Split par concept (validation={validation_size}, test={test_size}, seed={seed}).")
        group_column = "concept" if "concept" in df.columns else "instruction"
        groups = sorted(df[group_column].unique())
        random.Random(seed).shuffle(groups)
        n_test = max(1, round(len(groups) * test_size))
        n_validation = max(1, round(len(groups) * validation_size))
        group_split = {group: "train" for group in groups}
        group_split.update({group: "test" for group in groups[:n_test]})
        group_split.update({group: "validation" for group in groups[n_test : n_test + n_validation]})
        split = df[group_column].map(group_split)

    # E. Sauvegarde dans les OutputPaths de KFP
    # Les messages sont sérialisés en JSON, sinon le CSV contient une représentation illisible
    for column in metadata_columns:
        if column not in df.columns:
            df[column] = ""
    df["messages"] = df["messages"].map(json.dumps)
    output_paths = {
        "train": train_output_path,
        "validation": validation_output_path,
        "test": test_output_path,
    }
    for name, path in output_paths.items():
        subset = df.loc[split == name, ["messages", *metadata_columns]]
        subset.to_csv(path, index=False)
        logging.info(f"{len(subset)} lignes ({name}).")
        metrics.log_metric(f"{name}_rows", len(subset))

    metrics.log_metric("raw_rows", n_raw)
    metrics.log_metric("well_formed_rows", n_well_formed)
    metrics.log_metric("deduplicated_rows", n_deduplicated)
    metrics.log_metric("kept_rows", n_kept)

    logging.info("Succès ! Composant terminé.")
