from src.taxonomy import (
    LEVELS,
    SECTORS,
    THEMES,
    build_instruction,
    format_target_response,
    parse_sections,
    variants_for_concept,
)


def test_six_themes_of_twenty_unique_concepts():
    assert len(THEMES) == 6
    assert all(len(concepts) == 20 for concepts in THEMES.values())
    all_concepts = [concept for concepts in THEMES.values() for concept in concepts]
    assert len(set(all_concepts)) == 120


def test_build_instruction_matches_the_brief():
    assert (
        build_instruction("What are Scope 3 emissions?", "beginner", "coffee shop")
        == "I run a small coffee shop. What are Scope 3 emissions? Explain for a beginner."
    )


def test_format_then_parse_round_trip():
    response = format_target_response("A  definition.", "An example.", "An action.", "A check?")

    assert response.splitlines()[0] == "Definition: A definition."
    assert parse_sections(response) == {
        "Definition": "A definition.",
        "Example": "An example.",
        "Action": "An action.",
        "Check": "A check?",
    }


def test_parse_sections_accepts_bold_labels_and_ignores_plain_text():
    assert parse_sections("**Definition:** A.\n**Check:** B?") == {"Definition": "A.", "Check": "B?"}
    assert parse_sections("Scope 3 emissions are indirect emissions.") == {}


def test_variants_cover_all_levels_and_sectors_evenly():
    pairs = [pair for concept_index in range(120) for pair in variants_for_concept(concept_index)]

    assert len(pairs) == 600
    assert set(pairs) == {(level, sector) for level in LEVELS for sector in SECTORS}
    assert all(pairs.count(pair) == 50 for pair in set(pairs))
