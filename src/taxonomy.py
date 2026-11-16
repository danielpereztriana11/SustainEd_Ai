"""Thèmes, niveaux, profils d'entreprise et format de réponse de SustainEd AI.

Ce module est partagé par la génération du dataset (scripts/build_dataset.py) et l'app Chainlit :
la question posée au modèle a donc exactement la même forme à l'entraînement et en production.
Les composants Kubeflow et src/handler.py doivent rester autonomes et ne l'importent pas.
"""

import re

# Niveau -> formulation dans la question de l'utilisateur
LEVELS = {
    "beginner": "a beginner",
    "intermediate": "an intermediate learner",
    "advanced": "an advanced learner",
}

# Profil -> formulation dans la question de l'utilisateur
SECTORS = {
    "coffee shop": "I run a small coffee shop.",
    "online clothing store": "I run a small online clothing store.",
    "furniture workshop": "I run a small furniture workshop.",
    "IT consulting firm": "I run a small IT consulting firm.",
}

# Les quatre parties de chaque réponse, dans l'ordre
SECTIONS = ["Definition", "Example", "Action", "Check"]

# 6 thèmes x 20 concepts = 120 concepts, 5 variantes (niveau, profil) par concept = 600 exemples
THEMES = {
    "sustainability fundamentals and ESG": [
        "sustainability",
        "sustainable development",
        "ESG (environmental, social and governance)",
        "the triple bottom line",
        "the UN Sustainable Development Goals",
        "stakeholders",
        "materiality",
        "double materiality",
        "corporate social responsibility",
        "the difference between ESG and CSR",
        "natural capital",
        "biodiversity",
        "planetary boundaries",
        "a just transition",
        "sustainable finance",
        "ESG ratings",
        "corporate governance",
        "social sustainability",
        "human rights due diligence",
        "the business case for sustainability",
    ],
    "climate and Scope 1/2/3": [
        "greenhouse gases",
        "a carbon footprint",
        "Scope 1 emissions",
        "Scope 2 emissions",
        "Scope 3 emissions",
        "the difference between Scope 2 and Scope 3 emissions",
        "upstream and downstream emissions",
        "CO2 equivalent (CO2e)",
        "emission factors",
        "activity data",
        "location-based and market-based Scope 2 accounting",
        "a base year for emissions",
        "net zero",
        "carbon neutrality",
        "carbon offsets",
        "science-based targets",
        "climate change mitigation and adaptation",
        "climate risk for a business",
        "emissions intensity",
        "a greenhouse gas inventory",
    ],
    "circular economy": [
        "the circular economy",
        "the linear economy",
        "the waste hierarchy",
        "reuse",
        "repair",
        "refurbishment and remanufacturing",
        "recycling",
        "the difference between upcycling and downcycling",
        "designing for durability",
        "product-as-a-service",
        "extended producer responsibility",
        "life cycle thinking",
        "a life cycle assessment",
        "eco-design",
        "resource efficiency",
        "industrial symbiosis",
        "biodegradable and compostable materials",
        "packaging reduction",
        "reverse logistics",
        "zero waste",
    ],
    "sustainable business operations": [
        "energy efficiency",
        "an energy audit",
        "renewable energy",
        "a power purchase agreement",
        "water stewardship",
        "waste management",
        "sustainable procurement",
        "a supplier code of conduct",
        "green logistics",
        "sustainable business travel",
        "an environmental management system",
        "ISO 14001",
        "sustainable packaging",
        "food waste reduction",
        "employee engagement in sustainability",
        "green IT",
        "efficient heating and cooling",
        "sustainable sourcing of raw materials",
        "fair working conditions in the supply chain",
        "an internal carbon price",
    ],
    "sustainability reporting basics": [
        "a sustainability report",
        "non-financial reporting",
        "a materiality assessment",
        "key performance indicators for sustainability",
        "the GRI Standards",
        "the Voluntary SME Standard (VSME)",
        "the Corporate Sustainability Reporting Directive (CSRD)",
        "the ISSB standards",
        "the reporting boundary",
        "data quality in sustainability reporting",
        "assurance of sustainability information",
        "a baseline in sustainability reporting",
        "sustainability targets",
        "transparency",
        "the value chain in sustainability reporting",
        "sustainability data requests from large customers",
        "qualitative and quantitative disclosures",
        "comparability of sustainability data",
        "climate-related financial disclosures",
        "a sustainability policy",
    ],
    "practical SME actions and greenwashing": [
        "greenwashing",
        "vague environmental claims",
        "substantiating a green claim",
        "eco-labels",
        "carbon-neutral product claims",
        "a first sustainability action plan",
        "quick wins for cutting energy use",
        "measuring a first carbon footprint",
        "choosing sustainability priorities",
        "talking to suppliers about sustainability",
        "communicating sustainability to customers",
        "green marketing",
        "greenhushing",
        "selective disclosure",
        "third-party certification",
        "setting a realistic sustainability target",
        "the cost savings of sustainability measures",
        "sustainability training for staff",
        "tracking progress on sustainability",
        "avoiding misleading recycling claims",
    ],
}

# Mots-clés pour rattacher une question du dataset Hugging Face à un thème
THEME_KEYWORDS = {
    "sustainability fundamentals and ESG": [
        "esg", "sustainable development", "triple bottom line", "sdg", "stakeholder",
        "corporate social responsibility", "csr", "biodiversity", "governance",
    ],
    "climate and Scope 1/2/3": [
        "scope 1", "scope 2", "scope 3", "greenhouse", "ghg", "carbon", "emission",
        "net zero", "climate",
    ],
    "circular economy": [
        "circular", "recycl", "reuse", "waste", "life cycle", "lifecycle", "eco-design",
        "remanufactur",
    ],
    "sustainable business operations": [
        "energy efficiency", "renewable", "procurement", "supply chain", "supplier",
        "water", "logistics", "iso 14001",
    ],
    "sustainability reporting basics": [
        "reporting", "disclosure", "gri", "csrd", "vsme", "issb", "materiality", "assurance",
        "indicator",
    ],
    "practical SME actions and greenwashing": [
        "greenwash", "green claim", "eco-label", "ecolabel", "sme", "small business",
        "green marketing", "certification",
    ],
}


def build_instruction(question: str, level: str, sector: str) -> str:
    """Build the user message: profile, question, then requested level.

    >>> build_instruction("What are Scope 3 emissions?", "beginner", "coffee shop")
    'I run a small coffee shop. What are Scope 3 emissions? Explain for a beginner.'
    """
    return f"{SECTORS[sector]} {question.strip()} Explain for {LEVELS[level]}."


def format_target_response(definition: str, example: str, action: str, check: str) -> str:
    """Assemble the four-part answer, one labelled section per line."""
    parts = [definition, example, action, check]
    return "\n".join(
        f"{label}: {' '.join(text.split())}" for label, text in zip(SECTIONS, parts, strict=True)
    )


def parse_sections(response: str) -> dict[str, str]:
    """Return the text of each labelled section found in a response (missing ones are absent)."""
    pattern = r"^\s*\**(" + "|".join(SECTIONS) + r")\**\s*:\**\s*"
    found: dict[str, str] = {}
    current = None
    for line in response.splitlines():
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if match:
            current = match.group(1).capitalize()
            found[current] = line[match.end():].strip()
        elif current and line.strip():
            found[current] = f"{found[current]} {line.strip()}".strip()
    return found


def variants_for_concept(concept_index: int, n_variants: int = 5) -> list[tuple[str, str]]:
    """Pick (level, sector) pairs for a concept, rotating so that all 12 pairs are used evenly."""
    levels, sectors = list(LEVELS), list(SECTORS)
    n_pairs = len(levels) * len(sectors)
    # 3 et 4 sont premiers entre eux : i -> (i % 3, i % 4) parcourt les 12 combinaisons
    return [
        (levels[i % len(levels)], sectors[i % len(sectors)])
        for i in ((concept_index * n_variants + k) % n_pairs for k in range(n_variants))
    ]
