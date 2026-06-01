"""
Dataset d'évaluation du chatbot portfolio.

Contient des paires (question, réponse_attendue) couvrant les différents
types de questions qu'un visiteur pourrait poser. Ce dataset est utilisé
par le script d'évaluation automatique (evaluate.py).

NOTE : Les réponses attendues doivent être mises à jour après avoir rempli
data/contexte.txt avec les vraies informations du portfolio.
"""

EVALUATION_DATASET = [
    # --- Identité ---
    {
        "question": "Qui est Romain Sebire ?",
        "expected": "Romain Sebire est un développeur / ingénieur (à compléter avec les vraies infos).",
        "category": "identité",
    },
    {
        "question": "What is Romain's website?",
        "expected": "Romain's website is https://romainsebire.fr",
        "category": "identité",
        "lang": "en",
    },
    # --- Formation ---
    {
        "question": "Quelle est la formation de Romain ?",
        "expected": "À compléter avec les vraies formations après remplissage de contexte.txt.",
        "category": "formation",
    },
    {
        "question": "Où Romain a-t-il étudié ?",
        "expected": "À compléter.",
        "category": "formation",
    },
    # --- Compétences ---
    {
        "question": "Quelles sont les compétences techniques de Romain ?",
        "expected": "À compléter (langages, frameworks, outils, cloud...).",
        "category": "compétences",
    },
    {
        "question": "Romain connaît-il Python ?",
        "expected": "À compléter.",
        "category": "compétences",
    },
    {
        "question": "What programming languages does Romain know?",
        "expected": "To be completed.",
        "category": "compétences",
        "lang": "en",
    },
    # --- Projets ---
    {
        "question": "Quels projets Romain a-t-il réalisés ?",
        "expected": "À compléter avec les projets listés dans contexte.txt.",
        "category": "projets",
    },
    {
        "question": "Parle-moi du projet de fin d'études de Romain.",
        "expected": "À compléter avec les détails du PFE.",
        "category": "projets",
    },
    # --- Expériences ---
    {
        "question": "Quelles sont les expériences professionnelles de Romain ?",
        "expected": "À compléter.",
        "category": "expériences",
    },
    {
        "question": "Romain a-t-il fait des stages ?",
        "expected": "À compléter.",
        "category": "expériences",
    },
    # --- Hors-sujet (l'agent doit refuser) ---
    {
        "question": "Quelle est la capitale de la France ?",
        "expected": "L'agent doit indiquer qu'il ne répond qu'aux questions sur Romain.",
        "category": "hors-sujet",
    },
    {
        "question": "Écris-moi un script Python.",
        "expected": "L'agent doit refuser de générer du code.",
        "category": "hors-sujet",
    },
    # --- Langues / Centres d'intérêt ---
    {
        "question": "Quelles langues parle Romain ?",
        "expected": "À compléter.",
        "category": "langues",
    },
    {
        "question": "Quels sont les centres d'intérêt de Romain ?",
        "expected": "À compléter.",
        "category": "centres-intérêt",
    },
]
