"""Multilingual fixture text: English, French and non-Latin scripts (SPEC 13.2)."""

from __future__ import annotations

import random

SAMPLES: dict[str, str] = {
    "en": "Karim leads the installation planning of the solar project since March 2026.",
    "fr": "Karim dirige la planification de l'installation du projet solaire depuis mars 2026.",
    "fr_accents": "Émilie préfère les réponses courtes et déteste les réunions tôt le matin.",
    "ru": "Карим руководит планированием установки солнечного проекта с марта 2026 года.",
    "ar": "كريم يقود تخطيط تركيب المشروع الشمسي منذ مارس 2026.",
    "zh": "卡里姆自2026年3月起负责太阳能项目的安装规划。",
    "ja": "カリムは2026年3月から太陽光プロジェクトの設置計画を率いている。",
    "mixed": "Karim (كريم) said: « le projet Solar » starts in März — 3月.",
}

CANARY = "CANARY_zebra_7731_octopus"


def people(n: int) -> list[str]:
    first = ["Karim", "Émilie", "Nadia", "Youssef", "Léa", "Иван", "李雷", "Aiko", "Omar", "Sofia"]
    return [f"{first[i % len(first)]} {i // len(first):02d}" for i in range(n)]


def statements(n: int, *, seed: int = 7) -> list[str]:
    """Deterministic synthetic facts mixing the three scripts."""
    rng = random.Random(seed)
    subjects = people(50)
    verbs_en = ["works on", "moved to", "prefers", "manages", "dislikes", "studies"]
    objects_en = [
        "the solar project",
        "Lyon",
        "short answers",
        "the budget",
        "early meetings",
        "physics",
    ]
    verbs_fr = ["travaille sur", "a déménagé à", "préfère", "gère", "déteste", "étudie"]
    objects_fr = [
        "le projet solaire",
        "Lyon",
        "les réponses courtes",
        "le budget",
        "les réunions tôt",
        "la physique",
    ]
    verbs_ru = ["работает над", "переехал в", "предпочитает", "управляет", "не любит", "изучает"]
    objects_ru = [
        "солнечным проектом",
        "Лион",
        "короткие ответы",
        "бюджетом",
        "ранние встречи",
        "физику",
    ]
    out = []
    for i in range(n):
        subject = subjects[i % len(subjects)]
        k = rng.randrange(6)
        lang = rng.choice(("en", "en", "fr", "ru"))
        if lang == "en":
            out.append(f"{subject} {verbs_en[k]} {objects_en[k]} (note {i}).")
        elif lang == "fr":
            out.append(f"{subject} {verbs_fr[k]} {objects_fr[k]} (note {i}).")
        else:
            out.append(f"{subject} {verbs_ru[k]} {objects_ru[k]} (заметка {i}).")
    return out
