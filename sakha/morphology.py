"""Генератор якутских словоформ: аффиксы с 4-вариантным сингармонизмом
и ассимиляцией начального согласного аффикса к концу основы."""
from __future__ import annotations

from .phonology import VOWELS, normalize, suffix_vowel

VOICELESS = set("кпстхчф")
NASALS = set("мнҥ")


def _last(stem: str) -> str:
    return stem[-1]


def plural(stem: str) -> str:
    """-лар/-лэр/-лор/-лөр, -тар..., -дар..., -нар...
    оҕо→оҕолор, ат→аттар, күн→күннэр, хаар→хаардар."""
    irregular = {"кыыс": "кыргыттар", "уол": "уолаттар"}
    if stem in irregular:
        return irregular[stem]
    c = _last(stem)
    if c in VOWELS or c == "л":
        k = "л"
    elif c in "рй":
        k = "д"
    elif c in VOICELESS:
        k = "т"
    elif c in NASALS:
        k = "н"
    else:
        k = "л"
    v = suffix_vowel(stem, high=False)
    return stem + k + v + "р"


def dative(stem: str) -> str:
    """-ҕа после гласной, -га после л/р/й, -ка после глухих (ат→акка),
    -ха после х (ынах→ынахха), -ҥа после носовых (күн→күҥҥэ)."""
    c = _last(stem)
    v = suffix_vowel(stem, high=False)
    if c in VOWELS:
        return stem + "ҕ" + v
    if c in "лрй":
        return stem + "г" + v
    if c == "х":
        return stem + "х" + v
    if c in "тп":
        return stem[:-1] + "кк" + v
    if c in VOICELESS:
        return stem + "к" + v
    if c == "н":
        return stem[:-1] + "ҥҥ" + v
    if c in NASALS:
        return stem + "ҥ" + v
    return stem + "г" + v


def accusative(stem: str) -> str:
    """-ны/-ни/-ну/-нү после гласной, -ы/-и/-у/-ү после согласной."""
    v = suffix_vowel(stem, high=True)
    if _last(stem) in VOWELS:
        return stem + "н" + v
    # озвончение между гласными: балык→балыгы, ынах→ынаҕы, тап→табы
    voicing = {"к": "г", "х": "ҕ", "п": "б"}
    c = _last(stem)
    if c in voicing:
        stem = stem[:-1] + voicing[c]
    return stem + v


def ablative(stem: str) -> str:
    """-ттан после гласной, -тан после согласной (оҕоттон, мастан)."""
    v = suffix_vowel(stem, high=False)
    if _last(stem) in VOWELS:
        return stem + "тт" + v + "н"
    return stem + "т" + v + "н"


def past3(verb: str) -> str:
    """Прошедшее время 3 л. ед. ч.: бар→барда, кэл→кэллэ, көр→көрдө."""
    c = _last(verb)
    v = suffix_vowel(verb, high=False)
    if c in VOWELS or c in VOICELESS:
        k = "т"
    elif c == "л":
        k = "л"
    elif c in NASALS:
        k = "н"
    else:
        k = "д"
    return verb + k + v


def past1(verb: str) -> str:
    """Прошедшее 1 л. ед. ч.: бардым, кэллим, көрдүм."""
    stem = past3(verb)[:-1]
    return stem + suffix_vowel(verb, high=True) + "м"


CASES = {
    "nom": lambda s: s,
    "pl": plural,
    "dat": dative,
    "acc": accusative,
    "abl": ablative,
}


def paradigm(stem: str) -> dict[str, str]:
    stem = normalize(stem)
    return {name: f(stem) for name, f in CASES.items()}
