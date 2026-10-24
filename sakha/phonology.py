"""Якутская (саха) фонология: графема → фонема, слоги, ударение, сингармонизм.

Якутская орфография почти фонематична, поэтому G2P строится на правилах:
долгие гласные пишутся удвоением (аа, ыы...), есть четыре дифтонга
(ыа, иэ, уо, үө) и особые буквы ҕ, ҥ, һ, ө, ү, диграфы дь и нь.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VOWELS = "аыоуэиөү"
# гласная → (задний ряд?, огублённая?, высокая?)
VOWEL_FEATS = {
    "а": (True, False, False), "ы": (True, False, True),
    "о": (True, True, False),  "у": (True, True, True),
    "э": (False, False, False), "и": (False, False, True),
    "ө": (False, True, False),  "ү": (False, True, True),
}
DIPHTHONGS = {"ыа": "ɯa", "иэ": "ie", "уо": "uo", "үө": "yø"}
VOWEL_IPA = {"а": "a", "ы": "ɯ", "о": "o", "у": "u", "э": "e", "и": "i", "ө": "ø", "ү": "y"}

CONS_IPA = {
    "б": "b", "п": "p", "т": "t", "д": "d", "к": "k", "г": "g", "х": "q",
    "ҕ": "ɣ", "с": "s", "һ": "h", "м": "m", "н": "n", "ҥ": "ŋ", "л": "l",
    "р": "r", "й": "j", "ч": "c", "дь": "ɟ", "нь": "ɲ",
    # заимствования из русского
    "в": "v", "ф": "f", "з": "z", "ж": "ʒ", "ш": "ʃ", "щ": "ʃ", "ц": "ts",
}
LOAN_VOWELS = {"е": ["j", "e"], "ё": ["j", "o"], "ю": ["j", "u"], "я": ["j", "a"]}

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяҕҥөһү"


@dataclass
class Phone:
    sym: str          # фонема (IPA)
    long: bool = False
    stress: bool = False

    @property
    def is_vowel(self) -> bool:
        return self.sym in VOWEL_IPA.values() or self.sym in DIPHTHONGS.values()

    def __str__(self) -> str:
        return ("ˈ" if self.stress else "") + self.sym + ("ː" if self.long else "")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "ё")
    # латинские двойники, которые часто встречаются в якутских текстах
    text = (text.replace("h", "һ").replace("ö", "ө").replace("ÿ", "ү")
                .replace("ҕ", "ҕ").replace("ӈ", "ҥ").replace("ң", "ҥ"))
    text = re.sub(f"[^{ALPHABET} -]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def g2p_word(word: str) -> list[Phone]:
    w = normalize(word).replace(" ", "")
    out: list[Phone] = []
    i = 0
    while i < len(w):
        two = w[i:i + 2]
        if two in DIPHTHONGS:
            out.append(Phone(DIPHTHONGS[two], long=True)); i += 2; continue
        if len(two) == 2 and two[0] == two[1] and two[0] in VOWELS:
            out.append(Phone(VOWEL_IPA[two[0]], long=True)); i += 2; continue
        if two in ("дь", "нь"):
            out.append(Phone(CONS_IPA[two])); i += 2; continue
        ch = w[i]
        if ch in VOWEL_IPA:
            out.append(Phone(VOWEL_IPA[ch]))
        elif ch in LOAN_VOWELS:
            j, v = LOAN_VOWELS[ch]
            if not out or out[-1].is_vowel:
                out.append(Phone(j))
            out.append(Phone(v))
        elif ch in CONS_IPA:
            # удвоенный согласный → долгий (геминат)
            if out and out[-1].sym == CONS_IPA[ch] and not out[-1].long:
                out[-1].long = True
            else:
                out.append(Phone(CONS_IPA[ch]))
        i += 1
    # ударение в якутском - на последнем слоге
    for ph in reversed(out):
        if ph.is_vowel:
            ph.stress = True
            break
    return out


def g2p(text: str) -> list[list[Phone]]:
    return [g2p_word(w) for w in normalize(text).replace("-", " ").split()]


def to_ipa(text: str) -> str:
    return " ".join("".join(map(str, w)) for w in g2p(text))


def syllabify(word: str) -> list[str]:
    """Слогоделение: один согласный между гласными уходит к следующему слогу,
    из группы согласных к следующему слогу уходит последний (якутский не
    допускает стечения согласных в начале слога)."""
    w = normalize(word)
    units, i = [], 0
    while i < len(w):
        two = w[i:i + 2]
        if two in DIPHTHONGS or (len(two) == 2 and two[0] == two[1] and two[0] in VOWELS) or two in ("дь", "нь"):
            units.append(two); i += 2
        else:
            units.append(w[i]); i += 1
    is_v = [u[0] in VOWELS for u in units]
    vidx = [k for k, v in enumerate(is_v) if v]
    if len(vidx) <= 1:
        return [w]
    cuts = []
    for a, b in zip(vidx, vidx[1:]):
        n_cons = b - a - 1
        cuts.append(b if n_cons == 0 else b - 1)
    sylls, start = [], 0
    for c in cuts:
        sylls.append("".join(units[start:c])); start = c
    sylls.append("".join(units[start:]))
    return sylls


# ---------- сингармонизм ----------

def vowel_seq(word: str) -> list[str]:
    """Последовательность гласных; дифтонг и долгая гласная - одна единица."""
    w = normalize(word)
    seq, i = [], 0
    while i < len(w):
        two = w[i:i + 2]
        if two in DIPHTHONGS:
            seq.append(two); i += 2
        elif w[i] in VOWELS:
            seq.append(w[i]); i += 2 if two == w[i] * 2 else 1
        else:
            i += 1
    return seq


def _feats(v: str):
    # дифтонг ведёт себя по первому (высокому) элементу для ряда и огублённости
    return VOWEL_FEATS[v[0]]


def harmony_violations(word: str) -> list[tuple[str, str, str]]:
    """Проверка гармонии гласных. Правила:
    ряд: все гласные слова одного ряда;
    огублённость: после о/ө - только огублённые; после у/ү - высокие огублённые
    или низкие неогублённые (кулун, уҥа); после неогублённых - неогублённые."""
    seq = vowel_seq(word)
    bad = []
    for prev, cur in zip(seq, seq[1:]):
        pb, pr, ph = _feats(prev)
        lb, lr, lh = pb, pr, ph  # дифтонг уо/үө ведёт себя как у/ү (суол → суоллар)
        cb, cr, ch = _feats(cur)
        if cb != pb:
            bad.append((prev, cur, "ряд"))
            continue
        if not lr and cr:
            bad.append((prev, cur, "огублённость"))
        elif lr and not lh and not cr:
            bad.append((prev, cur, "огублённость"))
        elif lr and lh and not ch and cr:
            bad.append((prev, cur, "огублённость"))
    return bad


def suffix_vowel(stem: str, high: bool) -> str:
    """Гласная аффикса по последней гласной основы.
    high=True → 4 варианта ы/и/у/ү; high=False → а/э/о/ө."""
    seq = vowel_seq(stem)
    if not seq:
        return "ы" if high else "а"
    v = seq[-1]
    first = v[0]
    back, rnd, hi = VOWEL_FEATS[first]
    if high:
        return {(True, False): "ы", (False, False): "и", (True, True): "у", (False, True): "ү"}[(back, rnd)]
    if len(v) == 2 and v in DIPHTHONGS:      # уо→а, үө→э, ыа→а, иэ→э
        return "а" if back else "э"
    if rnd and not hi:                         # о→о, ө→ө
        return "о" if back else "ө"
    return "а" if back else "э"               # а,ы,у→а ; э,и,ү→э
