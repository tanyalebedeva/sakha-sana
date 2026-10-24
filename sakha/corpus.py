"""Генерация синтетического речевого корпуса: фразы по шаблонам
(с морфологией) × случайные дикторы × акустическая аугментация."""
from __future__ import annotations

import numpy as np

from . import lexicon as L
from .morphology import CASES, past1, past3
from .phonology import normalize
from .synth import SR, Speaker, synthesize

# лексемы, которые модель НИКОГДА не услышит при обучении
HELDOUT_NOUNS = ["куобах", "сүрэх", "кинигэ", "өрүс", "былыт", "нэһилиэк", "чыычаах", "доҕор"]
HELDOUT_ADJ = ["чэпчэки", "күөх"]
HELDOUT_VERBS = ["суруй"]


def _pick(rng, d):
    keys = list(d)
    return keys[rng.integers(len(keys))]


def make_sentence(rng: np.random.Generator, nouns, adjs, verbs) -> str:
    n = lambda: nouns[rng.integers(len(nouns))]
    a = lambda: adjs[rng.integers(len(adjs))]
    v = lambda: verbs[rng.integers(len(verbs))]
    case = lambda: list(CASES)[rng.integers(len(CASES))]
    t = rng.integers(7)
    if t == 0:
        return CASES[case()](n())
    if t == 1:
        return f"{a()} {CASES[case()](n())}"
    if t == 2:
        return f"{CASES['pl' if rng.random() < .4 else 'nom'](n())} {past3(v())}"
    if t == 3:
        return f"мин {CASES[['dat', 'acc', 'abl'][rng.integers(3)]](n())} {past1(v())}"
    if t == 4:
        return f"{_pick(rng, L.NUMBERS)} {n()}"
    if t == 5:
        return f"{a()} {n()} {past3(v())}"
    return _pick(rng, L.PHRASES) if rng.random() < .5 else f"{_pick(rng, L.PRONOUNS)} {past3(v())}"


# ---------------- псевдослова по фонотактике ----------------
ONSET0 = ["", "б", "к", "т", "с", "х", "д", "м", "н", "ч", "дь", "нь", "й", "һ"]
ONSET = ["б", "к", "т", "с", "х", "д", "м", "н", "ч", "дь", "нь", "й", "һ", "ҕ", "ҥ", "р", "л", "г"]
CODA = ["", "", "", "т", "к", "с", "х", "р", "л", "н", "м", "ҥ", "й", "п"]
HARM_NEXT = {  # допустимые гласные следующего слога (закон гармонии)
    "а": "аы", "ы": "аы", "о": "оу", "у": "уа", "э": "эи", "и": "эи", "ө": "өү", "ү": "үэ",
}
LONG = {"а": ["аа", "ыа"], "ы": ["ыы", "ыа"], "о": ["оо", "уо"], "у": ["уу", "уо"],
        "э": ["ээ", "иэ"], "и": ["ии", "иэ"], "ө": ["өө", "үө"], "ү": ["үү", "үө"]}


def pseudo_word(rng: np.random.Generator) -> str:
    """Несуществующее, но фонотактически возможное якутское слово."""
    n_syl = rng.choice([1, 2, 2, 3])
    v = "аыоуэиөү"[rng.integers(8)]
    w = ""
    for k in range(n_syl):
        on = ONSET0 if k == 0 else ONSET
        w += on[rng.integers(len(on))]
        nuc = v
        if rng.random() < 0.25:
            opts = LONG[v]
            nuc = opts[rng.integers(len(opts))] if k == 0 else opts[0]
        w += nuc
        if k == n_syl - 1 or rng.random() < 0.35:
            w += CODA[rng.integers(len(CODA))]
        nxt = HARM_NEXT[v]
        v = nxt[rng.integers(len(nxt))]
    return w


HELDOUT_ALL = set(HELDOUT_NOUNS + HELDOUT_ADJ + HELDOUT_VERBS)


def pseudo_sentence(rng: np.random.Generator, nouns) -> str:
    words = []
    for _ in range(rng.choice([1, 2, 2, 3])):
        if rng.random() < 0.2:
            stem = nouns[rng.integers(len(nouns))]
        else:
            stem = pseudo_word(rng)
            while any(stem.startswith(h) for h in HELDOUT_ALL):
                stem = pseudo_word(rng)
        words.append(CASES[list(CASES)[rng.integers(len(CASES))]](stem))
    return " ".join(words)


def split_vocab(heldout: bool):
    if heldout:
        return HELDOUT_NOUNS, HELDOUT_ADJ, HELDOUT_VERBS
    nouns = [w for w in L.NOUNS if w not in HELDOUT_NOUNS]
    adjs = [w for w in L.ADJECTIVES if w not in HELDOUT_ADJ]
    verbs = [w for w in L.VERBS if w not in HELDOUT_VERBS]
    return nouns, adjs, verbs


# ---------------- аугментация ----------------

def augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = y.astype(np.float64)
    if rng.random() < 0.5:                       # реверберация: экспоненциальный хвост
        rt = rng.uniform(0.05, 0.4)
        n = int(rt * SR)
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir[0] = 1.0
        y = np.convolve(y, ir * rng.uniform(0.05, 0.3) / np.sqrt(n) * 10, mode="full")[: len(y)] + y
    if rng.random() < 0.8:                       # шум (белый или «розовый»)
        snr = rng.uniform(5, 40)
        nz = rng.standard_normal(len(y))
        if rng.random() < 0.5:
            nz = np.cumsum(nz); nz -= np.convolve(nz, np.ones(200) / 200, mode="same")
        p_s = np.mean(y ** 2) + 1e-9
        nz *= np.sqrt(p_s / (np.mean(nz ** 2) + 1e-9) / 10 ** (snr / 10))
        y = y + nz
    if rng.random() < 0.3:                       # телефонная полоса
        from scipy.signal import butter, sosfilt
        y = sosfilt(butter(4, [300, 3400], "band", fs=SR, output="sos"), y)
    y *= rng.uniform(0.3, 1.0) / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


# ---------------- признаки ----------------

def _mel_fb(n_mels=40, n_fft=512, fmin=60, fmax=7800):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


MEL_FB = _mel_fb()


def logmel(y: np.ndarray, win=400, hop=160, n_fft=512) -> np.ndarray:
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None] + hop * np.arange(n)[:, None]
    frames = y[idx] * np.hamming(win)
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    m = np.log(spec @ MEL_FB.T + 1e-6)
    m = (m - m.mean(0)) / (m.std(0) + 1e-5)     # CMVN по высказыванию
    return m.astype(np.float32)


# ---------------- алфавит для CTC ----------------
CHARS = " абвгдеёжзийклмнопрстуфхцчшщыьэюяҕҥөһү"
BLANK = 0
C2I = {c: i + 1 for i, c in enumerate(CHARS)}
I2C = {i: c for c, i in C2I.items()}
VOCAB = len(CHARS) + 1


def encode(text: str) -> np.ndarray:
    return np.array([C2I[c] for c in normalize(text) if c in C2I], dtype=np.int32)


def decode(ids) -> str:
    return "".join(I2C.get(int(i), "") for i in ids)


def make_item(args):
    seed, heldout, aug, pseudo = args
    rng = np.random.default_rng(seed)
    vocab = split_vocab(heldout)
    if pseudo > 0 and rng.random() < pseudo:
        text = pseudo_sentence(rng, vocab[0])
    else:
        text = make_sentence(rng, *vocab)
    spk = Speaker.random(rng, sid=seed)
    y = synthesize(text, spk)
    if aug:
        y = augment(y, rng)
    return normalize(text), logmel(y)


def build(n: int, seed0: int, heldout: bool = False, aug: bool = True, procs: int = 2, pseudo: float = 0.0):
    from multiprocessing import Pool
    args = [(seed0 + i, heldout, aug, pseudo) for i in range(n)]
    with Pool(procs) as p:
        items = p.map(make_item, args, chunksize=64)
    texts = [t for t, _ in items]
    feats = [f for _, f in items]
    return texts, feats
