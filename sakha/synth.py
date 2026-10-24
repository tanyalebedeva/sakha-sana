"""Формантный синтезатор якутской речи (source-filter, в духе Клатта).

Конвейер: текст → фонемы (G2P) → сегменты с артикуляторными целями →
покадровые траектории (F0, форманты, амплитуды) → сглаживание (коартикуляция)
→ поблочный синтез резонаторами с переносом состояния фильтров.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .phonology import g2p

SR = 16000
HOP = 80            # 5 мс на кадр
FRAME_S = HOP / SR

# Гласные: F1, F2, F3 (взрослый мужской голос)
VOWEL_F = {
    "a": (750, 1300, 2500), "ɯ": (360, 1450, 2400), "o": (500, 880, 2400),
    "u": (320, 780, 2300), "e": (520, 1850, 2550), "i": (290, 2250, 2950),
    "ø": (430, 1500, 2300), "y": (290, 1700, 2150),
}
DIPH = {"ɯa": ("ɯ", "a"), "ie": ("i", "e"), "uo": ("u", "o"), "yø": ("y", "ø")}

# Согласные: класс, звонкость, (F1,F2,F3) локус/цель, (центр шума, полоса, громкость)
CONS = {
    "p": ("stop", 0, (250, 900, 2300), (1200, 1500, .5)),
    "b": ("stop", 1, (250, 900, 2300), (1200, 1500, .35)),
    "t": ("stop", 0, (250, 1700, 2700), (4000, 2000, .6)),
    "d": ("stop", 1, (250, 1700, 2700), (4000, 2000, .4)),
    "k": ("stop", 0, (250, 1800, 2500), (2200, 1200, .7)),
    "g": ("stop", 1, (250, 1800, 2500), (2200, 1200, .45)),
    "c": ("affr", 0, (280, 2000, 2800), (3200, 1600, .8)),
    "ɟ": ("affr", 1, (280, 2000, 2800), (3200, 1600, .55)),
    "ts": ("affr", 0, (280, 1700, 2700), (5500, 2500, .7)),
    "q": ("fric", 0, (450, 1100, 2400), (1300, 1000, .55)),
    "ɣ": ("vfric", 1, (420, 1250, 2400), (1300, 1000, .12)),
    "s": ("fric", 0, (300, 1700, 2700), (6000, 2200, .85)),
    "ʃ": ("fric", 0, (300, 1900, 2600), (3300, 1500, .8)),
    "f": ("fric", 0, (300, 1000, 2300), (6500, 4000, .25)),
    "h": ("asp", 0, None, (0, 0, .35)),
    "z": ("vfric", 1, (300, 1700, 2700), (6000, 2200, .45)),
    "ʒ": ("vfric", 1, (300, 1900, 2600), (3300, 1500, .4)),
    "v": ("vfric", 1, (300, 1100, 2300), (5000, 4000, .1)),
    "m": ("nasal", 1, (260, 1050, 2200), None),
    "n": ("nasal", 1, (260, 1550, 2550), None),
    "ɲ": ("nasal", 1, (260, 2000, 2750), None),
    "ŋ": ("nasal", 1, (260, 1250, 2300), None),
    "l": ("liquid", 1, (360, 1100, 2700), None),
    "r": ("trill", 1, (450, 1300, 1750), None),
    "j": ("glide", 1, (280, 2200, 3000), None),
}


@dataclass
class Speaker:
    """«Виртуальный диктор»: голос задаётся несколькими параметрами."""
    f0: float = 120.0          # базовая частота основного тона, Гц
    vtl: float = 1.0           # масштаб формант (обратно длине тракта)
    tempo: float = 1.0         # >1 — быстрее
    breath: float = 0.04       # придыхание
    jitter: float = 0.01
    shimmer: float = 0.05
    tilt: float = 0.0          # спектральный наклон (0..0.6)
    bw: float = 1.0            # масштаб полос формант
    pause: float = 0.03        # пауза между словами, с
    seed: int = 0

    @staticmethod
    def random(rng: np.random.Generator, sid: int = 0) -> "Speaker":
        female = rng.random() < 0.5
        child = rng.random() < 0.12
        if child:
            f0, vtl = rng.uniform(220, 300), rng.uniform(1.22, 1.35)
        elif female:
            f0, vtl = rng.uniform(165, 250), rng.uniform(1.08, 1.22)
        else:
            f0, vtl = rng.uniform(85, 150), rng.uniform(0.88, 1.03)
        return Speaker(f0=f0, vtl=vtl, tempo=rng.uniform(0.8, 1.25),
                       breath=rng.uniform(0.01, 0.12), jitter=rng.uniform(0.003, 0.025),
                       shimmer=rng.uniform(0.02, 0.12), tilt=rng.uniform(0, 0.6),
                       bw=rng.uniform(0.8, 1.4), pause=rng.uniform(0.0, 0.08), seed=sid)


@dataclass
class Seg:
    kind: str                     # vowel | stop | fric | ...
    dur: float                    # с
    f: tuple                      # целевые F1..F3 (или пара для дифтонга)
    f_end: tuple | None = None
    av: float = 1.0               # амплитуда голоса
    af: float = 0.0               # амплитуда шума
    fric: tuple = (0, 0)          # центр и полоса шума
    stress: bool = False
    trill: bool = False
    nasal: bool = False
    burst: float = 0.0            # энергия взрыва в начале сегмента
    label: str = ""


def _next_vowel_f(word, idx):
    for ph in word[idx + 1:]:
        if ph.sym in VOWEL_F:
            return VOWEL_F[ph.sym]
        if ph.sym in DIPH:
            return VOWEL_F[DIPH[ph.sym][0]]
    for ph in reversed(word[:idx]):
        if ph.sym in VOWEL_F:
            return VOWEL_F[ph.sym]
        if ph.sym in DIPH:
            return VOWEL_F[DIPH[ph.sym][1]]
    return VOWEL_F["a"]


def text_to_segments(text: str, spk: Speaker) -> list[Seg]:
    segs: list[Seg] = []
    words = g2p(text)
    for wi, word in enumerate(words):
        if wi > 0 and spk.pause > 0.005:
            segs.append(Seg("sil", spk.pause, (500, 1500, 2500), av=0, label="_"))
        for i, ph in enumerate(word):
            last_syl = all(p.sym not in VOWEL_F and p.sym not in DIPH for p in word[i + 1:])
            stretch = (1.25 if last_syl else 1.0) / spk.tempo
            s = ph.sym
            if s in VOWEL_F or s in DIPH:
                if s in DIPH:
                    f, fe = VOWEL_F[DIPH[s][0]], VOWEL_F[DIPH[s][1]]
                    dur = 0.18
                else:
                    f, fe = VOWEL_F[s], None
                    dur = 0.165 if ph.long else 0.085
                if ph.stress:
                    dur *= 1.15
                segs.append(Seg("vowel", dur * stretch, f, fe, stress=ph.stress, label=str(ph)))
                continue
            if s not in CONS:
                continue
            kind, voiced, loc, noise = CONS[s]
            vf = _next_vowel_f(word, i)
            gem = 1.9 if ph.long else 1.0
            if loc is None:            # һ — придыхание с формантами соседнего гласного
                segs.append(Seg("asp", 0.07 * gem * stretch, vf, av=0, af=noise[2], fric=(0, 0), label=s))
                continue
            loc = list(loc)
            if s in "kgŋ":            # велярные: локус зависит от ряда гласного
                loc[1] = 2200 if vf[1] > 1600 else 1150
            if s == "q":
                loc[1] = min(loc[1], vf[1])
            loc = tuple(loc)
            if kind == "stop":
                segs.append(Seg("closure", 0.055 * gem * stretch, loc, av=0.12 if voiced else 0.0, label=s))
                segs.append(Seg("release", 0.022 * stretch, loc, av=0.7 if voiced else 0.0,
                                af=noise[2] * 0.5, fric=noise[:2], burst=noise[2], label=s))
                if not voiced:     # короткое придыхание после глухого
                    segs.append(Seg("asp", 0.02 * stretch, vf, av=0, af=0.25, label=s))
            elif kind == "affr":
                segs.append(Seg("closure", 0.045 * gem * stretch, loc, av=0.12 if voiced else 0, label=s))
                segs.append(Seg("fric", 0.075 * stretch, loc, av=0.55 if voiced else 0,
                                af=noise[2], fric=noise[:2], burst=noise[2] * 0.6, label=s))
            elif kind in ("fric", "vfric"):
                segs.append(Seg("fric", 0.095 * gem * stretch, loc, av=0.6 if voiced else 0,
                                af=noise[2], fric=noise[:2], label=s))
            elif kind == "nasal":
                segs.append(Seg("nasal", 0.07 * gem * stretch, loc, av=0.55, nasal=True, label=s))
            elif kind == "liquid":
                segs.append(Seg("liquid", 0.065 * gem * stretch, loc, av=0.75, label=s))
            elif kind == "trill":
                segs.append(Seg("trill", 0.075 * gem * stretch, loc, av=0.75, trill=True, label=s))
            elif kind == "glide":
                segs.append(Seg("glide", 0.055 * stretch, loc, av=0.8, label=s))
    return segs


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    pad = np.pad(x, ((n, n),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(pad, k, mode="same")[n:-n]
    return np.stack([np.convolve(pad[:, j], k, mode="same") for j in range(x.shape[1])], 1)[n:-n]


def _resonator(f, bw):
    r = np.exp(-np.pi * bw / SR)
    a1 = 2 * r * np.cos(2 * np.pi * f / SR)
    a2 = -r * r
    return 1 - a1 - a2, a1, a2


def synthesize(text: str, spk: Speaker | None = None, return_tracks: bool = False):
    spk = spk or Speaker()
    rng = np.random.default_rng(spk.seed)
    segs = [Seg("sil", 0.08, (500, 1500, 2500), av=0, label="_")]
    segs += text_to_segments(text, spk)
    segs.append(Seg("sil", 0.1, (500, 1500, 2500), av=0, label="_"))

    # ---- покадровые траектории ----
    F, AV, AF, FC, FB, BURST, TR, NAS, STR = ([] for _ in range(9))
    for sg in segs:
        n = max(1, int(round(sg.dur / FRAME_S)))
        t = np.linspace(0, 1, n)[:, None]
        f0 = np.array(sg.f, float)
        f1 = np.array(sg.f_end if sg.f_end else sg.f, float)
        F.append(f0 * (1 - t) + f1 * t)
        AV += [sg.av] * n
        AF += [sg.af] * n
        FC += [sg.fric[0]] * n
        FB += [max(sg.fric[1], 1)] * n
        b = np.zeros(n); b[: min(2, n)] = sg.burst
        BURST.append(b)
        TR += [sg.trill] * n
        NAS += [sg.nasal] * n
        STR += [1.0 if sg.stress else 0.0] * n
    F = np.concatenate(F) * spk.vtl
    AV, AF, FC, FB = map(np.array, (AV, AF, FC, FB))
    BURST = np.concatenate(BURST)
    TR, NAS, STR = np.array(TR), np.array(NAS), np.array(STR)
    nfr = len(AV)

    F = _smooth(F, 7)                      # коартикуляция ~35 мс
    AV = _smooth(AV, 3)
    AF = _smooth(AF, 2)
    # дрожащее р: модуляция амплитуды ~28 Гц
    tt = np.arange(nfr) * FRAME_S
    AV = AV * np.where(TR, 0.55 + 0.45 * np.cos(2 * np.pi * 28 * tt), 1.0)

    # ---- основной тон: деклинация + подъём на ударных слогах ----
    prog = np.linspace(0, 1, nfr)
    f0 = spk.f0 * (1.12 - 0.27 * prog) * (1 + 0.13 * _smooth(STR, 15))
    f0 *= 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * tt + rng.uniform(0, 6))
    f0 = f0 * (1 + spk.jitter * rng.standard_normal(nfr))

    # ---- источник: глоттальный импульс Розенберга ----
    nsamp = nfr * HOP
    xs = (np.arange(nsamp) + 0.5) / HOP
    f0s = np.interp(xs, np.arange(nfr), f0)
    phase = np.cumsum(f0s / SR) % 1.0
    tp, tn = 0.4, 0.16
    g = np.where(phase < tp, 0.5 * (1 - np.cos(np.pi * phase / tp)),
                 np.where(phase < tp + tn, np.cos(0.5 * np.pi * (phase - tp) / tn), 0.0))
    period_id = np.cumsum(np.diff(phase, prepend=phase[0]) < 0)
    shim = 1 + spk.shimmer * rng.standard_normal(period_id.max() + 1)
    voice = np.diff(g, prepend=0.0) * shim[period_id] * 8.0      # излучение губ ≈ производная
    if spk.tilt > 0:
        voice = lfilter([1 - spk.tilt], [1, -spk.tilt], voice)
    avs = np.interp(xs, np.arange(nfr), AV)
    afs = np.interp(xs, np.arange(nfr), AF)
    noise = rng.standard_normal(nsamp)
    # придыхание модулируется глоттальным циклом
    asp = noise * spk.breath * (0.4 + g) * avs
    glottal = voice * avs + asp

    # ---- поблочная фильтрация ----
    out_v = np.zeros(nsamp)
    out_f = np.zeros(nsamp)
    zv = [np.zeros(2) for _ in range(5)]
    zf = np.zeros((2, 2))
    fric_src = rng.standard_normal(nsamp)
    for k in range(nfr):
        sl = slice(k * HOP, (k + 1) * HOP)
        x = glottal[sl]
        is_asp = AV[k] < 0.05 and AF[k] > 0.05 and FC[k] < 1
        if is_asp:                          # һ и придыхание идут через тракт
            x = x + fric_src[sl] * AF[k] * 0.5
        f1, f2, f3 = F[k]
        bws = (np.array([80, 100, 140, 220, 300]) * spk.bw)
        if NAS[k]:
            bws[0] *= 1.6; bws[1] *= 2.0
        forms = (f1, f2, f3, 3500 * spk.vtl, 4400 * spk.vtl)
        y = x
        for j, (ff, bb) in enumerate(zip(forms, bws)):
            ff = min(ff, SR / 2 - 400)
            b0, a1, a2 = _resonator(ff, bb)
            y, zv[j] = _lf(b0, a1, a2, y, zv[j])
        if NAS[k]:                          # носовой: приглушённый, с антирезонансом ~ F2
            y = y * 0.6
        out_v[sl] = y
        if FC[k] > 1 and not is_asp:
            sos = _bandpass(FC[k] * min(spk.vtl, 1.15), FB[k])
            nz = fric_src[sl] * (afs[sl] + BURST[k] * 1.2)
            yf, zf = sosfilt(sos, nz, zi=zf)
            out_f[sl] = yf * 0.45
        else:
            zf = zf * 0.0
    y = out_v + out_f * 1.0
    y = lfilter([1, -0.3], [1], y)          # лёгкая коррекция наклона
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    if return_tracks:
        return y.astype(np.float32), dict(F=F, AV=AV, AF=AF, f0=f0, segs=segs)
    return y.astype(np.float32)


_BP_CACHE: dict = {}


def _bandpass(fc, bw):
    key = (int(fc // 50), int(bw // 50))
    if key not in _BP_CACHE:
        lo = max(150.0, fc - bw / 2)
        hi = min(SR / 2 - 200.0, fc + bw / 2)
        _BP_CACHE[key] = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
    return _BP_CACHE[key]


def _lf(b0, a1, a2, x, z):
    """Резонатор 2-го порядка с переносом состояния между блоками."""
    y, zo = lfilter([b0], [1.0, -a1, -a2], x, zi=z)
    return y, zo


def save_wav(path: str, y: np.ndarray, sr: int = SR):
    from scipy.io import wavfile
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
