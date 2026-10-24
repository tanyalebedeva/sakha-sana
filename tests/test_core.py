"""Проверки: численные градиенты CTC и модели, морфология, фонология.
Запуск: python3 tests/test_core.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from sakha.nn import AcousticModel, ctc_loss
from sakha.morphology import paradigm, past3, past1
from sakha.phonology import to_ipa, harmony_violations, syllabify


def brute_ctc(logits, labels):
    """Полный перебор путей CTC - эталон для маленьких T."""
    import itertools
    T, V = logits.shape
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    tot = -np.inf
    for path in itertools.product(range(V), repeat=T):
        col = [p for i, p in enumerate(path) if (i == 0 or p != path[i - 1]) and p != 0]
        if col == list(labels):
            tot = np.logaddexp(tot, sum(lp[t, p] for t, p in enumerate(path)))
    return -tot


def test_ctc_bruteforce():
    rng = np.random.default_rng(0)
    for labels, T in [([1], 3), ([1, 2], 4), ([1, 1], 4), ([2, 1, 2], 5)]:
        z = rng.standard_normal((1, T, 3))
        loss, _ = ctc_loss(z, np.array([T]), [np.array(labels)])
        assert abs(loss - brute_ctc(z[0], labels)) < 1e-6, (labels, loss)


def test_ctc_grad():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2, 7, 5))
    T = np.array([7, 5]); labs = [np.array([1, 2, 2]), np.array([3, 4])]
    _, g = ctc_loss(z, T, labs)
    eps = 1e-5
    for _ in range(20):
        b, t, v = rng.integers(2), rng.integers(7), rng.integers(5)
        zp = z.copy(); zp[b, t, v] += eps
        zm = z.copy(); zm[b, t, v] -= eps
        num = (ctc_loss(zp, T, labs)[0] - ctc_loss(zm, T, labs)[0]) / (2 * eps)
        assert abs(num - g[b, t, v]) < 1e-4, (num, g[b, t, v])


def test_model_grad():
    rng = np.random.default_rng(2)
    m = AcousticModel(6, 5, width=8, dilations=(1, 2), k=3, seed=3)
    def to64(layer):
        for name in ("W", "b", "g"):
            if hasattr(layer, name):
                setattr(layer, name, getattr(layer, name).astype(np.float64))
        for sub in ("ln", "conv"):
            if hasattr(layer, sub):
                to64(getattr(layer, sub))
    for l in m.layers:
        to64(l)
    x = rng.standard_normal((2, 9, 6))
    mask = np.ones((2, 9), np.float32); mask[1, 7:] = 0
    T = np.array([9, 7]); labs = [np.array([1, 2]), np.array([3])]

    def loss_fn():
        return ctc_loss(m.forward(x, mask), T, labs)[0]

    _, g = ctc_loss(m.forward(x, mask), T, labs)
    m.backward(g, mask)
    grads = [gg.copy() for gg in m.grads()]
    eps = 1e-6
    errs = []
    for p, gp in zip(m.params(), grads):
        for _ in range(4):
            i = tuple(rng.integers(s) for s in p.shape)
            old = p[i]
            p[i] = old + eps; lp_ = loss_fn()
            p[i] = old - eps; lm_ = loss_fn()
            p[i] = old
            num = (lp_ - lm_) / (2 * eps)
            errs.append(abs(num - gp[i]) / (abs(num) + abs(gp[i]) + 1e-3))
    assert max(errs) < 1e-3, (np.median(errs), max(errs))


def test_morphology():
    exp = {
        "оҕо": {"pl": "оҕолор", "dat": "оҕоҕо", "acc": "оҕону", "abl": "оҕоттон"},
        "ат": {"pl": "аттар", "dat": "акка", "acc": "аты"},
        "күн": {"pl": "күннэр", "dat": "күҥҥэ"},
        "дьиэ": {"pl": "дьиэлэр", "dat": "дьиэҕэ"},
        "балык": {"pl": "балыктар", "dat": "балыкка", "acc": "балыгы"},
        "ынах": {"dat": "ынахха", "acc": "ынаҕы"},
        "бөрө": {"pl": "бөрөлөр"},
        "кыыс": {"pl": "кыргыттар"},
    }
    for stem, forms in exp.items():
        p = paradigm(stem)
        for k, v in forms.items():
            assert p[k] == v, (stem, k, p[k], v)
    assert past3("бар") == "барда" and past3("кэл") == "кэллэ" and past3("көр") == "көрдө"
    assert past1("бар") == "бардым" and past1("көр") == "көрдүм"


def test_phonology():
    assert to_ipa("саха тыла") == "saqˈa tɯlˈa"
    assert syllabify("олоҥхо") == ["о", "лоҥ", "хо"]
    assert not harmony_violations("кыргыттар")
    assert harmony_violations("оҕолар")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
