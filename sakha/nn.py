"""Мини-фреймворк на numpy: dilated Conv1D + LayerNorm + residual, CTC, Adam.
Без PyTorch — все производные выписаны вручную (и проверены численно в tests)."""
from __future__ import annotations

import numpy as np

NEG = -1e30


# ---------------------------------------------------------------- слои

class Conv1D:
    def __init__(self, cin, cout, k, d, rng):
        self.k, self.d = k, d
        self.W = (rng.standard_normal((k * cin, cout)) * np.sqrt(2.0 / (k * cin))).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def params(self):
        return [self.W, self.b]

    def forward(self, x):
        B, T, C = x.shape
        p = self.d * (self.k - 1) // 2
        xp = np.pad(x, ((0, 0), (p, p), (0, 0)))
        cols = np.stack([xp[:, j * self.d: j * self.d + T] for j in range(self.k)], 2)
        self.cache = (cols.reshape(B * T, -1), x.shape, p)
        return (self.cache[0] @ self.W + self.b).reshape(B, T, -1)

    def backward(self, g):
        cols, (B, T, C), p = self.cache
        g2 = g.reshape(B * T, -1)
        self.dW = cols.T @ g2
        self.db = g2.sum(0)
        dcols = (g2 @ self.W.T).reshape(B, T, self.k, C)
        dxp = np.zeros((B, T + 2 * p, C), g.dtype)
        for j in range(self.k):
            dxp[:, j * self.d: j * self.d + T] += dcols[:, :, j]
        return dxp[:, p: p + T]

    def grads(self):
        return [self.dW, self.db]


class LayerNorm:
    def __init__(self, c):
        self.g = np.ones(c, np.float32)
        self.b = np.zeros(c, np.float32)

    def params(self):
        return [self.g, self.b]

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + 1e-5)
        xh = (x - mu) * inv
        self.cache = (xh, inv)
        return xh * self.g + self.b

    def backward(self, g):
        xh, inv = self.cache
        self.dg = (g * xh).reshape(-1, xh.shape[-1]).sum(0)
        self.db = g.reshape(-1, xh.shape[-1]).sum(0)
        gx = g * self.g
        N = xh.shape[-1]
        return inv / N * (N * gx - gx.sum(-1, keepdims=True) - xh * (gx * xh).sum(-1, keepdims=True))

    def grads(self):
        return [self.dg, self.db]


class Linear:
    def __init__(self, cin, cout, rng):
        self.W = (rng.standard_normal((cin, cout)) * np.sqrt(1.0 / cin)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def params(self):
        return [self.W, self.b]

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        C = self.x.shape[-1]
        self.dW = self.x.reshape(-1, C).T @ g.reshape(-1, g.shape[-1])
        self.db = g.reshape(-1, g.shape[-1]).sum(0)
        return g @ self.W.T

    def grads(self):
        return [self.dW, self.db]


class ResBlock:
    """y = x + ReLU(Conv(LN(x)))"""

    def __init__(self, c, k, d, rng):
        self.ln = LayerNorm(c)
        self.conv = Conv1D(c, c, k, d, rng)
        self.conv.W *= 0.5

    def params(self):
        return self.ln.params() + self.conv.params()

    def forward(self, x, mask):
        h = self.conv.forward(self.ln.forward(x))
        self.relu = h > 0
        return (x + h * self.relu) * mask

    def backward(self, g):
        gh = self.conv.backward(g * self.relu)
        return g + self.ln.backward(gh)

    def grads(self):
        return self.ln.grads() + self.conv.grads()


class AcousticModel:
    def __init__(self, n_in, vocab, width=192, dilations=(1, 1, 2, 2, 4), k=5, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = dict(n_in=n_in, vocab=vocab, width=width, dilations=list(dilations), k=k)
        self.inp = Conv1D(n_in, width, 3, 1, rng)
        self.blocks = [ResBlock(width, k, d, rng) for d in dilations]
        self.ln = LayerNorm(width)
        self.out = Linear(width, vocab, rng)
        self.layers = [self.inp, *self.blocks, self.ln, self.out]

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def grads(self):
        return [g for l in self.layers for g in l.grads()]

    def forward(self, x, mask):
        m = mask[..., None]
        h = self.inp.forward(x * m)
        self.relu0 = h > 0
        h = h * self.relu0 * m
        for b in self.blocks:
            h = b.forward(h, m)
        return self.out.forward(self.ln.forward(h))

    def backward(self, g, mask):
        m = mask[..., None]
        g = self.ln.backward(self.out.backward(g * m))
        for b in reversed(self.blocks):
            g = b.backward(g * m)
        self.inp.backward(g * self.relu0 * m)

    def n_params(self):
        return sum(p.size for p in self.params())

    def state(self):
        return {f"p{i}": p for i, p in enumerate(self.params())}

    def load(self, state):
        for i, p in enumerate(self.params()):
            p[...] = state[f"p{i}"]


# ---------------------------------------------------------------- CTC

def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _lse(*xs):
    m = np.maximum.reduce(xs)
    m_safe = np.where(m <= NEG / 2, 0.0, m)
    s = sum(np.exp(x - m_safe) for x in xs)
    return np.where(m <= NEG / 2, NEG, m_safe + np.log(s + 1e-300))


def ctc_loss(logits, T_lens, labels, blank=0):
    """logits (B,T,V); labels — список массивов. Возвращает (loss_mean, dlogits).
    Прямой-обратный проход в лог-пространстве, векторизованный по батчу."""
    B, T, V = logits.shape
    lp = log_softmax(logits.astype(np.float64))
    L = np.array([len(l) for l in labels])
    S = 2 * L.max() + 1
    ext = np.full((B, S), blank, np.int64)
    for b, l in enumerate(labels):
        ext[b, 1:2 * len(l) + 1:2] = l
    Sb = 2 * L + 1
    valid = np.arange(S)[None] < Sb[:, None]
    skip = np.zeros((B, S), bool)
    skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    bidx = np.arange(B)[:, None]
    emit = lp[bidx[:, :, None], np.arange(T)[None, :, None], ext[:, None, :]]  # (B,T,S)

    alpha = np.full((B, T, S), NEG)
    alpha[:, 0, 0] = emit[:, 0, 0]
    alpha[:, 0, 1] = np.where(L > 0, emit[:, 0, 1], NEG)
    for t in range(1, T):
        a = alpha[:, t - 1]
        a1 = np.concatenate([np.full((B, 1), NEG), a[:, :-1]], 1)
        a2 = np.where(skip, np.concatenate([np.full((B, 2), NEG), a[:, :-2]], 1), NEG)
        new = _lse(a, a1, a2) + emit[:, t]
        new = np.where(valid, new, NEG)
        live = (t < T_lens)[:, None]
        alpha[:, t] = np.where(live, new, NEG)

    beta = np.full((B, T, S), NEG)
    skip_next = np.zeros((B, S), bool)
    skip_next[:, :-2] = skip[:, 2:]
    for t in range(T - 1, -1, -1):
        if t < T - 1:
            bn = beta[:, t + 1]
            b1 = np.concatenate([bn[:, 1:], np.full((B, 1), NEG)], 1)
            b2 = np.where(skip_next, np.concatenate([bn[:, 2:], np.full((B, 2), NEG)], 1), NEG)
            rec = _lse(bn, b1, b2) + emit[:, t]
        else:
            rec = np.full((B, S), NEG)
        init = np.full((B, S), NEG)
        init[np.arange(B), Sb - 1] = emit[np.arange(B), t, Sb - 1]
        has2 = Sb >= 2
        init[np.arange(B)[has2], Sb[has2] - 2] = emit[np.arange(B)[has2], t, Sb[has2] - 2]
        is_last = (t == T_lens - 1)[:, None]
        cur = np.where(is_last, init, rec)
        cur = np.where((t < T_lens)[:, None] & valid, cur, NEG)
        beta[:, t] = cur

    ll = _lse(alpha[np.arange(B), T_lens - 1, Sb - 1],
              np.where(Sb >= 2, alpha[np.arange(B), T_lens - 1, np.maximum(Sb - 2, 0)], NEG))
    gamma = alpha + beta - emit - ll[:, None, None]                  # log occupancy
    occ = np.exp(np.minimum(gamma, 0.0)) * (gamma > NEG / 2)
    onehot = np.zeros((B, S, V))
    onehot[bidx, np.arange(S)[None], ext] = 1.0
    onehot *= valid[..., None]
    post = occ @ onehot                                                # (B,T,V)
    tmask = (np.arange(T)[None] < T_lens[:, None])[..., None]
    grad = (np.exp(lp) - post) * tmask / B
    return float(-ll.mean()), grad.astype(np.float32)


def greedy_decode(logits, T_lens, blank=0):
    out = []
    best = logits.argmax(-1)
    for b in range(len(T_lens)):
        seq = best[b, : T_lens[b]]
        keep = np.concatenate([[True], seq[1:] != seq[:-1]]) & (seq != blank)
        out.append(seq[keep])
    return out


# ---------------------------------------------------------------- оптимизатор

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.98, wd=1e-5, clip=5.0):
        self.params, self.lr, self.b1, self.b2, self.wd, self.clip = params, lr, b1, b2, wd, clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        norm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
        scale = min(1.0, self.clip / (norm + 1e-9))
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale + self.wd * p
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(p.dtype)
        return norm
