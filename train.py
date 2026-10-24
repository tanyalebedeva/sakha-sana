"""Обучение CTC-распознавания якутской речи на синтетическом корпусе.
python3 train.py --n-train 8000 --epochs 20"""
import argparse, json, os, pickle, time

import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):  # кириллица и ҕҥһөү в консоли Windows
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sakha.corpus import VOCAB, build, decode, encode
from sakha.nn import Adam, AcousticModel, ctc_loss, greedy_decode

STACK = 2


def stack(f):
    T = len(f) // STACK * STACK
    return f[:T].reshape(T // STACK, -1)


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return int(d[-1])


def batches(feats, labels, bs, rng=None, specaug=False):
    order = np.argsort([len(f) for f in feats])
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if rng is not None:
        rng.shuffle(chunks)
    for idx in chunks:
        T = np.array([len(feats[i]) for i in idx])
        x = np.zeros((len(idx), T.max(), feats[idx[0]].shape[1]), np.float32)
        for k, i in enumerate(idx):
            f = feats[i].copy()
            if specaug:
                for _ in range(2):
                    w = rng.integers(0, 8); f0 = rng.integers(0, 40 - w)
                    f[:, f0:f0 + w] = 0; f[:, 40 + f0:40 + f0 + w] = 0
                for _ in range(2):
                    w = rng.integers(0, min(8, len(f) // 6 + 1)); t0 = rng.integers(0, max(1, len(f) - w))
                    f[t0:t0 + w] = 0
            x[k, :T[k]] = f
        mask = (np.arange(T.max())[None] < T[:, None]).astype(np.float32)
        yield idx, x, mask, T, [labels[i] for i in idx]


def evaluate(model, feats, texts, bs=64, n=None):
    labels = [encode(t) for t in texts]
    n = n or len(feats)
    ce = cn = we = wn = 0
    examples = []
    for idx, x, mask, T, _ in batches(feats[:n], labels[:n], bs):
        hyp = greedy_decode(model.forward(x, mask), T)
        for i, h in zip(idx, hyp):
            ref, out = texts[i], decode(h).strip()
            ce += edit(ref, out); cn += len(ref)
            we += edit(ref.split(), out.split()); wn += len(ref.split())
            if len(examples) < 12:
                examples.append((ref, out))
    return ce / cn, we / wn, examples


def load_or_build(path, *a, **kw):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t = time.time()
    texts, feats = build(*a, **kw)
    feats = [stack(f) for f in feats]
    with open(path, "wb") as fh:
        pickle.dump((texts, feats), fh)
    print(f"built {path}: {len(texts)} utts in {time.time() - t:.0f}s", flush=True)
    return texts, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--budget-min", type=float, default=90)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--pseudo", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    suffix = f"_ps{args.pseudo}" if args.pseudo else ""
    tr_texts, tr_feats = load_or_build(f"data/train_{args.n_train}{suffix}.pkl", args.n_train, 0, pseudo=args.pseudo)
    sets = {
        "seen_words": load_or_build("data/test_seen.pkl", 500, 50_000_000),
        "unseen_words": load_or_build("data/test_unseen.pkl", 500, 60_000_000, heldout=True),
        "clean": load_or_build("data/test_clean.pkl", 300, 70_000_000, aug=False),
    }
    tr_labels = [encode(t) for t in tr_texts]
    print("train hours: %.2f" % (sum(len(f) for f in tr_feats) * 0.02 / 3600), flush=True)

    model = AcousticModel(tr_feats[0].shape[1], VOCAB, width=args.width)
    print("params:", model.n_params(), flush=True)
    opt = Adam(model.params(), lr=args.lr)
    rng = np.random.default_rng(0)
    total_steps = args.epochs * ((len(tr_feats) + args.bs - 1) // args.bs)
    warm = 300
    log, best, t0, step = [], 9.0, time.time(), 0
    for ep in range(args.epochs):
        losses = []
        for _, x, mask, T, labs in batches(tr_feats, tr_labels, args.bs, rng, specaug=True):
            lr = args.lr * min(1, (step + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(step / total_steps, 1)))
            opt.lr = max(lr, 3e-5)
            logits = model.forward(x, mask)
            loss, g = ctc_loss(logits, T, labs)
            model.backward(g, mask)
            opt.step(model.grads())
            losses.append(loss); step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step} loss {np.mean(losses[-50:]):.3f} lr {opt.lr:.5f} "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
        cer, wer, ex = evaluate(model, sets["seen_words"][1], sets["seen_words"][0], n=200)
        rec = dict(epoch=ep, loss=float(np.mean(losses)), val_cer=cer, val_wer=wer,
                   minutes=(time.time() - t0) / 60)
        log.append(rec); print(rec, ex[:3], flush=True)
        if cer < best:
            best = cer
            np.savez(f"{args.out}/model.npz", **model.state(), cfg=json.dumps(model.cfg))
        json.dump(log, open(f"{args.out}/log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if (time.time() - t0) / 60 > args.budget_min:
            print("time budget reached"); break

    st = np.load(f"{args.out}/model.npz")
    model.load(st)
    results = {}
    for name, (texts, feats) in sets.items():
        cer, wer, ex = evaluate(model, feats, texts)
        results[name] = dict(cer=cer, wer=wer, examples=ex)
        print(name, f"CER {cer:.3f} WER {wer:.3f}", flush=True)
    json.dump(dict(results=results, log=log, pseudo=args.pseudo, params=model.n_params(), n_train=args.n_train,
                   train_hours=sum(len(f) for f in tr_feats) * 0.02 / 3600),
              open(f"{args.out}/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
