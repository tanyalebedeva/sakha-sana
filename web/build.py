"""Сборка самодостаточной страницы: int8-веса модели + таблица фильтров + JS + результаты."""
import base64, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sakha.corpus import CHARS

run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "runs", "main")
st = np.load(os.path.join(run, "model.npz"))
cfg = json.loads(str(st["cfg"]))
keys = sorted([k for k in st.files if k.startswith("p")], key=lambda k: int(k[1:]))

blob, tensors = bytearray(), []
for k in keys:
    a = st[k].astype(np.float32)
    if a.ndim == 2:                                  # матрицы - int8, масштаб на столбец
        sc = np.abs(a).max(0) / 127 + 1e-12
        q = np.clip(np.round(a / sc), -127, 127).astype(np.int8)
        tensors.append(dict(shape=list(a.shape), off=len(blob), scale=[float(f"{s:.6g}") for s in sc]))
        blob += q.tobytes()
    else:                                            # векторы - float32
        tensors.append(dict(shape=list(a.shape), f32=base64.b64encode(a.tobytes()).decode()))
W = dict(cfg=cfg, chars=CHARS, tensors=tensors, data=base64.b64encode(bytes(blob)).decode())
weights_js = "window.SANA_WEIGHTS=" + json.dumps(W, ensure_ascii=False) + ";"
open(os.path.join(HERE, "weights.js"), "w").write(weights_js.replace("window.", "globalThis.", 1))

res = json.load(open(os.path.join(run, "results.json")))
base_path = os.path.join(HERE, "..", "runs", "main", "results.json")
if os.path.abspath(base_path) != os.path.abspath(os.path.join(run, "results.json")) and os.path.exists(base_path):
    b = json.load(open(base_path))
    res["baseline"] = {k: dict(cer=v["cer"], wer=v["wer"], examples=v["examples"]) for k, v in b["results"].items()}
page = open(os.path.join(HERE, "page.html")).read()
page = page.replace("/*BP_TABLE*/", open(os.path.join(HERE, "bp_table.js")).read())
page = page.replace("/*WEIGHTS*/", weights_js)
page = page.replace("/*SANA_JS*/", open(os.path.join(HERE, "sana.js")).read())
page = page.replace("/*RESULTS*/null", json.dumps(res, ensure_ascii=False))
out = os.path.join(HERE, "sana.html")
open(out, "w").write(page)
print("page", round(len(page) / 1e6, 2), "MB; int8 weights", round(len(blob) / 1e6, 2), "MB")
