"""Экспорт таблицы полосовых фильтров (Баттерворт 2-го порядка) для JS-синтезатора."""
import base64, json, sys, os
import numpy as np
from scipy.signal import butter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakha.synth import CONS, SR

bws = sorted({int(v[3][1] // 50) for v in CONS.values() if v[3] and v[3][1] > 0})
nf = 161
sos = np.zeros((nf, len(bws), 12), np.float32)
for kf in range(nf):
    fc = max(kf * 50 + 25, 400)
    for kb, b in enumerate(bws):
        bw = b * 50
        lo = max(150.0, fc - bw / 2); hi = min(SR / 2 - 200.0, fc + bw / 2)
        if hi <= lo + 50:
            lo = hi - 400
        sos[kf, kb] = butter(2, [lo, hi], btype="band", fs=SR, output="sos").ravel()
b64 = base64.b64encode(sos.tobytes()).decode()
js = ("(function(r){const s='%s';const b=typeof atob==='function'?atob(s):Buffer.from(s,'base64').toString('binary');"
      "const u=Uint8Array.from(b,c=>c.charCodeAt(0));r.SANA_BP={nf:%d,bws:%s,sos:new Float32Array(u.buffer)};})"
      "(typeof window!=='undefined'?window:globalThis);") % (b64, nf, json.dumps(bws))
open(os.path.join(os.path.dirname(__file__), "bp_table.js"), "w", encoding="utf-8").write(js)
print(bws, len(js))
