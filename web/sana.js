// Саҥа - браузерный порт: G2P, морфология, формантный синтез, log-mel и CTC-распознавание.
// Повторяет sakha/*.py; таблица полосовых фильтров и веса модели подставляются build.py.
(function (root) {
  "use strict";
  const SR = 16000, HOP = 80, FRAME_S = HOP / SR;
  const VOWELS = "аыоуэиөү";
  const VF = { "а": [1, 0, 0], "ы": [1, 0, 1], "о": [1, 1, 0], "у": [1, 1, 1], "э": [0, 0, 0], "и": [0, 0, 1], "ө": [0, 1, 0], "ү": [0, 1, 1] };
  const DIPHTHONGS = { "ыа": "ɯa", "иэ": "ie", "уо": "uo", "үө": "yø" };
  const VIPA = { "а": "a", "ы": "ɯ", "о": "o", "у": "u", "э": "e", "и": "i", "ө": "ø", "ү": "y" };
  const CIPA = { "б": "b", "п": "p", "т": "t", "д": "d", "к": "k", "г": "g", "х": "q", "ҕ": "ɣ", "с": "s", "һ": "h", "м": "m", "н": "n", "ҥ": "ŋ", "л": "l", "р": "r", "й": "j", "ч": "c", "дь": "ɟ", "нь": "ɲ", "в": "v", "ф": "f", "з": "z", "ж": "ʒ", "ш": "ʃ", "щ": "ʃ", "ц": "ts" };
  const LOANV = { "е": ["j", "e"], "ё": ["j", "o"], "ю": ["j", "u"], "я": ["j", "a"] };
  const ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяҕҥөһү";
  const VSET = new Set([...Object.values(VIPA), ...Object.values(DIPHTHONGS)]);

  function normalize(t) {
    t = t.toLowerCase().replace(/h/g, "һ").replace(/ö/g, "ө").replace(/ÿ/g, "ү").replace(/[ӈң]/g, "ҥ");
    let o = "";
    for (const ch of t) o += (ALPHABET.includes(ch) || ch === " " || ch === "-") ? ch : " ";
    return o.replace(/\s+/g, " ").trim();
  }

  function g2pWord(word) {
    const w = normalize(word).replace(/ /g, "");
    const out = [];
    let i = 0;
    while (i < w.length) {
      const two = w.slice(i, i + 2);
      if (DIPHTHONGS[two]) { out.push({ sym: DIPHTHONGS[two], long: true }); i += 2; continue; }
      if (two.length === 2 && two[0] === two[1] && VOWELS.includes(two[0])) { out.push({ sym: VIPA[two[0]], long: true }); i += 2; continue; }
      if (two === "дь" || two === "нь") { out.push({ sym: CIPA[two] }); i += 2; continue; }
      const ch = w[i];
      if (VIPA[ch]) out.push({ sym: VIPA[ch] });
      else if (LOANV[ch]) {
        const [j, v] = LOANV[ch];
        if (!out.length || VSET.has(out[out.length - 1].sym)) out.push({ sym: j });
        out.push({ sym: v });
      } else if (CIPA[ch]) {
        const last = out[out.length - 1];
        if (last && last.sym === CIPA[ch] && !last.long) last.long = true;
        else out.push({ sym: CIPA[ch] });
      }
      i++;
    }
    for (let k = out.length - 1; k >= 0; k--) if (VSET.has(out[k].sym)) { out[k].stress = true; break; }
    return out;
  }
  const g2p = (text) => normalize(text).replace(/-/g, " ").split(" ").filter(Boolean).map(g2pWord);
  const phStr = (p) => (p.stress ? "ˈ" : "") + p.sym + (p.long ? "ː" : "");
  const toIPA = (text) => g2p(text).map((w) => w.map(phStr).join("")).join(" ");

  function units(w) {
    const u = []; let i = 0;
    while (i < w.length) {
      const two = w.slice(i, i + 2);
      if (DIPHTHONGS[two] || (two.length === 2 && two[0] === two[1] && VOWELS.includes(two[0])) || two === "дь" || two === "нь") { u.push(two); i += 2; }
      else { u.push(w[i]); i++; }
    }
    return u;
  }
  function syllabify(word) {
    const w = normalize(word);
    const u = units(w);
    const vidx = u.map((x, k) => VOWELS.includes(x[0]) ? k : -1).filter((k) => k >= 0);
    if (vidx.length <= 1) return [w];
    const cuts = [];
    for (let k = 0; k + 1 < vidx.length; k++) { const a = vidx[k], b = vidx[k + 1]; cuts.push(b - a - 1 === 0 ? b : b - 1); }
    const s = []; let st = 0;
    for (const c of cuts) { s.push(u.slice(st, c).join("")); st = c; }
    s.push(u.slice(st).join(""));
    return s;
  }
  function vowelSeq(word) {
    const w = normalize(word); const seq = []; let i = 0;
    while (i < w.length) {
      const two = w.slice(i, i + 2);
      if (DIPHTHONGS[two]) { seq.push(two); i += 2; }
      else if (VOWELS.includes(w[i])) { seq.push(w[i]); i += two === w[i] + w[i] ? 2 : 1; }
      else i++;
    }
    return seq;
  }
  function harmonyViolations(word) {
    const seq = vowelSeq(word); const bad = [];
    for (let k = 1; k < seq.length; k++) {
      const [pb, pr, ph] = VF[seq[k - 1][0]], [cb, cr, ch] = VF[seq[k][0]];
      if (cb !== pb) { bad.push([seq[k - 1], seq[k], "ряд"]); continue; }
      if ((!pr && cr) || (pr && !ph && !cr) || (pr && ph && !ch && cr)) bad.push([seq[k - 1], seq[k], "огублённость"]);
    }
    return bad;
  }
  function suffixVowel(stem, high) {
    const seq = vowelSeq(stem);
    if (!seq.length) return high ? "ы" : "а";
    const v = seq[seq.length - 1]; const [back, rnd, hi] = VF[v[0]];
    if (high) return back ? (rnd ? "у" : "ы") : (rnd ? "ү" : "и");
    if (DIPHTHONGS[v]) return back ? "а" : "э";
    if (rnd && !hi) return back ? "о" : "ө";
    return back ? "а" : "э";
  }
  const VL = new Set("кпстхчф"), NAS = new Set("мнҥ");
  const morph = {
    pl(s) {
      const irr = { "кыыс": "кыргыттар", "уол": "уолаттар" }; if (irr[s]) return irr[s];
      const c = s.at(-1); let k = "л";
      if (VOWELS.includes(c) || c === "л") k = "л"; else if ("рй".includes(c)) k = "д"; else if (VL.has(c)) k = "т"; else if (NAS.has(c)) k = "н";
      return s + k + suffixVowel(s, false) + "р";
    },
    dat(s) {
      const c = s.at(-1), v = suffixVowel(s, false);
      if (VOWELS.includes(c)) return s + "ҕ" + v;
      if ("лрй".includes(c)) return s + "г" + v;
      if (c === "х") return s + "х" + v;
      if ("тп".includes(c)) return s.slice(0, -1) + "кк" + v;
      if (VL.has(c)) return s + "к" + v;
      if (c === "н") return s.slice(0, -1) + "ҥҥ" + v;
      if (NAS.has(c)) return s + "ҥ" + v;
      return s + "г" + v;
    },
    acc(s) {
      const v = suffixVowel(s, true), c = s.at(-1);
      if (VOWELS.includes(c)) return s + "н" + v;
      const voi = { "к": "г", "х": "ҕ", "п": "б" };
      return (voi[c] ? s.slice(0, -1) + voi[c] : s) + v;
    },
    abl(s) { const v = suffixVowel(s, false); return s + (VOWELS.includes(s.at(-1)) ? "тт" : "т") + v + "н"; },
  };
  function paradigm(stem) { stem = normalize(stem).replace(/ /g, ""); return { nom: stem, pl: morph.pl(stem), dat: morph.dat(stem), acc: morph.acc(stem), abl: morph.abl(stem) }; }

  // ------------------------------------------------------------ синтез
  const VOWEL_F = { a: [750, 1300, 2500], "ɯ": [360, 1450, 2400], o: [500, 880, 2400], u: [320, 780, 2300], e: [520, 1850, 2550], i: [290, 2250, 2950], "ø": [430, 1500, 2300], y: [290, 1700, 2150] };
  const DIPH = { "ɯa": ["ɯ", "a"], ie: ["i", "e"], uo: ["u", "o"], "yø": ["y", "ø"] };
  const CONS = {
    p: ["stop", 0, [250, 900, 2300], [1200, 1500, .5]], b: ["stop", 1, [250, 900, 2300], [1200, 1500, .35]],
    t: ["stop", 0, [250, 1700, 2700], [4000, 2000, .6]], d: ["stop", 1, [250, 1700, 2700], [4000, 2000, .4]],
    k: ["stop", 0, [250, 1800, 2500], [2200, 1200, .7]], g: ["stop", 1, [250, 1800, 2500], [2200, 1200, .45]],
    c: ["affr", 0, [280, 2000, 2800], [3200, 1600, .8]], "ɟ": ["affr", 1, [280, 2000, 2800], [3200, 1600, .55]],
    ts: ["affr", 0, [280, 1700, 2700], [5500, 2500, .7]], q: ["fric", 0, [450, 1100, 2400], [1300, 1000, .55]],
    "ɣ": ["vfric", 1, [420, 1250, 2400], [1300, 1000, .12]], s: ["fric", 0, [300, 1700, 2700], [6000, 2200, .85]],
    "ʃ": ["fric", 0, [300, 1900, 2600], [3300, 1500, .8]], f: ["fric", 0, [300, 1000, 2300], [6500, 4000, .25]],
    h: ["asp", 0, null, [0, 0, .35]], z: ["vfric", 1, [300, 1700, 2700], [6000, 2200, .45]],
    "ʒ": ["vfric", 1, [300, 1900, 2600], [3300, 1500, .4]], v: ["vfric", 1, [300, 1100, 2300], [5000, 4000, .1]],
    m: ["nasal", 1, [260, 1050, 2200], null], n: ["nasal", 1, [260, 1550, 2550], null],
    "ɲ": ["nasal", 1, [260, 2000, 2750], null], "ŋ": ["nasal", 1, [260, 1250, 2300], null],
    l: ["liquid", 1, [360, 1100, 2700], null], r: ["trill", 1, [450, 1300, 1750], null], j: ["glide", 1, [280, 2200, 3000], null],
  };

  function rng(seed) { // mulberry32 + Box–Muller
    let a = seed >>> 0;
    const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    const n = () => { if (spare !== null) { const s = spare; spare = null; return s; } let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (!r || r >= 1); const m = Math.sqrt(-2 * Math.log(r) / r); spare = y * m; return x * m; };
    return { u, n };
  }

  function nextVowelF(word, idx) {
    for (let k = idx + 1; k < word.length; k++) { const s = word[k].sym; if (VOWEL_F[s]) return VOWEL_F[s]; if (DIPH[s]) return VOWEL_F[DIPH[s][0]]; }
    for (let k = idx - 1; k >= 0; k--) { const s = word[k].sym; if (VOWEL_F[s]) return VOWEL_F[s]; if (DIPH[s]) return VOWEL_F[DIPH[s][1]]; }
    return VOWEL_F.a;
  }
  const S = (kind, dur, f, o = {}) => Object.assign({ kind, dur, f, fEnd: null, av: 1, af: 0, fric: [0, 0], stress: false, trill: false, nasal: false, burst: 0, label: "" }, o);

  function textToSegments(text, spk) {
    const segs = []; const words = g2p(text);
    words.forEach((word, wi) => {
      if (wi > 0 && spk.pause > 0.005) segs.push(S("sil", spk.pause, [500, 1500, 2500], { av: 0, label: "_" }));
      word.forEach((ph, i) => {
        const lastSyl = word.slice(i + 1).every((p) => !VOWEL_F[p.sym] && !DIPH[p.sym]);
        const stretch = (lastSyl ? 1.25 : 1) / spk.tempo; const s = ph.sym;
        if (VOWEL_F[s] || DIPH[s]) {
          let f, fe = null, dur;
          if (DIPH[s]) { f = VOWEL_F[DIPH[s][0]]; fe = VOWEL_F[DIPH[s][1]]; dur = 0.18; } else { f = VOWEL_F[s]; dur = ph.long ? 0.165 : 0.085; }
          if (ph.stress) dur *= 1.15;
          segs.push(S("vowel", dur * stretch, f, { fEnd: fe, stress: !!ph.stress, label: phStr(ph) })); return;
        }
        if (!CONS[s]) return;
        const [kind, voiced, loc0, noise] = CONS[s]; const vf = nextVowelF(word, i); const gem = ph.long ? 1.9 : 1;
        if (!loc0) { segs.push(S("asp", 0.07 * gem * stretch, vf, { av: 0, af: noise[2], label: s })); return; }
        const loc = loc0.slice();
        if ("kgŋ".includes(s)) loc[1] = vf[1] > 1600 ? 2200 : 1150;
        if (s === "q") loc[1] = Math.min(loc[1], vf[1]);
        const nf = noise ? [noise[0], noise[1]] : [0, 0];
        if (kind === "stop") {
          segs.push(S("closure", 0.055 * gem * stretch, loc, { av: voiced ? 0.12 : 0, label: s }));
          segs.push(S("release", 0.022 * stretch, loc, { av: voiced ? 0.7 : 0, af: noise[2] * 0.5, fric: nf, burst: noise[2], label: s }));
          if (!voiced) segs.push(S("asp", 0.02 * stretch, vf, { av: 0, af: 0.25, label: s }));
        } else if (kind === "affr") {
          segs.push(S("closure", 0.045 * gem * stretch, loc, { av: voiced ? 0.12 : 0, label: s }));
          segs.push(S("fric", 0.075 * stretch, loc, { av: voiced ? 0.55 : 0, af: noise[2], fric: nf, burst: noise[2] * 0.6, label: s }));
        } else if (kind === "fric" || kind === "vfric") segs.push(S("fric", 0.095 * gem * stretch, loc, { av: voiced ? 0.6 : 0, af: noise[2], fric: nf, label: s }));
        else if (kind === "nasal") segs.push(S("nasal", 0.07 * gem * stretch, loc, { av: 0.55, nasal: true, label: s }));
        else if (kind === "liquid") segs.push(S("liquid", 0.065 * gem * stretch, loc, { av: 0.75, label: s }));
        else if (kind === "trill") segs.push(S("trill", 0.075 * gem * stretch, loc, { av: 0.75, trill: true, label: s }));
        else if (kind === "glide") segs.push(S("glide", 0.055 * stretch, loc, { av: 0.8, label: s }));
      });
    });
    return segs;
  }

  function hann(n) { const N = n + 2, k = []; let s = 0; for (let i = 1; i <= n; i++) { const v = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)); k.push(v); s += v; } return k.map((v) => v / s); }
  function smooth(x, n) { // как np.convolve(pad edge, hanning, 'same')
    if (n <= 1) return x.slice();
    const k = hann(n), L = x.length, out = new Float64Array(L), half = Math.floor((n - 1) / 2);
    for (let i = 0; i < L; i++) { let s = 0; for (let j = 0; j < n; j++) { const idx = Math.min(L - 1, Math.max(0, i + half - j)); s += k[j] * x[idx]; } out[i] = s; }
    return out;
  }
  function interpFrames(arr, nsamp) { const out = new Float64Array(nsamp), n = arr.length; for (let i = 0; i < nsamp; i++) { const x = (i + 0.5) / HOP; const i0 = Math.min(n - 1, Math.floor(x)), i1 = Math.min(n - 1, i0 + 1), t = Math.min(1, x - i0); out[i] = arr[i0] * (1 - t) + arr[i1] * t; } return out; }

  function bandpass(fc, bw) {
    const T = root.SANA_BP; const kf = Math.min(T.nf - 1, Math.floor(fc / 50));
    let kb = T.bws.indexOf(Math.floor(bw / 50)); if (kb < 0) kb = 0;
    const off = (kf * T.bws.length + kb) * 12; return T.sos.subarray(off, off + 12);
  }

  function synthesize(text, spk, opts = {}) {
    spk = Object.assign({ f0: 120, vtl: 1, tempo: 1, breath: 0.04, jitter: 0.01, shimmer: 0.05, tilt: 0, bw: 1, pause: 0.03, seed: 1 }, spk);
    const R = rng(spk.seed);
    const segs = [S("sil", 0.08, [500, 1500, 2500], { av: 0, label: "_" }), ...textToSegments(text, spk), S("sil", 0.1, [500, 1500, 2500], { av: 0, label: "_" })];
    const F = [[], [], []], AV = [], AF = [], FC = [], FB = [], BURST = [], TR = [], NASL = [], STR = [];
    for (const sg of segs) {
      const n = Math.max(1, Math.round(sg.dur / FRAME_S)); const fe = sg.fEnd || sg.f;
      for (let i = 0; i < n; i++) {
        const t = n === 1 ? 0 : i / (n - 1);
        for (let d = 0; d < 3; d++) F[d].push((sg.f[d] * (1 - t) + fe[d] * t) * spk.vtl);
        AV.push(sg.av); AF.push(sg.af); FC.push(sg.fric[0]); FB.push(Math.max(sg.fric[1], 1));
        BURST.push(i < 2 ? sg.burst : 0); TR.push(sg.trill); NASL.push(sg.nasal); STR.push(sg.stress ? 1 : 0);
      }
    }
    const nfr = AV.length;
    const Fs = F.map((f) => smooth(f, 7));
    let av = smooth(AV, 3); const af = smooth(AF, 2);
    for (let k = 0; k < nfr; k++) if (TR[k]) av[k] *= 0.55 + 0.45 * Math.cos(2 * Math.PI * 28 * k * FRAME_S);
    const strS = smooth(STR, 15), vf = R.u() + 0.5, vph = R.u() * 6, f0 = new Float64Array(nfr);
    for (let k = 0; k < nfr; k++) {
      const prog = nfr > 1 ? k / (nfr - 1) : 0;
      f0[k] = spk.f0 * (1.12 - 0.27 * prog) * (1 + 0.13 * strS[k]) * (1 + 0.03 * Math.sin(2 * Math.PI * vf * k * FRAME_S + vph)) * (1 + spk.jitter * R.n());
    }
    const nsamp = nfr * HOP, f0s = interpFrames(f0, nsamp), avs = interpFrames(av, nsamp), afs = interpFrames(af, nsamp);
    const tp = 0.4, tn = 0.16; const glottal = new Float64Array(nsamp), g = new Float64Array(nsamp);
    let phase = 0, prevG = 0, shim = 1 + spk.shimmer * R.n(), tiltY = 0;
    for (let i = 0; i < nsamp; i++) {
      const np_ = phase + f0s[i] / SR; if (np_ >= 1) shim = 1 + spk.shimmer * R.n(); phase = np_ % 1;
      const gv = phase < tp ? 0.5 * (1 - Math.cos(Math.PI * phase / tp)) : phase < tp + tn ? Math.cos(0.5 * Math.PI * (phase - tp) / tn) : 0;
      g[i] = gv; let v = (gv - prevG) * shim * 8; prevG = gv;
      if (spk.tilt > 0) { tiltY = (1 - spk.tilt) * v + spk.tilt * tiltY; v = tiltY; }
      glottal[i] = v * avs[i] + R.n() * spk.breath * (0.4 + gv) * avs[i];
    }
    const out = new Float64Array(nsamp); const zv = [0, 1, 2, 3, 4].map(() => [0, 0]); let zf = new Float64Array(4);
    const fric = new Float64Array(nsamp); for (let i = 0; i < nsamp; i++) fric[i] = R.n();
    const baseBw = [80, 100, 140, 220, 300];
    for (let k = 0; k < nfr; k++) {
      const s0 = k * HOP; const isAsp = av[k] < 0.05 && af[k] > 0.05 && FC[k] < 1;
      const bws = baseBw.map((b) => b * spk.bw); if (NASL[k]) { bws[0] *= 1.6; bws[1] *= 2.0; }
      const forms = [Fs[0][k], Fs[1][k], Fs[2][k], 3500 * spk.vtl, 4400 * spk.vtl];
      const coefs = forms.map((ff, j) => { ff = Math.min(ff, SR / 2 - 400); const r = Math.exp(-Math.PI * bws[j] / SR); const a1 = 2 * r * Math.cos(2 * Math.PI * ff / SR), a2 = -r * r; return [1 - a1 - a2, a1, a2]; });
      const useF = FC[k] > 1 && !isAsp; const sos = useF ? bandpass(FC[k] * Math.min(spk.vtl, 1.15), FB[k]) : null;
      if (!useF) zf.fill(0);
      for (let i = s0; i < s0 + HOP; i++) {
        let y = glottal[i] + (isAsp ? fric[i] * af[k] * 0.5 : 0);
        for (let j = 0; j < 5; j++) { const [b0, a1, a2] = coefs[j], z = zv[j]; const yy = b0 * y + z[0]; z[0] = a1 * yy + z[1]; z[1] = a2 * yy; y = yy; }
        if (NASL[k]) y *= 0.6;
        if (useF) {
          let x = fric[i] * (afs[i] + BURST[k] * 1.2);
          for (let sct = 0; sct < 2; sct++) { const o = sct * 6, b0 = sos[o], b1 = sos[o + 1], b2 = sos[o + 2], a1 = sos[o + 4], a2 = sos[o + 5]; const yy = b0 * x + zf[sct * 2]; zf[sct * 2] = b1 * x - a1 * yy + zf[sct * 2 + 1]; zf[sct * 2 + 1] = b2 * x - a2 * yy; x = yy; }
          y += x * 0.45;
        }
        out[i] = y;
      }
    }
    let prev = 0, mx = 1e-9; const res = new Float32Array(nsamp);
    for (let i = 0; i < nsamp; i++) { const v = out[i] - 0.3 * prev; prev = out[i]; res[i] = v; mx = Math.max(mx, Math.abs(v)); }
    for (let i = 0; i < nsamp; i++) res[i] = res[i] / mx * 0.9;
    if (opts.tracks) return { y: res, segs, f0, F: Fs };
    return res;
  }

  function addNoise(y, snrDb, seed) {
    if (snrDb >= 60) return y;
    const R = rng(seed); let ps = 0; for (const v of y) ps += v * v; ps /= y.length;
    const nz = new Float32Array(y.length); let pn = 0; for (let i = 0; i < y.length; i++) { nz[i] = R.n(); pn += nz[i] * nz[i]; } pn /= y.length;
    const sc = Math.sqrt(ps / pn / Math.pow(10, snrDb / 10)); const o = new Float32Array(y.length); let mx = 1e-9;
    for (let i = 0; i < y.length; i++) { o[i] = y[i] + nz[i] * sc; mx = Math.max(mx, Math.abs(o[i])); }
    for (let i = 0; i < y.length; i++) o[i] = o[i] / mx * 0.9; return o;
  }

  // ------------------------------------------------------------ признаки
  function fft(re, im) { // in-place radix-2
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let j = 0; j < len / 2; j++) { const a = i + j, b = a + len / 2; const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr; re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti; const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr; } }
    }
  }
  let MELFB = null;
  function melFB(nm = 40, nfft = 512, fmin = 60, fmax = 7800) {
    const h2m = (f) => 2595 * Math.log10(1 + f / 700), m2h = (m) => 700 * (Math.pow(10, m / 2595) - 1);
    const lo = h2m(fmin), hi = h2m(fmax); const bins = [];
    for (let i = 0; i < nm + 2; i++) bins.push(Math.floor((nfft + 1) * m2h(lo + (hi - lo) * i / (nm + 1)) / SR));
    const fb = [];
    for (let m = 1; m <= nm; m++) { const row = new Float64Array(nfft / 2 + 1); const l = bins[m - 1], c = bins[m], r = bins[m + 1]; for (let k = l; k < c; k++) row[k] = (k - l) / Math.max(c - l, 1); for (let k = c; k < r; k++) row[k] = (r - k) / Math.max(r - c, 1); fb.push(row); }
    return fb;
  }
  function powerFrames(y, win, hop, nfft, pre) {
    let x = y; if (pre) { x = new Float64Array(y.length); x[0] = y[0]; for (let i = 1; i < y.length; i++) x[i] = y[i] - 0.97 * y[i - 1]; }
    if (x.length < win) { const p = new Float64Array(win); p.set(x); x = p; }
    const n = 1 + Math.floor((x.length - win) / hop), frames = [];
    const w = new Float64Array(win); for (let i = 0; i < win; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (win - 1));
    for (let t = 0; t < n; t++) {
      const re = new Float64Array(nfft), im = new Float64Array(nfft);
      for (let i = 0; i < win; i++) re[i] = x[t * hop + i] * w[i];
      fft(re, im); const p = new Float64Array(nfft / 2 + 1); for (let k = 0; k <= nfft / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k]; frames.push(p);
    }
    return frames;
  }
  function logmel(y) {
    MELFB = MELFB || melFB(); const P = powerFrames(y, 400, 160, 512, true);
    const M = P.map((p) => MELFB.map((row) => { let s = 0; for (let k = 0; k < row.length; k++) if (row[k]) s += row[k] * p[k]; return Math.log(s + 1e-6); }));
    const T = M.length, D = 40;
    for (let d = 0; d < D; d++) { let mu = 0; for (let t = 0; t < T; t++) mu += M[t][d]; mu /= T; let v = 0; for (let t = 0; t < T; t++) v += (M[t][d] - mu) ** 2; const sd = Math.sqrt(v / T) + 1e-5; for (let t = 0; t < T; t++) M[t][d] = (M[t][d] - mu) / sd; }
    return M;
  }
  function spectrogram(y) { return powerFrames(y, 400, 120, 512, false).map((p) => p.map((v) => 10 * Math.log10(v + 1e-10))); }

  // ------------------------------------------------------------ распознавание
  let MODEL = null;
  function loadModel() {
    if (MODEL) return MODEL;
    const W = root.SANA_WEIGHTS; if (!W) return null;
    const bin = typeof atob === "function" ? atob(W.data) : Buffer.from(W.data, "base64").toString("binary");
    const bytes = new Int8Array(bin.length); for (let i = 0; i < bin.length; i++) bytes[i] = (bin.charCodeAt(i) << 24) >> 24;
    const tensors = W.tensors.map((t) => {
      const n = t.shape.reduce((a, b) => a * b, 1); const out = new Float32Array(n); const cols = t.shape.length === 2 ? t.shape[1] : 1;
      if (t.f32) { const f = new Float32Array(Uint8Array.from(atob ? atob(t.f32) : Buffer.from(t.f32, "base64").toString("binary"), (c) => c.charCodeAt(0)).buffer); out.set(f); }
      else for (let i = 0; i < n; i++) out[i] = bytes[t.off + i] * t.scale[i % cols];
      return { shape: t.shape, d: out };
    });
    MODEL = { cfg: W.cfg, t: tensors, chars: W.chars }; return MODEL;
  }
  function conv(x, T, C, W, b, k, d, cout) {
    const p = d * (k - 1) / 2, out = new Float32Array(T * cout);
    for (let t = 0; t < T; t++) {
      const o = t * cout; for (let c = 0; c < cout; c++) out[o + c] = b[c];
      for (let j = 0; j < k; j++) {
        const tt = t + j * d - p; if (tt < 0 || tt >= T) continue;
        for (let ci = 0; ci < C; ci++) { const xv = x[tt * C + ci]; if (!xv) continue; const wr = (j * C + ci) * cout; for (let c = 0; c < cout; c++) out[o + c] += xv * W[wr + c]; }
      }
    }
    return out;
  }
  function ln(x, T, C, g, b) { const o = new Float32Array(T * C); for (let t = 0; t < T; t++) { let mu = 0, v = 0; for (let c = 0; c < C; c++) mu += x[t * C + c]; mu /= C; for (let c = 0; c < C; c++) v += (x[t * C + c] - mu) ** 2; const inv = 1 / Math.sqrt(v / C + 1e-5); for (let c = 0; c < C; c++) o[t * C + c] = (x[t * C + c] - mu) * inv * g[c] + b[c]; } return o; }
  function recognize(y) {
    const M = loadModel(); if (!M) return null;
    const mel = logmel(y); const T = Math.floor(mel.length / 2), D = 80, x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) for (let d = 0; d < 40; d++) { x[t * D + d] = mel[2 * t][d]; x[t * D + 40 + d] = mel[2 * t + 1][d]; }
    const { width: Wd, dilations, k, vocab } = M.cfg; const P = M.t; let pi = 0; const nx = () => P[pi++].d;
    let h = conv(x, T, D, nx(), nx(), 3, 1, Wd); for (let i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0;
    for (const dl of dilations) { const g = nx(), bb = nx(), Wc = nx(), bc = nx(); const c = conv(ln(h, T, Wd, g, bb), T, Wd, Wc, bc, k, dl, Wd); for (let i = 0; i < h.length; i++) h[i] += c[i] > 0 ? c[i] : 0; }
    const hn = ln(h, T, Wd, nx(), nx()); const Wo = nx(), bo = nx();
    const ids = [], probs = [];
    for (let t = 0; t < T; t++) {
      let best = 0, bv = -1e9; const lg = new Float32Array(vocab);
      for (let v = 0; v < vocab; v++) { let s = bo[v]; for (let c = 0; c < Wd; c++) s += hn[t * Wd + c] * Wo[c * vocab + v]; lg[v] = s; if (s > bv) { bv = s; best = v; } }
      let z = 0; for (let v = 0; v < vocab; v++) z += Math.exp(lg[v] - bv); probs.push(1 / z); ids.push(best);
    }
    let text = "", prevId = -1; const align = [];
    ids.forEach((id, t) => { if (id !== prevId && id !== 0) { text += M.chars[id - 1]; align.push({ t, ch: M.chars[id - 1], p: probs[t] }); } prevId = id; });
    return { text: text.trim(), frames: ids, align, T };
  }

  const api = { SR, normalize, g2p, toIPA, syllabify, harmonyViolations, paradigm, suffixVowel, synthesize, addNoise, logmel, spectrogram, recognize, loadModel, textToSegments };
  if (typeof module !== "undefined") module.exports = api; else root.Sana = api;
})(typeof window !== "undefined" ? window : globalThis);
