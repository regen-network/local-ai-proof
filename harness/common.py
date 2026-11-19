"""Shared helpers: text normalisation, BM25, local model clients, power sampling, metrics.

Everything here runs on the local machine. Model calls go only to 127.0.0.1.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to
was were will with which shall must should may can not no any all each per be been being their
there these those than then into such where when who whom whose what how also other project""".split())


def tokens(text: str) -> list[str]:
    t = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in t if w not in STOP and len(w) > 1]


class BM25:
    """Plain Okapi BM25 — the 'rules first' baseline. No model, no training."""

    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


# ---------------------------------------------------------------- local model clients

def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    # Explicitly bypass any proxy: loopback only.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def assert_loopback(port: int) -> None:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    addrs = [l.split()[8] for l in out.splitlines()[1:] if len(l.split()) > 8]
    if not addrs or any(not a.startswith("127.0.0.1:") for a in addrs):
        raise SystemExit(f"port {port}: listener not loopback-only ({addrs})")


def embed(port: int, texts: list[str], batch: int = 16) -> list[list[float]]:
    vecs = []
    for i in range(0, len(texts), batch):
        r = _post(f"http://127.0.0.1:{port}/v1/embeddings", {"input": texts[i:i + batch], "model": "local"})
        vecs += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    return vecs


def chat(port: int, system: str, user: str, max_tokens: int = 64, grammar: str | None = None) -> tuple[str, dict]:
    payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "temperature": 0, "max_tokens": max_tokens, "cache_prompt": True}
    if grammar:
        payload["grammar"] = grammar
    r = _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    return r["choices"][0]["message"]["content"], r.get("usage", {})


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-12)


# ---------------------------------------------------------------- energy

class PowerSampler:
    """Samples whole-machine power from macOS battery telemetry (SystemLoad, mW) once a second.

    powermetrics needs root; this does not. It measures the WHOLE machine, so we subtract an
    idle baseline taken just before the run. Treat results as rough (±30%).
    """

    def __init__(self, interval: float = 1.0):
        self.interval, self.samples, self._stop = interval, [], threading.Event()

    @staticmethod
    def read_mw() -> float | None:
        out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True).stdout
        m = re.search(r'"SystemLoad"=(\d+)', out)
        return float(m.group(1)) if m else None

    def baseline(self, seconds: int = 8) -> float | None:
        vals = []
        for _ in range(seconds):
            v = self.read_mw()
            if v is not None:
                vals.append(v)
            time.sleep(1)
        return sorted(vals)[len(vals) // 2] if vals else None

    def _run(self):
        while not self._stop.is_set():
            v = self.read_mw()
            if v is not None:
                self.samples.append((time.time(), v))
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.time(); self.th = threading.Thread(target=self._run, daemon=True); self.th.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self.th.join(); self.t1 = time.time()

    def joules(self, idle_mw: float | None) -> dict:
        if not self.samples:
            return {"available": False}
        mean_mw = sum(v for _, v in self.samples) / len(self.samples)
        secs = self.t1 - self.t0
        out = {"available": True, "seconds": round(secs, 2), "mean_system_w": round(mean_mw / 1000, 2),
               "total_j": round(mean_mw / 1000 * secs, 1)}
        if idle_mw is not None:
            out["idle_w"] = round(idle_mw / 1000, 2)
            out["above_idle_j"] = round(max(0.0, (mean_mw - idle_mw) / 1000) * secs, 1)
        return out


# ---------------------------------------------------------------- metrics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def rank_metrics(ranked: list[list[str]], gold: list[list[str]]) -> dict:
    n = len(gold)
    h1 = sum(1 for r, g in zip(ranked, gold) if r[:1] and r[0] in g)
    h3 = sum(1 for r, g in zip(ranked, gold) if set(r[:3]) & set(g))
    h5 = sum(1 for r, g in zip(ranked, gold) if set(r[:5]) & set(g))
    mrr = 0.0
    for r, g in zip(ranked, gold):
        for i, s in enumerate(r):
            if s in g:
                mrr += 1 / (i + 1); break
    return {"n": n, "hit@1": round(h1 / n, 3), "hit@1_ci95": wilson(h1, n), "hit@3": round(h3 / n, 3),
            "hit@3_ci95": wilson(h3, n), "hit@5": round(h5 / n, 3), "mrr": round(mrr / n, 3),
            "hit@1_count": h1, "hit@3_count": h3}


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def calibration(pairs: list[dict], bins: int = 5) -> dict:
    """Brier score and expected calibration error of p_yes against reviewer labels, on the reliable
    labels only (reviewer-cited passage = 1, unrelated passage = 0)."""
    xs = [(p["p_yes"], p["label"]) for p in pairs if p["kind"] in ("positive", "easy_negative")]
    if not xs:
        return {}
    brier = sum((p - y) ** 2 for p, y in xs) / len(xs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in xs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if sel:
            ece += len(sel) / len(xs) * abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return {"n": len(xs), "brier": round(brier, 3), "ece_5bin": round(ece, 3)}
