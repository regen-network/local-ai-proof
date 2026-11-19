"""Task 2 with a tiny model TRAINED on reviewer decisions (no generative model at all).

Features per (requirement, passage) pair: embedding cosine (max over passage chunks), BM25 score
normalised within the requirement, and the passage's rank under each. A 5-parameter logistic
regression learns from one review's reviewer-cited locations and is tested on the OTHER review
(cross-review transfer, registration <-> issuance). Labels are the same pairs as judge.py.

  python3 harness/judge_trained.py --train private/tasks/reg.json --test private/tasks/iss.json --embed-port 8791
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from common import BM25, assert_loopback, cosine, embed, wilson, write_json
from judge import build_pairs
from rank import chunks, query_text


def features(task: dict, port: int) -> list[dict]:
    corpus = task["corpus"]
    owner, txt = [], []
    for i, c in enumerate(corpus):
        for ch in chunks(c["text"]):
            owner.append(i); txt.append("search_document: " + ch)
    dv = embed(port, txt)
    bm = BM25([c["text"] for c in corpus])
    pairs = build_pairs(task)
    cache = {}
    for pr in pairs:
        if pr["id"] not in cache:
            qv = embed(port, ["search_query: " + pr["q"]])[0]
            cos = [-1.0] * len(corpus)
            for o, v in zip(owner, dv):
                cos[o] = max(cos[o], cosine(qv, v))
            b = bm.scores(pr["q"])
            bmax = max(b) or 1.0
            rc = {i: r for r, i in enumerate(sorted(range(len(corpus)), key=lambda i: -cos[i]))}
            rb = {i: r for r, i in enumerate(sorted(range(len(corpus)), key=lambda i: -b[i]))}
            cache[pr["id"]] = (cos, [x / bmax for x in b], rc, rb)
        cos, bn, rc, rb = cache[pr["id"]]
        i = pr["p"]
        pr["x"] = [cos[i], bn[i], 1 / (1 + rc[i]), 1 / (1 + rb[i])]
    return pairs


def fit(X, y, epochs=3000, lr=0.5, l2=1e-3):
    w = [0.0] * (len(X[0]) + 1)
    for _ in range(epochs):
        g = [0.0] * len(w)
        for x, t in zip(X, y):
            z = w[0] + sum(a * b for a, b in zip(w[1:], x))
            p = 1 / (1 + math.exp(-z))
            g[0] += p - t
            for k, a in enumerate(x):
                g[k + 1] += (p - t) * a
        w = [wi - lr * (gi / len(X) + l2 * wi) for wi, gi in zip(w, g)]
    return w


def predict(w, x):
    return 1 / (1 + math.exp(-(w[0] + sum(a * b for a, b in zip(w[1:], x)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--embed-port", type=int, required=True)
    ap.add_argument("--tag", default="nomic137m-logreg")
    a = ap.parse_args()
    assert_loopback(a.embed_port)
    tr = features(json.load(open(a.train)), a.embed_port)
    t0 = time.time()
    te = features(json.load(open(a.test)), a.embed_port)
    feat_s = time.time() - t0
    # Train: cited passage = 1, any passage the reviewer did not cite = 0.
    w = fit([p["x"] for p in tr], [p["label"] for p in tr])
    # Threshold chosen on TRAIN only: maximise balanced accuracy (positives vs easy negatives).
    best = (0, 0.5)
    for th in [i / 100 for i in range(5, 96)]:
        tp = sum(predict(w, p["x"]) >= th for p in tr if p["kind"] == "positive") / sum(p["kind"] == "positive" for p in tr)
        fp = sum(predict(w, p["x"]) >= th for p in tr if p["kind"] == "easy_negative") / sum(p["kind"] == "easy_negative" for p in tr)
        best = max(best, ((tp + 1 - fp) / 2, th))
    th = best[1]
    for p in te:
        p["p_yes"] = round(predict(w, p["x"]), 4); p["pred"] = int(p["p_yes"] >= th)

    def rate(kind):
        xs = [p for p in te if p["kind"] == kind]
        k = sum(p["pred"] for p in xs)
        return {"n": len(xs), "said_yes": k, "yes_rate": round(k / len(xs), 3), "ci95": wilson(k, len(xs))}

    pos, easy, hard = rate("positive"), rate("easy_negative"), rate("hard_negative")
    byid = {}
    for p in te:
        byid.setdefault(p["id"], {})[p["kind"]] = p["p_yes"]
    wins = [1.0 if v["positive"] > v["easy_negative"] else 0.0 for v in byid.values()]
    wins_h = [1.0 if v["positive"] > v["hard_negative"] else 0.0 for v in byid.values()]
    out = {"task": Path(a.test).stem, "trained_on": Path(a.train).stem, "tag": a.tag, "mode": "trained_logreg",
           "weights": [round(x, 3) for x in w], "threshold": th, "n_pairs": len(te),
           "recall_on_cited_evidence": pos, "false_yes_on_unrelated": easy, "yes_on_uncited_top_keyword_match": hard,
           "balanced_accuracy_reliable_labels": round((pos["yes_rate"] + 1 - easy["yes_rate"]) / 2, 3),
           "pairwise_cited_beats_unrelated": round(sum(wins) / len(wins), 3),
           "pairwise_cited_beats_top_uncited": round(sum(wins_h) / len(wins_h), 3),
           "timing": {"per_pair_mean_s": round(feat_s / len(te), 4), "total_s": round(feat_s, 2)},
           "tokens_total": None, "energy": None}
    write_json(f"private/runs/judge-{out['task']}-{a.tag}.json", {**out, "pairs": [{k: v for k, v in p.items() if k not in ("q", "x")} for p in te]})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
