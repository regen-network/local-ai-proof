"""Task 1 — evidence location. Rank corpus passages for each requirement; score against the
human reviewer's cited locations.

Methods (smallest first):
  bm25     rules only: keyword scoring, no model
  embed    small open embedding model (llama.cpp server on 127.0.0.1, --embed-port)
  hybrid   reciprocal-rank fusion of bm25 + embed
  judge    hybrid top-K re-ranked by a small generative model's P(yes) "does this passage hold
           the evidence?" (--chat-port), e.g. OLMo-2 1B / 7B

  python3 harness/rank.py --task private/tasks/reg.json --method hybrid --embed-port 8791 --tag nomic
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from common import BM25, PowerSampler, assert_loopback, chat, cosine, embed, rank_metrics, write_json, _post

JUDGE_SYS = ("You check documents for a carbon-credit registry review. You will see one requirement and one "
             "passage from a project document. Answer only 'yes' if the passage itself contains evidence that "
             "addresses the requirement, otherwise answer only 'no'.")


def query_text(it: dict) -> str:
    return f"{it['category']}. {it['requirement']} Accepted evidence: {it['accepted_evidence']}"


def chunks(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size]); i += size - overlap
    return out or [text]


def gold_match(sid: str, gold: list[str]) -> bool:
    return any(sid == g or sid.startswith(g + ".") for g in gold)


def judge_prompt(req: str, passage: str, max_chars: int = 3500) -> str:
    return f"Requirement:\n{req}\n\nPassage:\n{passage[:max_chars]}\n\nDoes the passage contain evidence for the requirement? Answer yes or no."


# Few-shot examples are SYNTHETIC (invented project), never real review content.
FEWSHOT = [
    ("Crediting Period. The crediting period must be stated. Accepted evidence: Project plan crediting period",
     "Crediting Period\nThe crediting period is ten years, from 1 March 2025 to 28 February 2035.", "yes"),
    ("Crediting Period. The crediting period must be stated. Accepted evidence: Project plan crediting period",
     "Local Stakeholder Consultation\nNeighbours and the parish council were informed in writing; no objections were received.", "no"),
    ("Land Tenure. The project must demonstrate legal right to the land. Accepted evidence: Title registers; ownership statement",
     "Leakage\nYield reduction leakage is assessed against national average yields.", "no"),
    ("Land Tenure. The project must demonstrate legal right to the land. Accepted evidence: Title registers; ownership statement",
     "Project Ownership\nOwnership of carbon rights rests with the land owner, evidenced by the land registry title.", "yes"),
]


def p_yes(port: int, req: str, passage: str, max_chars: int = 3500, fewshot: bool = False) -> tuple[float, dict]:
    msgs = [{"role": "system", "content": JUDGE_SYS}]
    if fewshot:
        for q, p, a in FEWSHOT:
            msgs += [{"role": "user", "content": judge_prompt(q, p)}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": judge_prompt(req, passage, max_chars)})
    payload = {"messages": msgs,
               "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10, "cache_prompt": True}
    r = _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    lp = r["choices"][0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
    py = sum(math.exp(t["logprob"]) for t in lp if t["token"].strip().lower() in ("yes", "y"))
    pn = sum(math.exp(t["logprob"]) for t in lp if t["token"].strip().lower() in ("no", "n"))
    return (py / (py + pn) if py + pn > 0 else 0.5), r.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", choices=["random", "bm25", "embed", "hybrid", "judge"], required=True)
    ap.add_argument("--embed-port", type=int)
    ap.add_argument("--chat-port", type=int)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="private/runs")
    ap.add_argument("--energy", action="store_true")
    a = ap.parse_args()

    task = json.load(open(a.task))
    corpus, items = task["corpus"], [it for it in task["items"] if it["gold"]]
    sids = [c["sid"] for c in corpus]
    for p in (a.embed_port, a.chat_port):
        if p:
            assert_loopback(p)

    ps = PowerSampler() if a.energy else None
    idle = ps.baseline() if ps else None

    t_start = time.time()
    ranked, per_item_s, usage_tokens = [], [], 0
    bm = BM25([c["text"] for c in corpus])

    doc_vecs = None
    if a.method in ("embed", "hybrid", "judge"):
        chunk_owner, chunk_txt = [], []
        for i, c in enumerate(corpus):
            for ch in chunks(c["text"]):
                chunk_owner.append(i); chunk_txt.append("search_document: " + ch)
        t0 = time.time()
        doc_vecs = (chunk_owner, embed(a.embed_port, chunk_txt))
        index_s = time.time() - t0
    else:
        index_s = 0.0

    ctx = ps.__enter__() if ps else None
    for it in items:
        t0 = time.time()
        q = query_text(it)
        b = bm.scores(q)
        order_b = sorted(range(len(corpus)), key=lambda i: -b[i])
        if a.method == "random":
            order = list(range(len(corpus))); random.Random(it["id"]).shuffle(order)
        elif a.method == "bm25":
            order = order_b
        else:
            qv = embed(a.embed_port, ["search_query: " + q])[0]
            best = [-1.0] * len(corpus)
            for owner, v in zip(*doc_vecs):
                best[owner] = max(best[owner], cosine(qv, v))
            order_e = sorted(range(len(corpus)), key=lambda i: -best[i])
            if a.method == "embed":
                order = order_e
            else:
                rrf = [0.0] * len(corpus)
                for o in (order_b, order_e):
                    for r, i in enumerate(o):
                        rrf[i] += 1 / (60 + r)
                order = sorted(range(len(corpus)), key=lambda i: -rrf[i])
                if a.method == "judge":
                    top = order[:a.k]
                    scored = []
                    for i in top:
                        p, u = p_yes(a.chat_port, q, corpus[i]["text"])
                        usage_tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                        scored.append((p, -top.index(i), i))
                    order = [i for _, _, i in sorted(scored, reverse=True)] + order[a.k:]
        ranked.append([sids[i] for i in order])
        per_item_s.append(time.time() - t0)
    if ps:
        ps.__exit__(None, None, None)

    # Expand gold to concrete corpus ids (a gold "2" or "1.6" covers its sub-sections).
    gold = [[s for s in sids if gold_match(s, it["gold"])] for it in items]
    m = rank_metrics(ranked, gold)
    name = Path(a.task).stem
    run = {"task": name, "method": a.method, "tag": a.tag, "k": a.k if a.method == "judge" else None,
           "corpus_size": len(corpus), "metrics": m,
           "timing": {"index_s": round(index_s, 2), "per_item_mean_s": round(sum(per_item_s) / len(per_item_s), 3),
                      "per_item_max_s": round(max(per_item_s), 3), "total_s": round(time.time() - t_start, 1)},
           "tokens_total": usage_tokens or None,
           "energy": ps.joules(idle) if ps else None}
    write_json(f"{a.out_dir}/{name}-{a.method}{('-' + a.tag) if a.tag else ''}.json",
               {**run, "per_item": [{"id": it["id"], "gold": g, "top5": r[:5]} for it, g, r in zip(items, gold, ranked)]})
    print(json.dumps({k: v for k, v in run.items()}, indent=None))


if __name__ == "__main__":
    main()
