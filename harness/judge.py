"""Task 2 — evidence sufficiency. For a (requirement, passage) pair, does the passage hold evidence
for the requirement? A small generative model answers yes/no.

Why this task: the review agent's April 2026 evaluation showed an over-scoring bias (a hosted
120B model marked every requirement 'covered', including the three a reviewer had not approved
in a synthetic mixed-verdict fixture). A useful small model must say 'no' when the evidence is not
there.

Pairs per checklist row that has a human-cited location:
  positive       the passage the human reviewer cited                       (label yes)
  easy negative  a random passage the reviewer did not cite                 (label no; reliable)
  hard negative  the top keyword-ranked passage the reviewer did not cite   (label 'not cited';
                 ambiguous — it may well hold related evidence, so a 'yes' here is an upper
                 bound on over-scoring, not a certain error)

  python3 harness/judge.py --task private/tasks/reg.json --chat-port 8792 --tag olmo2-1b
  python3 harness/judge.py ... --generate      # for reasoning models that cannot answer in 1 token
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from common import BM25, PowerSampler, _post, assert_loopback, wilson, write_json
from rank import JUDGE_SYS, gold_match, p_yes, query_text


def gen_yes(port: int, req: str, passage: str, max_chars: int = 3500) -> tuple[float, dict]:
    user = f"Requirement:\n{req}\n\nPassage:\n{passage[:max_chars]}\n\nDoes the passage contain evidence for the requirement? Answer yes or no."
    payload = {"messages": [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
               "temperature": 0, "max_tokens": 512, "cache_prompt": True,
               "chat_template_kwargs": {"reasoning_effort": "low"}}
    r = _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    txt = (r["choices"][0]["message"].get("content") or "").strip().lower()
    return (1.0 if re.match(r"\W*yes", txt) else 0.0), r.get("usage", {})


def build_pairs(task: dict) -> list[dict]:
    corpus = task["corpus"]
    bm = BM25([c["text"] for c in corpus])
    pairs = []
    for it in task["items"]:
        if not it["gold"]:
            continue
        q = query_text(it)
        s = bm.scores(q)
        order = sorted(range(len(corpus)), key=lambda i: -s[i])
        gold_idx = [i for i in order if gold_match(corpus[i]["sid"], it["gold"])]
        non = [i for i in order if i not in gold_idx]
        if not gold_idx or not non:
            continue
        rnd = random.Random(it["id"]).choice(non[1:] or non)
        pairs += [{"id": it["id"], "kind": "positive", "label": 1, "q": q, "p": gold_idx[0]},
                  {"id": it["id"], "kind": "hard_negative", "label": 0, "q": q, "p": non[0]},
                  {"id": it["id"], "kind": "easy_negative", "label": 0, "q": q, "p": rnd}]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--chat-port", type=int)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--fewshot", action="store_true", help="prepend 4 synthetic worked examples")
    ap.add_argument("--always-yes", action="store_true", help="the 'covered for everything' baseline")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--energy", action="store_true")
    a = ap.parse_args()
    task = json.load(open(a.task))
    pairs = build_pairs(task)
    if a.chat_port:
        assert_loopback(a.chat_port)
    ps = PowerSampler() if a.energy else None
    idle = ps.baseline() if ps else None
    if ps:
        ps.__enter__()
    times, toks = [], 0
    for pr in pairs:
        t0 = time.time()
        if a.always_yes:
            p, u = 1.0, {}
        elif a.generate:
            p, u = gen_yes(a.chat_port, pr["q"], task["corpus"][pr["p"]]["text"])
        else:
            p, u = p_yes(a.chat_port, pr["q"], task["corpus"][pr["p"]]["text"], fewshot=a.fewshot)
        pr["p_yes"] = round(p, 4)
        pr["pred"] = int(p >= a.threshold)
        times.append(time.time() - t0)
        toks += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    if ps:
        ps.__exit__(None, None, None)

    def rate(kind):
        xs = [p for p in pairs if p["kind"] == kind]
        k = sum(p["pred"] for p in xs)
        return {"n": len(xs), "said_yes": k, "yes_rate": round(k / len(xs), 3), "ci95": wilson(k, len(xs))}

    pos, easy, hard = rate("positive"), rate("easy_negative"), rate("hard_negative")
    # balanced accuracy on the reliable labels (positives vs easy negatives)
    bal = round((pos["yes_rate"] + (1 - easy["yes_rate"])) / 2, 3)
    # ranking quality independent of threshold: P(p_yes[positive] > p_yes[easy neg]) for the same row
    byid = {}
    for p in pairs:
        byid.setdefault(p["id"], {})[p["kind"]] = p["p_yes"]
    wins = [1.0 if v["positive"] > v["easy_negative"] else 0.5 if v["positive"] == v["easy_negative"] else 0.0 for v in byid.values()]
    wins_h = [1.0 if v["positive"] > v["hard_negative"] else 0.5 if v["positive"] == v["hard_negative"] else 0.0 for v in byid.values()]
    out = {"task": Path(a.task).stem, "tag": a.tag, "mode": "always_yes" if a.always_yes else ("generate" if a.generate else ("logprob-fewshot" if a.fewshot else "logprob")),
           "threshold": a.threshold, "n_pairs": len(pairs),
           "recall_on_cited_evidence": pos, "false_yes_on_unrelated": easy, "yes_on_uncited_top_keyword_match": hard,
           "balanced_accuracy_reliable_labels": bal,
           "pairwise_cited_beats_unrelated": round(sum(wins) / len(wins), 3),
           "pairwise_cited_beats_top_uncited": round(sum(wins_h) / len(wins_h), 3),
           "timing": {"per_pair_mean_s": round(sum(times) / len(times), 3), "total_s": round(sum(times), 1)},
           "tokens_total": toks or None, "energy": ps.joules(idle) if ps else None}
    write_json(f"private/runs/judge-{out['task']}-{a.tag}.json", {**out, "pairs": [{k: v for k, v in p.items() if k != 'q'} for p in pairs]})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
