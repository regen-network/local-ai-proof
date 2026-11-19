"""Score existing large-model outputs on the same evidence-location yardstick (Task 1, sections).

  --baseline  review-agent evaluation baseline JSON (per_row with ra_page_citations), e.g. the
              April 2026 run on a hosted open-weight 120B model
  --draft     a review-report Markdown draft with '### REQ-NNN' blocks (e.g. a Claude-written draft)

Cited section numbers are read in order of appearance; citations to other documents are skipped.
Rows are aligned to the human checklist by requirement-text similarity.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re

from common import rank_metrics, write_json
from rank import gold_match


def sec_ids(s: str) -> list[str]:
    out = []
    for m in re.finditer(r"(?:Section|Sections|§)\s*((?:\d+\.)*\d+)|(Appendix)", s):
        sid = m.group(1) or "Appendix"
        out.append(sid.rstrip("."))
    return out


def dedupe(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--draft")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    task = json.load(open(a.task))
    sids = [c["sid"] for c in task["corpus"]]
    items = [it for it in task["items"] if it["gold"]]

    preds: dict[str, list[str]] = {}
    if a.baseline:
        rows = json.load(open(a.baseline))["per_row"]
        for it in items:
            best = max(rows, key=lambda r: difflib.SequenceMatcher(None, r["gt_requirement_text"], it["requirement"]).ratio())
            cites = [c for c in best["ra_page_citations"] if "Project Plan" in c]
            preds[it["id"]] = dedupe(x for c in cites for x in sec_ids(c))
    if a.draft:
        md = open(a.draft).read()
        blocks = re.split(r"^### (REQ-\d+)[^\n]*\n", md, flags=re.M)
        req_text = {blocks[i]: blocks[i + 1] for i in range(1, len(blocks) - 1, 2)}
        # Align REQ-NNN to checklist rows via the baseline's matched requirement ids when given,
        # otherwise by order.
        order = sorted(req_text)
        for k, it in enumerate(task["items"]):
            if it["gold"] and k < len(order):
                preds[it["id"]] = dedupe(sec_ids(req_text[order[k]]))

    ranked, gold, n_empty = [], [], 0
    for it in items:
        p = [s for s in preds.get(it["id"], []) if any(gold_match(c, [s]) or c == s for c in sids)]
        # map a cited parent (e.g. 1.6) to its first concrete corpus id
        concrete = []
        for s in p:
            concrete += [c for c in sids if c == s or c.startswith(s + ".")][:1]
        concrete = dedupe(concrete)
        n_empty += not concrete
        ranked.append(concrete)
        gold.append([s for s in sids if gold_match(s, it["gold"])])
    m = rank_metrics(ranked, gold)
    out = {"task": task.get("kind"), "method": "reference", "tag": a.tag, "metrics": m, "items_without_section_cite": n_empty,
           "per_item": [{"id": it["id"], "gold": g, "top5": r[:5]} for it, g, r in zip(items, gold, ranked)]}
    write_json(f"private/runs/reg-ref-{a.tag}.json", out)
    print(a.tag, m, "no-cite:", n_empty)


if __name__ == "__main__":
    main()
