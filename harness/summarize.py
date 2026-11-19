"""Collect run files into one aggregate table. Strips per-item detail (ids, gold, rankings, pairs),
so the output carries no review content and can be committed.

  python3 harness/summarize.py private/runs results/summary-v0.2.json
"""
from __future__ import annotations

import glob
import json
import sys

from common import calibration, write_json

DROP = {"per_item", "pairs"}


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = []
    for f in sorted(glob.glob(f"{src}/*.json")):
        j = json.load(open(f))
        row = {k: v for k, v in j.items() if k not in DROP} | {"run": f.split("/")[-1][:-5]}
        if j.get("pairs") and "p_yes" in j["pairs"][0]:
            row["calibration"] = calibration(j["pairs"])
        rows.append(row)
    write_json(out, {"note": "Aggregates only. Labels: one past soil-carbon project's signed registry checklists "
                             "(registration n=22 located rows; issuance n=24). Wide intervals; proof of method, not a benchmark.",
                     "runs": rows})
    for r in rows:
        if "metrics" in r:
            m = r["metrics"]
            print(f"{r['run']:45s} hit@1 {m['hit@1']:.2f} {tuple(m['hit@1_ci95'])}  hit@3 {m['hit@3']:.2f}  mrr {m['mrr']:.2f}")
        else:
            print(f"{r['run']:45s} recall {r['recall_on_cited_evidence']['yes_rate']:.2f}  false-yes(unrelated) "
                  f"{r['false_yes_on_unrelated']['yes_rate']:.2f}  yes(top-uncited) {r['yes_on_uncited_top_keyword_match']['yes_rate']:.2f}  "
                  f"bal-acc {r['balanced_accuracy_reliable_labels']:.2f}  brier {r.get('calibration', {}).get('brier')}  s/pair {r['timing']['per_pair_mean_s']}")


if __name__ == "__main__":
    main()
