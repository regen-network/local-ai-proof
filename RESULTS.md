---
title: Local AI proof point — results
status: draft
access: public
---

# Local AI proof point — results

**Question.** Can the review agent's language steps run on small, fully local models, judged
against what human reviewers actually decided?

**Short answer.** For the two narrow tasks tested, yes — but not by swapping in a small chatbot.
Keyword rules plus a 137M-parameter open encoder find the evidence about as well as the large
models did, and a five-weight classifier trained on reviewer decisions judges evidence better
than any zero-shot model we tried, in about a hundredth of a second per check on a laptop. A
1.5B fully open generative model used zero-shot over-scores, just as the large model did.
OLMo-2 7B is more careful but still well behind the trained classifier and inconsistent across the two reviews. This is one project, about 46 labelled requirements: proof of method, not a
benchmark.

## 1. Setup

- **Ground truth:** the signed reviewer checklists for one past soil-carbon project — a
  registration review (22 requirements with a cited location) and a credit-issuance review (24).
  Each row names the document sections, tables or figures the reviewer accepted as evidence.
- **Corpus:** the project plan split into 37 numbered sections (registration); the 2024
  monitoring report split into 13 text pages (issuance). The signed reviews themselves were
  excluded.
- **Machine:** one Apple M5 Max laptop, 128 GB. All models in `llama-server` on 127.0.0.1 under an
  OS network-deny sandbox (checked: no DNS or internet from inside; loopback works).
- **Models:** see `results/MODELS.md` (repo, revision, sha256, licence, openness for each).
- **Code:** `harness/`. Re-run with `harness/run_all.sh`. Aggregates (with calibration) in `results/summary-v0.2.json`.

## 2. Task 1 — evidence location

Given a requirement, rank the passages; is the reviewer-cited one first (hit@1) or in the top 3?

| Approach | Model size | Registration hit@1 | hit@3 | Issuance hit@1 | hit@3 | Time / requirement |
|---|---|---|---|---|---|---|
| Random | — | 0.00 | 0.09 | 0.08 | 0.33 | — |
| Rules (BM25) | 0 | 0.73 | 0.86 | 0.67 | 0.83 | < 1 ms |
| Encoder (nomic-embed v1.5) | 137M | 0.77 | 0.86 | 0.67 | 0.88 | ~12 ms |
| Rules + encoder (hybrid) | 137M | 0.73 | **0.91** | **0.71** | **0.88** | ~12 ms |
| Hybrid, top 5 re-ranked by OLMo-2 1B | 1.5B | 0.27 | 0.73 | 0.33 | 0.54 | ~0.7 s |
| Hybrid, top 5 re-ranked by OLMo-2 7B | 7.3B | 0.46 | 0.82 | 0.38 | 0.83 | ~1–1.7 s |
| *Reference: hosted GPT-OSS-120B run, agent v2.5.0 (April)* | 120B | 0.46 | 0.64 | — | — | — |
| *Reference: Claude-written review draft (April)* | frontier | 0.64 | 0.68 | — | — | — |

95% intervals are wide: hit@1 0.73 on 22 items is roughly 0.52–0.87.

**Reading the references fairly.** The large-model outputs were not built as ranked lists: they
cite sections inside written findings, some requirements got no section citation at all (5 of 22
for the 120B run, 1 of 22 for Claude), some citations point at methodology sections rather than
the project plan, and both runs had the signed reviews in their document folder. So they are
indicative, not like-for-like. What they do show: the small local route is not behind them on
finding evidence.

**Error types (rules + encoder):**
- *Summary-section pull* (5 of 6 registration top-1 misses): the plan's "Project Eligibility"
  section restates many requirements in brief and ranks first; the cited section is usually 2nd or 3rd.
- *Neighbouring section* (crediting period vs permanence period).
- *Misleading category label*: where a row's category label points to a different part of the
  plan than its requirement text, the label words pull the ranking the wrong way.
- *Evidence in a figure*: two issuance rows cite a page that is mostly a figure (about 250
  characters of text). Text-only methods can't see it. Needs OCR or a layout-aware step.

## 3. Task 2 — evidence sufficiency (does this passage hold the evidence?)

Pairs per requirement: the reviewer-cited passage (should be "yes"), an unrelated passage (should
be "no"), and the top keyword match the reviewer did not cite (ambiguous: it may hold related
evidence, so "yes" there is an upper bound on over-scoring, not a certain error).

| Approach | Size | Reg: recall on cited | Reg: false "yes" on unrelated | Iss: recall | Iss: false "yes" | Balanced accuracy (reg / iss) | Time / check |
|---|---|---|---|---|---|---|---|
| "Covered for everything" (the earlier pipeline's behaviour) | — | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 / 0.50 | — |
| **Trained classifier** (5 weights on encoder + keyword features; trained on the *other* review) | 137M + 5 | **0.91** | **0.00** | **0.83** | **0.04** | **0.95 / 0.90** | ~10 ms |
| OLMo-2 1B, zero-shot | 1.5B | 0.86 | 0.64 | 0.96 | 0.75 | 0.61 / 0.60 | ~0.2 s |
| OLMo-2 1B, few-shot (4 synthetic examples) | 1.5B | 0.82 | 0.73 | 1.00 | 0.96 | 0.55 / 0.52 | ~0.1–0.15 s |
| OLMo-2 7B, zero-shot | 7.3B | 0.55 | 0.05 | 0.04 | 0.00 | 0.75 / 0.52 | ~0.2–0.3 s |
| OLMo-2 7B, threshold tuned on the other review | 7.3B | 0.86 | 0.32 | 0.54 | 0.17 | 0.77 / 0.69 | ~0.2–0.3 s |
| *Reference: gpt-oss-20b, zero-shot* | 21B MoE | 0.64 | 0.05 | 0.54 | 0.04 | 0.80 / 0.75 | ~0.5–0.65 s |

"Yes" on the top uncited keyword match: trained classifier 0.59 / 0.42; OLMo-2 1B 0.59 / 0.96;
OLMo-2 7B 0.36 / 0.00; gpt-oss-20b 0.41 / 0.00. Many of these are probably legitimate (the reviewer cited one place; the
evidence often appears in two).

**Calibration** (Brier score of the "yes" probability against reviewer labels; lower is better;
0.25 is a coin flip at 0.5): trained classifier **0.08 / 0.13**; OLMo-2 1B 0.26 / 0.27 (few-shot
0.30 / 0.33); OLMo-2 7B 0.23 / 0.36; gpt-oss-20b 0.21 / 0.25 (it answers yes/no only).
A calibrated probability matters because it lets confident items be pre-filled and unsure ones go
to a person.

**What this shows**
- The trained classifier is the best and smallest judge here. Its learned weights lean almost
  entirely on the passage's rank under the encoder, so it is really "encoder + a threshold learned
  from reviewers". That is the point: reviewer decisions are the training signal.
- A small generative model used zero-shot is not a substitute. OLMo-2 1B says "yes" to most
  passages, including unrelated ones; neither a threshold tuned on the other review nor four
  worked examples fixes it (balanced accuracy 0.52–0.61). Same failure as the large model's
  "everything is covered".
- OLMo-2 7B goes the other way: careful on the registration review, but on the issuance report
  (layout-heavy page text) it says "no" to almost everything. It does rank cited passages above
  unrelated ones (pairwise 0.86 / 0.79), so a tuned threshold recovers some ground (0.77 / 0.69),
  still well short of the trained classifier. As a re-ranker it lowers hit@1.
- gpt-oss-20b is cautious instead: few false yeses, but it misses a third to a half of the cited
  evidence.
- **Caveat that matters:** the train/test split is two reviews of the *same* project (same
  developer, same document templates). Transfer to a new developer or methodology is untested.

## 4. Time, compute and energy

- **Compute per requirement.** The current route sends a whole document to a large model per
  requirement × mapped document (the project plan here is ~64K characters, ~16K tokens). The local
  route embeds the document once (~0.4–0.8 s for the whole plan), then spends ~12 ms per
  requirement to locate and ~10 ms per passage to judge. A generative judge on the top 5 passages
  reads ~2K tokens per requirement.
- **Hardware need.** The encoder route fits in well under 1 GB of memory; it would run on any
  recent laptop. OLMo-2 1B needs ~3 GB, 7B ~15 GB at the publisher's 16-bit format.
- **Energy: not reliably measured this round.** `powermetrics` needs an administrator password;
  the fallback (macOS battery telemetry) refreshes only every ~20+ seconds, and another workload
  was loading the machine during the runs. Upper bounds from time ×
  100 W above idle: encoder route **≤ ~1–2 J per requirement**; OLMo-2 1B judge ≤ ~20 J per check; OLMo-2 7B ≤ ~30 J per check;
  gpt-oss-20b ≤ ~65 J per check. The one quiet-machine reading (gpt-oss-20b, registration) was
  ~8 J per check above idle. For publishable numbers: rerun `harness/bench_energy.py` alongside
  `sudo powermetrics --samplers cpu_power,gpu_power -i 1000` on an otherwise idle machine, or on a
  metered plug.

## 5. What this does and doesn't support

Supports:
- "Finding the evidence" does not need a frontier model: rules plus a 137M open encoder put the
  reviewer-cited section in the top three 88–91% of the time on this review (86–88% for
  the encoder alone, 83–86% for rules alone).
- "Judging the evidence" can be learned from reviewer decisions with a tiny model, and that beats
  zero-shot small and mid-size generative models on this data.
- Small *generative* models (1.5B and 7B, fully open) are not good zero-shot judges here. Use
  them last, for wording, or fine-tune them on reviewer decisions before they judge anything.

Does not support (yet):
- Any claim about verdict accuracy (all labels here are "approved").
- Generalisation beyond one developer's document templates.
- Field extraction (dates, areas, IDs) or finding drafting on small models; not tested.
- Energy figures fit for publication.

## 6. Next tests (in order)

1. Add the two other projects in the review agent's evaluation suite (their signed checklists are
   held by the review-agent team) — triples the labelled set, adds a second developer.
2. Real negatives: requirements a reviewer did not approve or approved with comments.
3. Field extraction with verbatim checks (dates, areas, IDs, owners).
4. OCR / layout step for figure-only evidence.
5. Clean energy measurement.
6. Fine-tune a small model (encoder first, then OLMo-class) on reviewer decisions from more
   projects; Olmo 3 7B (no publisher GGUF yet; convert or use MLX) as the generative candidate.
