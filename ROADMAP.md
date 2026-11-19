---
title: Roadmap — every Claims Engine review step on local or community compute
status: draft
access: public
---

# Roadmap: every review step on local or community compute

**Goal.** Every machine step in a Claims Engine review runs on a small model on the reviewer's
machine or on community compute. A frontier model is an exception that the evidence owner has
consented to, and each use is logged. Every determination records where it ran.

**Starting point** (see `ASSESSMENT.md`, `RESULTS.md`):
- Document sorting and requirement mapping already run on rules, with no model.
- Evidence extraction sends a whole document (up to ~80–150K characters) to a large model per
  requirement and asks it for snippets, a confidence score and a "matches accepted evidence" flag.
  Structured field extraction, unified analysis and cross-document synthesis also call a large model.
- First measurements: on one past project's signed reviews, rules plus a 137M-parameter encoder
  find the reviewer-cited section in the top 3 for 88–91% of requirements, and a 5-weight
  classifier trained on reviewer decisions separates cited evidence from unrelated passages with
  ~0.90–0.95 balanced accuracy, in milliseconds per check. Fully open generative models used
  zero-shot did worse (OLMo-2 1B over-scores; OLMo-2 7B 0.52–0.77). Small sample; proof of method.

## Principle: decompose, then shrink

The big-model call does three jobs at once: **find** the passage, **judge** whether it holds the
evidence, and **write** it up. Split them, and each job needs far less model:

| Job | Smallest thing that works | Escalate to |
|---|---|---|
| Find the passage | rules (BM25) + compact encoder (≈100M params) | trained encoder |
| Judge sufficiency | classifier trained on reviewer decisions | small fully open generative model (OLMo-class 1–7B) |
| Pull out the value (date, area, ID, owner) | parsers + extractive span model; values copied verbatim and checked | small generative model with a schema; never invents numbers |
| Cross-check documents | deterministic code (dates, areas, totals, IDs) | small generative model to word the finding |
| Draft the finding | template filled from the above | small generative model |

Models never do arithmetic: totals, buffers and reconciliations come from code.

## Stages

Owners and dates are placeholders: **[OWNER]**, **[DATE]**. "Month" counts from project start.

### Stage 0 — Measure and record (months 0–2) · [OWNER: review-agent engineer]

1. Publish the evaluation harness (this repo) and run it on every past review we are authorised to
   reuse, across at least two developers and two methodologies.
2. Collect **negative examples**: requirements a reviewer did not approve or approved with
   comments. Today's labels are all "approved", which cannot test over-scoring. Until real ones
   exist, use clearly marked synthetic gaps.
3. **Record where each determination ran** (Claims Engine user story CE-US-0048, in a private story repository; its content is summarised here). Each machine determination carries:
   execution location (`local` / `community` / `external`), model id, revision and file hash,
   runtime, host class, energy estimate, and, for `external`, the provider and the consent
   reference. A project marked local-only that records an `external` determination is flagged, and
   the round cannot be marked complete (the story's third acceptance criterion). Expressed in W3C
   PROV-O terms (the model as the associated agent, the host as the location).
4. Add a local backend: the agent's OpenAI-compatible route already honours `OPENAI_BASE_URL`, so
   it can point at a sandboxed `llama-server` on 127.0.0.1. Verify end to end.

### Stage 1 — Find and judge locally (months 1–4) · [OWNER]

5. Replace whole-document prompting for *locating* evidence with rules + compact encoder
   (shown here). The model sees only the top few passages, not the whole document.
6. Replace the "matches accepted evidence" judgement with a small classifier trained on reviewer
   confirm/remove decisions. A fully open generative model (OLMo-class) joins only after it is
   fine-tuned on those decisions and passes the same gate; used zero-shot it did not (results §3).
7. Run in **shadow mode**: the local route runs alongside the current route on live reviews; only
   the current route's output is shown. Compare both against the reviewer's final decisions.

### Stage 2 — Extract fields locally (months 3–6) · [OWNER]

8. Dates, areas, project IDs, owners and tenure: parsers plus an extractive span model over the
   located passage. Every value must appear verbatim in the source, checked by a validator that
   also rejects any number not present in the source.
9. Cross-document checks (start date vs crediting period, area across plan and shapefile, totals)
   in deterministic code.

### Stage 3 — Train task-specific models with the community (months 4–9) · [OWNER]

10. Train small task models (encoder for location and sufficiency; extractive model for fields;
    a small fully open generative model, OLMo-class, for wording findings) on **reviewer decisions
    from reviews whose evidence owners consented to training** (the default is that evidence does
    not train models), plus public methodology texts and synthetic cases.
11. Train on community or renewably powered research compute (partner to be confirmed). Publish
    model cards with data provenance, licence and energy used for training.
12. Inference stays on reviewers' laptops or a community server; models are small enough for a
    16 GB laptop.

### Stage 4 — Local by default (months 8–12) · [OWNER]

13. Promote each step through the ladder **shadow → side by side (reviewer sees both) → local
    default with logged fallback → local only**, one step at a time, each behind its quality gate.
14. The frontier route is off by default. It runs only when (a) the project's processing terms
    allow `external`, (b) the evidence owner has consented, and (c) the reviewer asks for it on a
    named item. Each call is logged with the item 3 fields and shown in the round report.

## Quality gates (set against human reviewers, on held-out projects)

A step moves up the ladder only when, on projects it was not trained on, it meets all of:

| Step | Gate (proposed thresholds; review leads and the registry workflow owner to set) |
|---|---|
| Evidence location | reviewer-cited location in top 3 for ≥ 90% of requirements, and not worse than the current route on the same reviews |
| Sufficiency | recall on reviewer-cited evidence ≥ 90%; false "yes" on unrelated passages ≤ 5%; catches ≥ [N]% of requirements a reviewer did not approve |
| Field extraction | ≥ 95% exact match on reviewer-checked values; 0 invented numbers |
| Cross-checks | 100% of deterministic checks reproducible; every generated finding cites its passages |
| Whole round | reviewer override rate no higher than the current route; reviewer time per requirement no higher |

Each gate run publishes: sample size and confidence interval, error types, time per record, and
energy per record.

## Energy and cost reporting

Per record and per round: wall time, whole-machine watts above idle (macOS telemetry now;
`powermetrics` or a metered plug for published numbers), tokens where a generative model runs, and
where it ran. Training energy reported from the cluster.

## Risks

- **Small labelled set.** One project so far. Mitigation: Stage 0 widens to all reusable reviews.
- **No real negatives.** Over-scoring can't be measured on all-approved reviews. Mitigation: item 2.
- **Consent for training.** Default is no training on evidence. Mitigation: train on decisions
  and public texts; seek explicit consent per data owner; synthetic data for the rest.
- **Scanned and image-heavy documents.** Local OCR (Tesseract is already the agent's fallback)
  before any model step.
- **Model family choice.** Slimmed-down versions of large commercial models carry those models'
  training-data and provenance questions. Prefer encoders and fully open families (weights, data
  and code published); record provenance for each.
