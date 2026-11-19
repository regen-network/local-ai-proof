---
title: Local AI proof point — assessment
status: draft
access: public
---

# Local AI proof point — assessment

Can the Claims Engine's review steps run on small, fully local models? This memo covers what we
had to work with before testing. Results are in `RESULTS.md`.

## 1. Hardware and runtimes

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M5 Max |
| CPU | 18 cores (6 super + 12 performance) |
| GPU | 40-core, Metal 4 (unified memory) |
| Memory | 128 GB unified |
| llama.cpp | 0.6.0 (Homebrew), `llama-server` with Metal |
| Ollama | not installed (not needed) |
| MLX | not installed (installable with `uv`; not needed for this round) |
| Power measurement | `powermetrics` needs administrator rights, which this run did not have. macOS battery telemetry (`SystemLoad`) refreshes only every ~20+ s and the machine was shared with another workload, so energy is reported as upper bounds (see results). |

**Runtime choice: llama.cpp `llama-server`**, bound to 127.0.0.1 and run under an OS-level
network-deny sandbox (`harness/serve.sh`, profile `harness/llm-net-deny.sb`): the model process
cannot reach DNS or the internet, only loopback. This is the stricter of the two usual local
options (llama.cpp or MLX, network denied, loopback only).

Larger open-weight models already on the machine (gpt-oss-120b/20b, Gemma 4 31B) are slimmed-down
members of large commercial model families. They are used here **only as a reference point**,
never as the proposed answer. The proposed path favours encoders and fully open families whose
weights, training data and code are published.

## 2. Where the review agent uses a model

Source: the source code of Regen's Registry Review agent (an MCP server; private repository), read at
commit `ee27d86` (12 May 2026) from a read-only clone kept outside this repo.

| Step | How it runs today | Model? |
|---|---|---|
| Document discovery and type classification | filename patterns, file type, default | no |
| Requirement mapping (requirement → documents) | keyword/rules (`tools/mapping_tools.py`) | no |
| Evidence extraction (per requirement × mapped document) | the **whole document** (capped at ~80–150K characters) goes to a model, which returns snippets, confidence and a schema-match flag (`tools/evidence_tools.py`) | **yes** |
| Structured field extraction (dates, land tenure, project IDs) | model (`extractors/llm_extractors.py`), off by default (`llm_extraction_enabled=false`) | yes, optional |
| Unified analysis (one-shot over all documents) | model (`prompts/unified_analysis.py`, `tools/analyze_llm.py`) | yes |
| Cross-document validation synthesis | model (`validation/llm_synthesis.py`) | yes |
| Report generation | templates | no |
| Human review | reviewer confirms or removes each mapping | human |

Backends: `llm_backend` = auto / api (Anthropic) / cli (Claude Code) / openai. Anthropic models:
Haiku 4.5 in development, Sonnet 4.5 in production.

**Backends in practice.** The agent's changelog (v2.2.0–v2.5.0, April 2026) records end-to-end
runs on **open-weight models on a hosted endpoint** (GPT-OSS-120B, then a Gemma 4 31B swap test;
a Qwen trial failed on endpoint timeouts) through the `openai` backend. That backend reads the
standard `OPENAI_BASE_URL`, so the existing agent can already be pointed at a local
`llama-server` without code changes (not tested end to end here).

**Live reviews use Gemma 4 31B**, an open-weight model of about 31 billion parameters, on a
hosted endpoint through the `openai` backend. It is still a slimmed-down member of a large
commercial model family, which is the gap this work sets out to close.

## 3. Ground truth

| Source | What it holds | Use |
|---|---|---|
| Signed human reviewer checklists (.docx) for one past soil-carbon project: a **registration** review (23 requirements) and a **credit issuance** review (29 requirements) | For every requirement: the verdict, and the "Submitted Material" — the document sections, tables and figures the reviewer accepted as evidence | **Gold labels** for evidence location |
| The project's submitted documents (project plan, monitoring report) | the corpus the reviewer worked from | Test corpus |
| Review agent's evaluation baselines (`tests/evaluation/baselines/`, v2.2–v2.5) | the hosted 120B run's verdicts and page/section citations, matched to the same human checklist | Reference: large open-weight model |
| An April 2026 Claude-written review draft for the same project | per-requirement findings with section citations | Reference: Claude |
| The review agent's prior local review sessions and response cache | discovery and mapping only; evidence extraction never ran; cache empty | Not usable as ground truth |

Two cautions on ground truth:

- **Every human verdict in both checklists is "Approved".** Verdict agreement is therefore
  uninformative: the review agent's own baselines show 23/23 agreement for a model that marks
  everything "covered", and in its synthetic mixed-verdict fixture the same model marked all 23
  "covered", missing all three that a reviewer had not approved. So this round tests **where the
  evidence is** and **whether a passage really holds it**, not verdicts.
- **One project, ~46 labelled requirements.** Confidence intervals are wide. This is a proof of
  method, not a benchmark.

The Claude draft and the hosted-120B run both had the signed reviews available in their document
folder, so they are not clean comparisons. The local tests exclude those files.

## 4. Data handling

- Real review material stays on the reviewer's machine, under `private/` (gitignored). The harness
  reads it, local models process it on 127.0.0.1 inside the network-deny sandbox, and only
  **aggregate scores** leave `private/`.
- The repo carries code, a synthetic example task and aggregate results. No project evidence,
  client or person names, or review text.
- The hosted coding agent that built the harness read the checklist structure and cited section
  names to write the parser. That is the same exposure the review agent's hosted route already has
  for this material. For a stricter line, `prep.py`, `rank.py` and `judge.py` run without anyone
  reading content; only aggregates are printed.

Reuse of this past review is confirmed for evaluation, for publishing aggregate scores, and for
training the classifier on its reviewer decisions. The project is never named, and no evidence or
review text leaves `private/`.

## 5. Plan

**Tasks** (narrow, each a real step of evidence extraction):

1. **Evidence location** — given a requirement, rank the document's sections (registration: 37
   project-plan sections) or pages (issuance: 13 report pages); score hit@1 / hit@3 against the
   reviewer's cited locations.
2. **Evidence sufficiency** — for (requirement, passage) pairs, does the passage hold the
   evidence? Positives are the reviewer-cited passages; negatives are unrelated passages, plus
   the top keyword match the reviewer did not cite (ambiguous). Targets the over-scoring bias.

**Approaches, smallest first:** see `results/MODELS.md` for repository, revision, sha256, licence
and openness of each model.

| Step | Model | Size | Licence | Provenance |
|---|---|---|---|---|
| Rules | BM25 keyword scoring | 0 | — | code in `harness/common.py` |
| Compact encoder | `nomic-ai/nomic-embed-text-v1.5` (GGUF f16) | 137M params, 0.27 GB | Apache-2.0 | Nomic AI (US); training code and data released |
| Small fully open generative | `allenai/OLMo-2-0425-1B-Instruct` (GGUF F16) | 1.5B, 3.0 GB | Apache-2.0 | Ai2 (US); open weights, data, code, logs |
| Small fully open generative | `allenai/OLMo-2-1124-7B-Instruct` (GGUF F16) | 7.3B, 14.6 GB | Apache-2.0 | Ai2 (US); as above |
| Reference only | `gpt-oss-20b` MXFP4 | 21B MoE, 12 GB | Apache-2.0 | OpenAI (US); weights only |

- All files are the publisher's own release format (no re-quantisation), pinned to a Hugging Face
  revision and sha256-verified. About 18 GB in total.
- Model runs take minutes; downloads took ~70 minutes.
- Olmo 3 7B (Ai2, Nov 2025, Apache-2.0) has no publisher GGUF; it would need conversion or MLX.
  Next round.
