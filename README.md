# Small local models for registry review — evaluation harness

Can the AI steps in a carbon-registry evidence review run on small, fully local models instead of
a hosted frontier model? This repo holds a small harness that measures that against **human
reviewers' signed decisions**, plus the results of a first run and a roadmap.

- `ASSESSMENT.md` — hardware, runtimes, which review steps use a model, ground truth, plan
- `RESULTS.md` — what we measured, with error types, time and energy
- `ROADMAP.md` — stepwise path to every review step running locally or on community compute
- `results/` — aggregate scores only
- `harness/` — the code
- `synthetic/` — an invented example task, so the harness runs without any real review material

## The two tasks

1. **Evidence location.** Given a methodology requirement, rank a project document's sections or
   pages. Scored by hit@1 / hit@3 against the sections the human reviewer cited in the signed
   checklist.
2. **Evidence sufficiency.** Given a requirement and a passage, does the passage hold evidence for
   it? Scored on reviewer-cited passages (should say yes) and unrelated passages (should say no).
   This targets the "everything is covered" bias seen in an earlier large-model run.

## Approaches, smallest first

| Approach | Model | Params | Licence |
|---|---|---|---|
| Rules | BM25 keyword scoring | 0 | — |
| Compact encoder | nomic-embed-text-v1.5 | 137M | Apache-2.0 |
| Trained classifier | 5-weight logistic regression on encoder + keyword features, trained on reviewer decisions | 5 | — |
| Small fully open generative | OLMo-2 1B / 7B Instruct (Ai2) | 1.5B / 7.3B | Apache-2.0 |
| Reference only | gpt-oss-20b (local) | 21B MoE | Apache-2.0 |

All models run in `llama-server` on 127.0.0.1 inside an OS network-deny sandbox (`harness/serve.sh`).

## Run it on the synthetic example

```bash
python3 harness/rank.py --task synthetic/task-synthetic.json --method bm25 --out-dir synthetic/runs
```

With models (paths are yours):

```bash
harness/serve.sh /path/to/nomic-embed-text-v1.5.f16.gguf 8791 embed 2048
harness/serve.sh /path/to/OLMo-2-0425-1B-Instruct-F16.gguf 8792 chat 8192
python3 harness/rank.py  --task synthetic/task-synthetic.json --method hybrid --embed-port 8791 --out-dir synthetic/runs
python3 harness/judge.py --task synthetic/task-synthetic.json --chat-port 8792 --tag olmo2-1b
harness/serve.sh stop 8791; harness/serve.sh stop 8792
```

## Run it on a real past review (stays local)

`harness/prep.py` turns a signed reviewer checklist (.docx) and the project document into a task
file under `private/` (gitignored). Real review content, client names and per-item outputs never
leave `private/`. `harness/summarize.py` writes aggregate scores only.

Requirements: Python 3.11+ (standard library only), `pdftotext` (poppler) for PDF reports,
llama.cpp `llama-server`, macOS `sandbox-exec` for the network-deny sandbox.

## Data rules

- Commit code, synthetic fixtures and aggregate scores only.
- No real project evidence, client or person names, or review text.
- Model files are not committed; provenance (repo, revision, sha256) is recorded in the results.

## Licence

Apache-2.0 (see `LICENSE`). Model licences are listed in `results/MODELS.md`.
