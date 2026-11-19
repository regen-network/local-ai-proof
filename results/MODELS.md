# Models used — provenance and licence

All files are the publisher's own release format (no re-quantisation by us), pinned to a Hugging
Face revision and checked against the published sha256. All ran in llama.cpp `llama-server`
0.6.0 (build 11429) on 127.0.0.1 under an OS network-deny sandbox, on one Apple M5 Max laptop
(128 GB unified memory).

| Role | Model | Publisher (origin) | Params | File | HF revision | sha256 | Licence | Openness |
|---|---|---|---|---|---|---|---|---|
| Rules baseline | BM25 (Okapi), `harness/common.py` | — | 0 | — | — | — | Apache-2.0 (this repo) | code |
| Compact encoder | nomic-embed-text-v1.5 | Nomic AI (US) | 137M | `nomic-embed-text-v1.5.f16.gguf` (0.27 GB) | `0188c9bf409793f810680a5a431e7b899c46104c` | `f7af6f66802f4df86eda10fe9bbcfc75c39562bed48ef6ace719a251cf1c2fdb` | Apache-2.0 | weights, training code and training data released |
| Trained classifier | logistic regression on encoder + keyword features | this repo | 5 weights | — | — | — | Apache-2.0 (this repo) | trained on reviewer decisions; weights in results |
| Small generative | OLMo-2-0425-1B-Instruct | Ai2 (US) | 1.5B | `OLMo-2-0425-1B-Instruct-F16.gguf` (2.97 GB) | `62f8c199538474c3e33ed5d7e0580abd66686a27` | `376249dff02917a8cf5ea413d2bfdf2802f10f7d92973d95a6f28b2204ac4eab` | Apache-2.0 | weights, data (OLMo-mix, Dolmino, Tülu 3), code, logs |
| Small generative | OLMo-2-1124-7B-Instruct | Ai2 (US) | 7.3B | `olmo-2-1124-7B-instruct-F16.gguf` (14.6 GB) | `410e0069f64869e4b1d17d8de04810b881fd824b` | `150ed8566f4648391d1990d68e256c4ba04c513167b4bd8a1c48665b67ee68a7` | Apache-2.0 | as above |
| Reference only | gpt-oss-20b | OpenAI (US) | 21B (3.6B active, MoE) | `gpt-oss-20b-MXFP4.gguf` (12.1 GB, ggml-org conversion) | `ef9b12f2ff56c69cf32153a02784e7a3c88bf524` | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` | Apache-2.0 | weights only |

Earlier large-model outputs scored for comparison (not re-run): a hosted GPT-OSS-120B run of the
review agent (April 2026, agent v2.2.0 and v2.5.0) and a Claude-written review draft (April 2026).
